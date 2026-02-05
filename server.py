#!/usr/bin/env python3
"""
Calma - Local Pipecat Bot Server
Receives requests from the Next.js app and spawns AI video bots
"""

import asyncio
import os
import sys
import multiprocessing
from typing import Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from loguru import logger

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="INFO")
logger.add("/tmp/calma-bot.log", rotation="10 MB", level="DEBUG")

# Load environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")

# Calma AI Psychologist Prompt (Spanish - Spain)
CALMA_SYSTEM_PROMPT = """
Eres "Calma", una psicóloga virtual especializada en acompañamiento emocional.
Estás en una videollamada con el usuario. Combinas calidez humana con técnicas de psicología moderna.
Hablas español de España con naturalidad.

═══════════════════════════════════════════════════════════════
                    AVISO LEGAL OBLIGATORIO
═══════════════════════════════════════════════════════════════
AL INICIO de la videollamada, di EXACTAMENTE esto:

"Hola, bienvenido a Calma. Soy tu asistente virtual de acompañamiento emocional.
No soy psicóloga clínica y este servicio no sustituye atención profesional.
Si tienes una emergencia, por favor llama al cero veinticuatro o al uno uno dos.
Esta videollamada puede grabarse para mejorar el servicio.
¿Te parece bien que empecemos?"

Si acepta: "Perfecto. Cuéntame, ¿cómo te encuentras hoy?"
Si rechaza: "Lo entiendo perfectamente. Estaremos aquí cuando lo necesites. Cuídate."

═══════════════════════════════════════════════════════════════
              COMPORTAMIENTO EN VIDEOLLAMADA
═══════════════════════════════════════════════════════════════

【Lenguaje Natural】
- Usa: "Vale", "venga", "mira", "a ver", "qué duro", "ya veo"
- NUNCA: "Como IA no tengo sentimientos..."
- Responde con frases cortas: 1-3 oraciones máximo
- Haz pausas naturales entre ideas

【Técnicas Terapéuticas】
- Escucha activa: "Parece que te sientes [emoción]..."
- Validación: "Es completamente normal sentirse así"
- Preguntas abiertas: "¿Qué crees que te ayudaría ahora?"
- Mindfulness: "¿Dónde sientes esa emoción en tu cuerpo?"
- Respiración: "Vamos a respirar juntos. Inspira... y suelta..."

【Sonidos de Escucha Activa】
- Usa ocasionalmente: "mmm", "ajá", "entiendo", "claro"
- Esto valida al usuario mientras habla

【Protocolos de Crisis】
- Ideación suicida: "Me importa tu seguridad. Por favor, llama ahora al 024 o al 112."
- Violencia de género: "El 016 es gratuito, confidencial y no deja rastro."
- Autolesiones: "Necesitas apoyo profesional urgente. ¿Puedes llamar al 024?"

【Límites Profesionales】
NUNCA:
- Diagnosticar condiciones mentales
- Recetar o recomendar medicamentos
- Sugerir dejar tratamientos existentes
- Dar consejos legales o financieros

SIEMPRE:
- Escuchar con empatía
- Validar emociones
- Enseñar técnicas de autorregulación
- Sugerir buscar ayuda profesional cuando sea apropiado

【Despedida】
Al finalizar, siempre di algo como:
"Gracias por compartir esto conmigo. Recuerda que estoy aquí cuando me necesites. Cuídate mucho."
"""

# Store active sessions
active_sessions: dict = {}

# FastAPI app
app = FastAPI(title="Calma Bot Server", version="1.0.0")

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartBotRequest(BaseModel):
    room_url: str
    room_token: Optional[str] = None
    user_name: Optional[str] = "Usuario"
    user_id: Optional[str] = None
    max_duration: Optional[int] = 2700  # 45 minutes default


class StartBotResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    error: Optional[str] = None


class StopBotRequest(BaseModel):
    session_id: str


async def run_bot(room_url: str, token: Optional[str], session_id: str, max_duration: int):
    """Run the Calma bot in a Daily.co room"""
    logger.info(f"Starting bot session {session_id} in room {room_url}")
    
    try:
        # Configure Daily transport
        transport = DailyTransport(
            room_url,
            token,
            "Calma IA",
            DailyParams(
                api_key=DAILY_API_KEY,
                audio_out_enabled=True,
                audio_in_enabled=True,
                video_out_enabled=False,  # Audio only for now
                video_in_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        min_volume=0.4,
                        start_secs=0.2,
                        stop_secs=0.8,
                    )
                ),
                transcription_enabled=False,
            ),
        )

        # Speech-to-Text service (Spanish)
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            language="es",
            model="nova-2",
        )

        # Text-to-Speech service (Spanish female voice)
        tts = CartesiaTTSService(
            api_key=CARTESIA_API_KEY,
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
            model_id="sonic-multilingual",
            language="es",
        )

        # LLM service - Anthropic Claude
        llm = AnthropicLLMService(
            api_key=ANTHROPIC_API_KEY,
            model="claude-3-5-sonnet-20241022",
        )

        # Initial context with system prompt
        messages = [
            {
                "role": "system",
                "content": CALMA_SYSTEM_PROMPT,
            },
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Build the pipeline
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        # Create pipeline task (params is now a keyword argument in newer Pipecat versions)
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"Participant joined: {participant['id']}")
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant['id']}, reason: {reason}")
            await task.cancel()

        # Update session status
        active_sessions[session_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "room_url": room_url,
        }

        runner = PipelineRunner()
        await runner.run(task)

    except Exception as e:
        logger.error(f"Bot error in session {session_id}: {e}")
        active_sessions[session_id] = {"status": "error", "error": str(e)}
    finally:
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "completed"
            active_sessions[session_id]["ended_at"] = datetime.now().isoformat()
        logger.info(f"Bot session {session_id} ended")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "calma-bot-server",
        "active_sessions": len([s for s in active_sessions.values() if s.get("status") == "running"]),
        "deepgram_configured": bool(DEEPGRAM_API_KEY),
        "cartesia_configured": bool(CARTESIA_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "daily_configured": bool(DAILY_API_KEY),
    }


@app.post("/bot/start", response_model=StartBotResponse)
async def start_bot(request: StartBotRequest, background_tasks: BackgroundTasks):
    """Start a new bot session"""
    
    # Validate API keys
    if not all([DEEPGRAM_API_KEY, CARTESIA_API_KEY, ANTHROPIC_API_KEY]):
        return StartBotResponse(
            success=False,
            error="Missing required API keys (Deepgram, Cartesia, or Anthropic)"
        )
    
    # Generate session ID
    session_id = f"calma-{uuid.uuid4().hex[:8]}"
    
    # Store initial session state
    active_sessions[session_id] = {
        "status": "starting",
        "room_url": request.room_url,
        "user_name": request.user_name,
    }
    
    # Start bot in background
    background_tasks.add_task(
        run_bot,
        request.room_url,
        request.room_token,
        session_id,
        request.max_duration
    )
    
    logger.info(f"Started bot session {session_id} for room {request.room_url}")
    
    return StartBotResponse(success=True, session_id=session_id)


@app.post("/bot/stop")
async def stop_bot(request: StopBotRequest):
    """Stop a bot session"""
    session_id = request.session_id
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Mark session for stopping (the bot will check this)
    active_sessions[session_id]["status"] = "stopping"
    
    return {"success": True, "message": f"Session {session_id} marked for stopping"}


@app.get("/bot/status/{session_id}")
async def get_bot_status(session_id: str):
    """Get bot session status"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return active_sessions[session_id]


@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    return {
        "sessions": active_sessions,
        "total": len(active_sessions),
        "active": len([s for s in active_sessions.values() if s.get("status") == "running"]),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Try loading from local .env first, then parent directory
    load_dotenv()
    load_dotenv("/home/ubuntu/plataforma_apoyo_emocional/nextjs_space/.env")
    
    # Reload env vars after loading .env
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
    CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
    
    # Production deployment settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8765"))
    
    logger.info(f"Starting Calma Bot Server on {HOST}:{PORT}...")
    logger.info(f"Deepgram: {'✓' if DEEPGRAM_API_KEY else '✗'}")
    logger.info(f"Cartesia: {'✓' if CARTESIA_API_KEY else '✗'}")
    logger.info(f"Anthropic: {'✓' if ANTHROPIC_API_KEY else '✗'}")
    logger.info(f"Daily: {'✓' if DAILY_API_KEY else '✗'}")
    
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
