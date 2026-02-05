"""
Calma Bot Server - Simplified version
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Calma Bot Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "deepgram_configured": bool(os.getenv("DEEPGRAM_API_KEY")),
        "cartesia_configured": bool(os.getenv("CARTESIA_API_KEY")),
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "daily_configured": bool(os.getenv("DAILY_API_KEY")),
    }

@app.post("/bot/start")
async def start_bot():
    return {"error": "Bot temporalmente no disponible", "status": "unavailable"}

@app.post("/bot/stop")
async def stop_bot():
    return {"success": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)
