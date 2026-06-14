"""Entry point for PneumoDetect AI FastAPI server.
Usage: python server.py
This script loads .env file, then starts uvicorn on port 8000.
"""
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()  # Load .env file before importing app modules
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )
