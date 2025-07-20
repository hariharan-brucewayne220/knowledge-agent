"""
Start the KnowAgent web server
"""

import asyncio
import os
import uvicorn
from app import app, initialize_system

async def startup():
    """Initialize the system before starting the server"""
    await initialize_system()

if __name__ == "__main__":
    # Add ffmpeg to PATH
    ffmpeg_path = r"C:\Users\porul\Desktop\ffmpeg-master-latest-win64-gpl-shared\bin"
    if ffmpeg_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
    
    print("[STARTING] KnowAgent Web Interface...")
    print("[URL] Server will be available at: http://localhost:8080")
    print("\n[FEATURES]")
    print("- Claude-style UI with ChatGPT colors")
    print("- Enhanced KnowAgent with OpenAI integration")
    print("- Local Whisper transcription")
    print("- PDF upload and processing")
    print("- YouTube processing with topic classification")
    print("- Robust caching system")
    print("- Mobile-responsive design")
    print("[CONFIG] FFmpeg configured for audio processing")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Disable reload to avoid issues
        log_level="info"
    )