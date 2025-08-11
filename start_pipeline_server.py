#!/usr/bin/env python3
"""
Pipeline Server Startup Script
"""

import sys
import os
import time
from pathlib import Path

print("KNOWAGENT RESEARCH SERVER WITH PIPELINE")
print("=" * 50)

# Check virtual environment
print("[INFO] Checking environment...")
if sys.prefix != sys.base_prefix:
    print(f"[OK] Virtual environment active: {sys.prefix}")
else:
    print("[WARNING] Virtual environment not detected")
    print("         Consider using: myenv/Scripts/python.exe")

# Check required directories
required_dirs = ["watched_pdfs", "incoming_documents", "uploads", "src/pipeline"]
for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"[OK] Directory exists: {dir_name}")
    else:
        print(f"[WARNING] Directory missing: {dir_name}")
        if dir_name in ["watched_pdfs", "incoming_documents"]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"[CREATED] {dir_name}")
            except Exception as e:
                print(f"[ERROR] Could not create {dir_name}: {e}")

# Check configuration
config_path = Path("pipeline_config.json")
if config_path.exists():
    print(f"[OK] Pipeline configuration found")
else:
    print(f"[WARNING] Pipeline configuration not found")

# Check imports
print("\n[INFO] Checking imports...")
try:
    import fastapi
    print("[OK] FastAPI available")
except ImportError:
    print("[ERROR] FastAPI not installed")
    print("       Install with: pip install fastapi uvicorn")

try:
    # Add current directory to Python path
    import sys
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    if 'src' not in sys.path:
        sys.path.insert(0, 'src')
    
    from src.pipeline.content_ingestion_pipeline import ContentIngestionPipeline
    print("[OK] Pipeline modules available")
except ImportError as e:
    print(f"[ERROR] Pipeline import failed: {e}")
    print("       This is non-critical - server will still start")

print("\n[INFO] Starting server...")
print("=" * 30)
print("Server will be available at: http://localhost:8000")
print("API endpoints will be at: http://localhost:8000/api/...")
print("Pipeline API at: http://localhost:8000/api/pipeline/...")
print("")
print("Press Ctrl+C to stop the server")
print("=" * 30)

# Start the server
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("[ERROR] uvicorn not installed")
        print("       Install with: pip install uvicorn")
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped by user")
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")