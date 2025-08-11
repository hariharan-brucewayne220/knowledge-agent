#!/usr/bin/env python3
"""
Start Pipeline Daemon Script
Simple script to start the content ingestion pipeline daemon
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_environment():
    """Check if environment is ready"""
    print("Checking environment...")
    
    # Check virtual environment
    if sys.prefix != sys.base_prefix:
        print(f"[OK] Virtual environment active: {sys.prefix}")
    else:
        print("[WARNING] Virtual environment not detected")
    
    # Check required directories
    required_dirs = ["watched_pdfs", "incoming_documents", "src/pipeline"]
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
        print("[OK] Pipeline configuration found")
    else:
        print("[WARNING] Pipeline configuration not found")
    
    print()

def main():
    print("CONTENT INGESTION PIPELINE DAEMON")
    print("=" * 40)
    
    check_environment()
    
    print("Starting pipeline daemon...")
    print("The daemon will:")
    print("• Monitor watched_pdfs/ directory every 30 seconds")
    print("• Automatically process new PDF files")
    print("• Run continuously until stopped")
    print()
    print("Press Ctrl+C to stop the daemon")
    print("=" * 40)
    
    try:
        # Import and run the daemon
        import sys
        sys.path.append('.')
        
        from pipeline_daemon import main as daemon_main
        import asyncio
        
        asyncio.run(daemon_main())
        
    except KeyboardInterrupt:
        print("\n" + "=" * 40)
        print("Pipeline daemon stopped by user")
    except Exception as e:
        print(f"Error starting daemon: {e}")

if __name__ == "__main__":
    main()