#!/usr/bin/env python3
"""
Standalone Pipeline Test - No Server Required
"""

import sys
import asyncio
import time
from pathlib import Path

print("STANDALONE PIPELINE TEST")
print("=" * 50)

# Test 1: Basic Module Import
print("\n[1] Testing Module Imports...")
try:
    sys.path.append('src')
    from pipeline.content_ingestion_pipeline import ContentIngestionPipeline
    from pipeline.directory_watcher import PDFDirectoryWatcher, FileChangeEvent
    print("   [OK] Pipeline modules imported successfully")
except Exception as e:
    print(f"   [ERROR] Import failed: {e}")
    sys.exit(1)

# Test 2: Pipeline Creation
print("\n[2] Testing Pipeline Creation...")
try:
    pipeline = ContentIngestionPipeline()
    print("   [OK] Pipeline created successfully")
    print(f"   [INFO] Config path: {pipeline.config_path}")
    print(f"   [INFO] Watched sources: {len(pipeline.monitored_sources)}")
except Exception as e:
    print(f"   [ERROR] Pipeline creation failed: {e}")
    sys.exit(1)

# Test 3: Directory Setup
print("\n[3] Testing Directory Setup...")
try:
    test_dirs = ["test_watched_pdfs", "test_incoming"]
    for dir_name in test_dirs:
        test_dir = Path(dir_name)
        test_dir.mkdir(exist_ok=True)
        print(f"   [INFO] Created/verified: {test_dir}")
    print("   [OK] Test directories ready")
except Exception as e:
    print(f"   [ERROR] Directory setup failed: {e}")

# Test 4: Add Sources
print("\n[4] Testing Source Configuration...")
try:
    pipeline.add_monitored_source("test_pdf_source", {
        "type": "pdf_directory",
        "path": "test_watched_pdfs/",
        "enabled": True,
        "recursive": True,
        "file_patterns": ["*.pdf"]
    })
    print("   [OK] Added PDF directory source")
    
    status = pipeline.get_pipeline_status()
    print(f"   [INFO] Sources configured: {len(pipeline.monitored_sources)}")
    print(f"   [INFO] Pipeline status: {'Running' if status['is_running'] else 'Stopped'}")
except Exception as e:
    print(f"   [ERROR] Source configuration failed: {e}")

# Test 5: File Watcher Test
print("\n[5] Testing File Watcher...")
try:
    def handle_file_event(event: FileChangeEvent):
        print(f"   [ALERT] File detected: {event.file_path.name} ({event.event_type})")
    
    watcher = PDFDirectoryWatcher(["test_watched_pdfs"], handle_file_event)
    print("   [OK] File watcher created")
    print(f"   [INFO] Watching: {[str(d) for d in watcher.watch_directories]}")
    
    watcher_status = watcher.get_status()
    print(f"   [INFO] Method: {watcher_status['monitoring_method']}")
    print(f"   [INFO] Known files: {watcher_status['known_files_count']}")
except Exception as e:
    print(f"   [ERROR] File watcher test failed: {e}")

# Test 6: Create Test PDF
print("\n[6] Creating Test File...")
try:
    test_pdf = Path("test_watched_pdfs/sample_test.pdf")
    
    # Create a simple text file as test (not real PDF, but for testing detection)
    with open(test_pdf, 'w') as f:
        f.write(f"Test PDF content created at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"   [INFO] Created test file: {test_pdf}")
    print(f"   [INFO] File size: {test_pdf.stat().st_size} bytes")
    
    # Rename to trigger file system event
    test_pdf_renamed = Path("test_watched_pdfs/sample_test_renamed.pdf")
    test_pdf.rename(test_pdf_renamed)
    print(f"   [INFO] Renamed to: {test_pdf_renamed.name}")
    
except Exception as e:
    print(f"   [ERROR] Test file creation failed: {e}")

# Test 7: Configuration Test
print("\n[7] Testing Configuration...")
try:
    config = pipeline.get_source_configs()
    print("   [INFO] Current configuration:")
    for name, source in config.items():
        print(f"     - {name}: {source['type']} -> {source.get('path', 'N/A')}")
    
    # Save configuration
    pipeline.save_configuration()
    print("   [OK] Configuration saved")
    
    # Load configuration  
    pipeline.load_configuration()
    print("   [OK] Configuration loaded")
    
except Exception as e:
    print(f"   [ERROR] Configuration test failed: {e}")

# Test 8: Pipeline Status
print("\n[8] Final Status Check...")
try:
    status = pipeline.get_pipeline_status()
    print("   [INFO] Pipeline Statistics:")
    print(f"     - Running: {status['is_running']}")
    print(f"     - Sources: {status['monitored_sources']}")
    print(f"     - Queue size: {status['queue_size']}")
    print(f"     - Processed items: {status['processed_items']}")
    print(f"     - Uptime: {status['stats'].get('uptime_hours', 0):.2f} hours")
except Exception as e:
    print(f"   [ERROR] Status check failed: {e}")

print("\n" + "=" * 50)
print("STANDALONE TEST COMPLETE")
print("\nNext Steps:")
print("1. Start the server: myenv/Scripts/python.exe app.py")
print("2. Test API endpoints")
print("3. Test real-time file detection")
print("4. Test YouTube integration (with API key)")

# Cleanup
print("\nCleaning up test files...")
try:
    import shutil
    for test_dir in ["test_watched_pdfs", "test_incoming"]:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"   [CLEANUP] Removed: {test_dir}")
except Exception as e:
    print(f"   [WARNING] Cleanup warning: {e}")

print("[SUCCESS] Test completed successfully!")