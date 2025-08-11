#!/usr/bin/env python3
"""
Advanced Directory Watcher for Real-time PDF Detection
Uses OS-level file system events for immediate detection
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Create dummy classes for when watchdog is not available
    class FileSystemEventHandler:
        def __init__(self): pass
        def on_created(self, event): pass
        def on_moved(self, event): pass
        def on_modified(self, event): pass
    class Observer:
        def __init__(self): pass
        def schedule(self, handler, path, recursive=True): pass
        def start(self): pass
        def stop(self): pass
        def join(self): pass
    logging.warning("watchdog not installed - falling back to polling mode")

logger = logging.getLogger(__name__)

@dataclass
class FileChangeEvent:
    """Represents a file system change event"""
    file_path: Path
    event_type: str  # 'created', 'modified', 'moved'
    timestamp: float
    file_size: int
    is_pdf: bool

class PDFFileHandler(FileSystemEventHandler):
    """Watchdog event handler for PDF files"""
    
    def __init__(self, callback: Callable[[FileChangeEvent], None]):
        super().__init__()
        self.callback = callback
        self.processed_files: Set[str] = set()
        self.debounce_time = 1.0  # seconds to wait before processing
        self.pending_files: Dict[str, float] = {}
    
    def on_created(self, event):
        if not event.is_directory:
            self._handle_file_event(event.src_path, 'created')
    
    def on_moved(self, event):
        if not event.is_directory:
            self._handle_file_event(event.dest_path, 'moved')
    
    def on_modified(self, event):
        if not event.is_directory:
            self._handle_file_event(event.src_path, 'modified')
    
    def _handle_file_event(self, file_path: str, event_type: str):
        """Handle a file system event with debouncing"""
        path = Path(file_path)
        
        # Only process PDF files
        if not path.suffix.lower() == '.pdf':
            return
        
        # Skip if file is still being written (size = 0 or very small)
        try:
            if path.stat().st_size < 1024:  # Skip files smaller than 1KB
                return
        except (OSError, FileNotFoundError):
            return
        
        # Debouncing: wait a bit before processing to avoid processing partial files
        current_time = time.time()
        file_key = str(path)
        
        self.pending_files[file_key] = current_time
        
        # Schedule processing after debounce delay
        asyncio.create_task(self._process_after_delay(file_key, event_type, current_time))
    
    async def _process_after_delay(self, file_key: str, event_type: str, scheduled_time: float):
        """Process file after debounce delay"""
        await asyncio.sleep(self.debounce_time)
        
        # Check if this is still the most recent event for this file
        if file_key in self.pending_files and self.pending_files[file_key] == scheduled_time:
            path = Path(file_key)
            
            try:
                if path.exists() and path.stat().st_size > 1024:
                    event = FileChangeEvent(
                        file_path=path,
                        event_type=event_type,
                        timestamp=time.time(),
                        file_size=path.stat().st_size,
                        is_pdf=True
                    )
                    
                    # Call the callback
                    self.callback(event)
                    
            except Exception as e:
                logger.error(f"Error processing file event for {file_key}: {e}")
            finally:
                # Clean up pending files
                self.pending_files.pop(file_key, None)

class PDFDirectoryWatcher:
    """
    Advanced PDF directory watcher with real-time detection
    Supports both OS-level events (via watchdog) and polling fallback
    """
    
    def __init__(self, watch_directories: List[str], callback: Callable[[FileChangeEvent], None]):
        self.watch_directories = [Path(d) for d in watch_directories]
        self.callback = callback
        self.is_running = False
        
        # Watchdog components (if available)
        self.observer: Optional[Observer] = None
        self.handlers: List[PDFFileHandler] = []
        
        # Polling fallback
        self.poll_interval = 10.0  # seconds
        self.known_files: Dict[str, float] = {}  # path -> modification time
        
        # Ensure directories exist
        for directory in self.watch_directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Watching directory: {directory}")
    
    async def start_watching(self):
        """Start watching directories for new PDF files"""
        if self.is_running:
            logger.warning("Directory watcher is already running")
            return
        
        self.is_running = True
        logger.info(f"🔍 Starting PDF directory watcher for {len(self.watch_directories)} directories")
        
        # Initialize known files
        await self._scan_existing_files()
        
        if WATCHDOG_AVAILABLE:
            await self._start_watchdog_monitoring()
        else:
            await self._start_polling_monitoring()
    
    async def stop_watching(self):
        """Stop watching directories"""
        if not self.is_running:
            return
        
        logger.info("⏹️ Stopping PDF directory watcher")
        self.is_running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
    
    async def _scan_existing_files(self):
        """Scan existing files to establish baseline"""
        logger.info("📄 Scanning existing PDF files...")
        
        total_files = 0
        for directory in self.watch_directories:
            if not directory.exists():
                continue
            
            for pdf_file in directory.rglob("*.pdf"):
                try:
                    self.known_files[str(pdf_file)] = pdf_file.stat().st_mtime
                    total_files += 1
                except Exception as e:
                    logger.warning(f"Could not stat file {pdf_file}: {e}")
        
        logger.info(f"📊 Found {total_files} existing PDF files")
    
    async def _start_watchdog_monitoring(self):
        """Start real-time monitoring using watchdog"""
        logger.info("👁️ Starting real-time file system monitoring")
        
        try:
            self.observer = Observer()
            
            for directory in self.watch_directories:
                handler = PDFFileHandler(self._handle_file_change)
                self.handlers.append(handler)
                
                self.observer.schedule(handler, str(directory), recursive=True)
            
            self.observer.start()
            
            # Keep the monitoring alive
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Watchdog monitoring failed: {e}")
            # Fall back to polling
            await self._start_polling_monitoring()
    
    async def _start_polling_monitoring(self):
        """Fallback polling-based monitoring"""
        logger.info(f"📊 Starting polling-based monitoring (interval: {self.poll_interval}s)")
        
        while self.is_running:
            try:
                await self._poll_for_changes()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error during polling: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _poll_for_changes(self):
        """Poll directories for changes"""
        current_files = {}
        
        # Scan all directories
        for directory in self.watch_directories:
            if not directory.exists():
                continue
            
            for pdf_file in directory.rglob("*.pdf"):
                try:
                    file_path = str(pdf_file)
                    mtime = pdf_file.stat().st_mtime
                    current_files[file_path] = mtime
                    
                    # Check if this is a new or modified file
                    if (file_path not in self.known_files or 
                        self.known_files[file_path] < mtime):
                        
                        event_type = 'created' if file_path not in self.known_files else 'modified'
                        
                        event = FileChangeEvent(
                            file_path=pdf_file,
                            event_type=event_type,
                            timestamp=time.time(),
                            file_size=pdf_file.stat().st_size,
                            is_pdf=True
                        )
                        
                        self._handle_file_change(event)
                        
                except Exception as e:
                    logger.warning(f"Could not process file {pdf_file}: {e}")
        
        # Update known files
        self.known_files = current_files
    
    def _handle_file_change(self, event: FileChangeEvent):
        """Handle a detected file change"""
        logger.info(f"📁 Detected {event.event_type} PDF: {event.file_path.name} ({event.file_size:,} bytes)")
        
        try:
            # Call the callback function
            self.callback(event)
        except Exception as e:
            logger.error(f"Error in file change callback: {e}")
    
    def add_watch_directory(self, directory: str):
        """Add a new directory to watch"""
        new_dir = Path(directory)
        if new_dir not in self.watch_directories:
            new_dir.mkdir(parents=True, exist_ok=True)
            self.watch_directories.append(new_dir)
            
            # If already running, add to watchdog observer
            if self.is_running and self.observer and WATCHDOG_AVAILABLE:
                handler = PDFFileHandler(self._handle_file_change)
                self.handlers.append(handler)
                self.observer.schedule(handler, str(new_dir), recursive=True)
            
            logger.info(f"Added watch directory: {new_dir}")
    
    def remove_watch_directory(self, directory: str):
        """Remove a directory from watching"""
        remove_dir = Path(directory)
        if remove_dir in self.watch_directories:
            self.watch_directories.remove(remove_dir)
            logger.info(f"Removed watch directory: {remove_dir}")
    
    def get_status(self) -> Dict:
        """Get watcher status information"""
        return {
            'is_running': self.is_running,
            'watch_directories': [str(d) for d in self.watch_directories],
            'monitoring_method': 'watchdog' if WATCHDOG_AVAILABLE and self.observer else 'polling',
            'poll_interval': self.poll_interval,
            'known_files_count': len(self.known_files),
            'watchdog_available': WATCHDOG_AVAILABLE
        }

class BatchPDFProcessor:
    """
    Processes multiple PDF files efficiently with batching and error handling
    """
    
    def __init__(self, batch_size: int = 5, batch_timeout: float = 30.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_files: List[FileChangeEvent] = []
        self.last_batch_time = time.time()
        self.processing_callback: Optional[Callable] = None
    
    def set_processing_callback(self, callback: Callable[[List[FileChangeEvent]], None]):
        """Set callback function for processing batches of files"""
        self.processing_callback = callback
    
    async def add_file(self, event: FileChangeEvent):
        """Add file to processing batch"""
        self.pending_files.append(event)
        
        # Process batch if we hit size limit or timeout
        should_process = (
            len(self.pending_files) >= self.batch_size or
            (time.time() - self.last_batch_time) > self.batch_timeout
        )
        
        if should_process:
            await self._process_batch()
    
    async def _process_batch(self):
        """Process current batch of files"""
        if not self.pending_files or not self.processing_callback:
            return
        
        batch = self.pending_files.copy()
        self.pending_files.clear()
        self.last_batch_time = time.time()
        
        logger.info(f"📦 Processing batch of {len(batch)} PDF files")
        
        try:
            # Process batch (could be async)
            if asyncio.iscoroutinefunction(self.processing_callback):
                await self.processing_callback(batch)
            else:
                self.processing_callback(batch)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
    
    async def force_process_remaining(self):
        """Force process any remaining files in batch"""
        if self.pending_files:
            await self._process_batch()

# Example usage and testing
async def example_usage():
    """Example of how to use the PDF directory watcher"""
    
    def handle_pdf_change(event: FileChangeEvent):
        """Handle detected PDF file changes"""
        print(f"🆕 New PDF detected: {event.file_path.name}")
        print(f"   Event: {event.event_type}")
        print(f"   Size: {event.file_size:,} bytes")
        print(f"   Time: {datetime.fromtimestamp(event.timestamp)}")
    
    # Create watcher for multiple directories
    watcher = PDFDirectoryWatcher(
        watch_directories=[
            "watched_pdfs/",
            "incoming_documents/", 
            "research_papers/"
        ],
        callback=handle_pdf_change
    )
    
    print("🚀 Starting PDF Directory Watcher")
    print("Drop PDF files in the watched directories to see real-time detection!")
    print("Press Ctrl+C to stop")
    
    try:
        await watcher.start_watching()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping watcher...")
        await watcher.stop_watching()
        print("✅ Watcher stopped")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_usage())