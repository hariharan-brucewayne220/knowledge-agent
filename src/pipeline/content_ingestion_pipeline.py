#!/usr/bin/env python3
"""
Real-Time Content Ingestion Pipeline
Automatically monitors and processes new content from multiple sources
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing components
import sys
sys.path.append('..')
from storage.unified_content_store import UnifiedContentStore
from agents.youtube_agent import YouTubeAgent
from agents.pdf_agent import PDFAgent

logger = logging.getLogger(__name__)

class ContentSourceType(Enum):
    PDF_DIRECTORY = "pdf_directory"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    YOUTUBE_CHANNEL = "youtube_channel"
    URL_MONITOR = "url_monitor"

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ContentItem:
    """Represents a piece of content to be processed"""
    id: str
    source_type: ContentSourceType
    source_path: str  # File path, URL, etc.
    title: str
    discovered_at: float
    status: ProcessingStatus = ProcessingStatus.PENDING
    processed_at: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PipelineStats:
    """Pipeline statistics and performance metrics"""
    total_discovered: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    processing_time_avg: float = 0.0
    last_discovery: Optional[float] = None
    last_processing: Optional[float] = None
    uptime_start: float = time.time()
    
    @property
    def success_rate(self) -> float:
        if self.total_discovered == 0:
            return 0.0
        return (self.total_processed / self.total_discovered) * 100
    
    @property
    def uptime_hours(self) -> float:
        return (time.time() - self.uptime_start) / 3600

class ContentIngestionPipeline:
    """
    Real-time content ingestion pipeline that monitors multiple sources
    and automatically processes new content as it becomes available.
    """
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        self.config_path = config_path
        self.content_store = UnifiedContentStore()
        self.youtube_agent = YouTubeAgent()
        self.pdf_agent = PDFAgent()
        
        # Pipeline state
        self.is_running = False
        self.monitored_sources: Dict[str, Dict] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processed_items: Dict[str, ContentItem] = {}
        self.stats = PipelineStats()
        
        # Processing configuration
        self.max_concurrent_processors = 3
        self.discovery_interval = 30  # seconds
        self.retry_failed_after = 3600  # 1 hour
        
        # Load configuration
        self.load_configuration()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _resolve_env_variables(self, value):
        """Resolve environment variables in configuration values"""
        if isinstance(value, str) and value.startswith("ENV:"):
            env_var = value[4:]  # Remove "ENV:" prefix
            return os.getenv(env_var)
        elif isinstance(value, dict):
            return {k: self._resolve_env_variables(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_variables(v) for v in value]
        return value

    def load_configuration(self):
        """Load pipeline configuration from file"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Resolve environment variables
                config = self._resolve_env_variables(config)
                    
                self.monitored_sources = config.get('monitored_sources', {})
                self.max_concurrent_processors = config.get('max_concurrent_processors', 3)
                self.discovery_interval = config.get('discovery_interval', 30)
                self.retry_failed_after = config.get('retry_failed_after', 3600)
                
                # YouTube monitoring configuration
                self.youtube_config = config.get('youtube_monitoring', {
                    'enabled': False,
                    'check_strategy': 'startup_only',
                    'check_interval_hours': 24,
                    'last_youtube_check': None
                })
                
                logger.info(f"Loaded configuration: {len(self.monitored_sources)} sources")
                
                # Log YouTube API availability
                if self.youtube_config.get('enabled') and self.youtube_config.get('api_key'):
                    logger.info("📺 YouTube API key configured")
                elif self.youtube_config.get('enabled'):
                    logger.warning("📺 YouTube monitoring enabled but no API key found")
                    
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        else:
            # Create default configuration
            self.create_default_configuration()
    
    def create_default_configuration(self):
        """Create default pipeline configuration"""
        default_config = {
            "monitored_sources": {
                "pdf_watch_folder": {
                    "type": "pdf_directory",
                    "path": "watched_pdfs/",
                    "enabled": True,
                    "recursive": True,
                    "file_patterns": ["*.pdf"]
                },
                "research_videos": {
                    "type": "youtube_playlist",
                    "url": "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID",
                    "enabled": False,
                    "check_interval": 3600
                }
            },
            "max_concurrent_processors": 3,
            "discovery_interval": 30,
            "retry_failed_after": 3600,
            "processing_settings": {
                "pdf_chunk_size": 1000,
                "video_segment_duration": 30,
                "enable_topic_classification": True,
                "create_embeddings": True
            }
        }
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration at {self.config_path}")
    
    def add_monitored_source(self, name: str, source_config: Dict[str, Any]):
        """Add a new source to monitor"""
        self.monitored_sources[name] = source_config
        self.save_configuration()
        logger.info(f"Added monitored source: {name}")
    
    def remove_monitored_source(self, name: str):
        """Remove a monitored source"""
        if name in self.monitored_sources:
            del self.monitored_sources[name]
            self.save_configuration()
            logger.info(f"Removed monitored source: {name}")
    
    def save_configuration(self):
        """Save current configuration to file"""
        config = {
            "monitored_sources": self.monitored_sources,
            "max_concurrent_processors": self.max_concurrent_processors,
            "discovery_interval": self.discovery_interval,
            "retry_failed_after": self.retry_failed_after
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def start_pipeline(self):
        """Start the content ingestion pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Content Ingestion Pipeline")
        
        # Check if YouTube monitoring is needed on startup
        await self._check_youtube_on_startup()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._processing_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self.is_running = False
    
    async def stop_pipeline(self):
        """Stop the content ingestion pipeline"""
        logger.info("⏹️ Stopping Content Ingestion Pipeline")
        self.is_running = False
    
    async def _check_youtube_on_startup(self):
        """Check if YouTube monitoring is needed on server startup"""
        if not self.youtube_config.get('enabled', False):
            return
        
        # Check last monitoring time
        last_check = self.youtube_config.get('last_youtube_check')
        check_interval_hours = self.youtube_config.get('check_interval_hours', 24)
        current_time = time.time()
        
        # If never checked or more than X hours ago
        if (last_check is None or 
            (current_time - last_check) > (check_interval_hours * 3600)):
            
            logger.info(f"📺 YouTube check needed (last check: {'never' if last_check is None else f'{(current_time - last_check)/3600:.1f}h ago'})")
            
            try:
                await self._perform_youtube_check()
                
                # Update last check time in config
                self.youtube_config['last_youtube_check'] = current_time
                self._save_youtube_check_time(current_time)
                
            except Exception as e:
                logger.error(f"YouTube monitoring failed: {e}")
        else:
            logger.info(f"📺 YouTube check not needed (last check: {(current_time - last_check)/3600:.1f}h ago)")
    
    async def _perform_youtube_check(self):
        """Perform YouTube monitoring check"""
        # This would implement the actual YouTube API checking
        # For now, just log that it would happen
        logger.info("📺 Would check YouTube channels/playlists for new videos here")
        logger.info("📺 (YouTube API integration can be enabled when API key is provided)")
        
    def _save_youtube_check_time(self, timestamp: float):
        """Save the last YouTube check time to config file"""
        try:
            # Update the config file with new timestamp
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if 'youtube_monitoring' not in config:
                    config['youtube_monitoring'] = {}
                
                config['youtube_monitoring']['last_youtube_check'] = timestamp
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save YouTube check time: {e}")
    
    async def _discovery_loop(self):
        """Main discovery loop - finds new content from monitored sources"""
        logger.info("🔍 Started content discovery loop")
        
        while self.is_running:
            try:
                discovered_count = 0
                
                for source_name, source_config in self.monitored_sources.items():
                    if not source_config.get('enabled', True):
                        continue
                    
                    source_type = ContentSourceType(source_config['type'])
                    
                    if source_type == ContentSourceType.PDF_DIRECTORY:
                        discovered_count += await self._discover_pdf_files(source_name, source_config)
                    elif source_type == ContentSourceType.YOUTUBE_PLAYLIST:
                        discovered_count += await self._discover_youtube_playlist(source_name, source_config)
                    elif source_type == ContentSourceType.YOUTUBE_CHANNEL:
                        discovered_count += await self._discover_youtube_channel(source_name, source_config)
                
                if discovered_count > 0:
                    self.stats.last_discovery = time.time()
                    logger.info(f"📄 Discovered {discovered_count} new items")
                
                # Wait before next discovery cycle
                await asyncio.sleep(self.discovery_interval)
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying
    
    async def _processing_loop(self):
        """Main processing loop - processes discovered content"""
        logger.info("⚙️ Started content processing loop")
        
        # Create processor pool
        semaphore = asyncio.Semaphore(self.max_concurrent_processors)
        
        while self.is_running:
            try:
                # Get item from queue (with timeout to allow checking if still running)
                try:
                    item = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process item with concurrency control
                await self._process_content_item(item, semaphore)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_loop(self):
        """Cleanup loop - retries failed items and maintains pipeline health"""
        logger.info("🧹 Started pipeline cleanup loop")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Retry failed items after configured delay
                retry_items = []
                for item_id, item in self.processed_items.items():
                    if (item.status == ProcessingStatus.FAILED and 
                        item.processed_at and 
                        current_time - item.processed_at > self.retry_failed_after):
                        retry_items.append(item)
                
                for item in retry_items:
                    item.status = ProcessingStatus.PENDING
                    item.error_message = None
                    await self.processing_queue.put(item)
                    logger.info(f"🔄 Retrying failed item: {item.id}")
                
                # Clean up old completed items (keep only recent ones)
                cutoff_time = current_time - (7 * 24 * 3600)  # 7 days
                cleanup_ids = [
                    item_id for item_id, item in self.processed_items.items()
                    if (item.status == ProcessingStatus.COMPLETED and
                        item.processed_at and 
                        item.processed_at < cutoff_time)
                ]
                
                for item_id in cleanup_ids:
                    del self.processed_items[item_id]
                
                if cleanup_ids:
                    logger.info(f"🗑️ Cleaned up {len(cleanup_ids)} old processed items")
                
                # Wait before next cleanup cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _discover_pdf_files(self, source_name: str, config: Dict) -> int:
        """Discover new PDF files in monitored directories"""
        try:
            watch_path = Path(config['path'])
            if not watch_path.exists():
                watch_path.mkdir(parents=True, exist_ok=True)
                return 0
            
            file_patterns = config.get('file_patterns', ['*.pdf'])
            recursive = config.get('recursive', True)
            
            discovered_files = []
            
            for pattern in file_patterns:
                if recursive:
                    discovered_files.extend(watch_path.rglob(pattern))
                else:
                    discovered_files.extend(watch_path.glob(pattern))
            
            new_items = 0
            for file_path in discovered_files:
                # Generate unique ID for this file
                item_id = f"pdf_{file_path.stem}_{int(file_path.stat().st_mtime)}"
                
                # Skip if already processed
                if item_id in self.processed_items:
                    continue
                
                # Create content item
                content_item = ContentItem(
                    id=item_id,
                    source_type=ContentSourceType.PDF_DIRECTORY,
                    source_path=str(file_path),
                    title=file_path.stem,
                    discovered_at=time.time(),
                    metadata={
                        'file_size': file_path.stat().st_size,
                        'modified_at': file_path.stat().st_mtime,
                        'source_name': source_name
                    }
                )
                
                # Add to processing queue
                await self.processing_queue.put(content_item)
                self.processed_items[item_id] = content_item
                new_items += 1
                self.stats.total_discovered += 1
            
            return new_items
            
        except Exception as e:
            logger.error(f"Error discovering PDF files in {source_name}: {e}")
            return 0
    
    async def _discover_youtube_playlist(self, source_name: str, config: Dict) -> int:
        """Discover new videos in YouTube playlists"""
        # This would implement YouTube playlist monitoring
        # For now, return 0 (placeholder implementation)
        logger.debug(f"YouTube playlist discovery not yet implemented for {source_name}")
        return 0
    
    async def _discover_youtube_channel(self, source_name: str, config: Dict) -> int:
        """Discover new videos in YouTube channels"""
        # This would implement YouTube channel monitoring  
        # For now, return 0 (placeholder implementation)
        logger.debug(f"YouTube channel discovery not yet implemented for {source_name}")
        return 0
    
    async def _process_content_item(self, item: ContentItem, semaphore: asyncio.Semaphore):
        """Process a single content item"""
        async with semaphore:
            item.status = ProcessingStatus.PROCESSING
            start_time = time.time()
            
            try:
                logger.info(f"⚙️ Processing {item.source_type.value}: {item.title}")
                
                if item.source_type == ContentSourceType.PDF_DIRECTORY:
                    await self._process_pdf_item(item)
                elif item.source_type in [ContentSourceType.YOUTUBE_PLAYLIST, ContentSourceType.YOUTUBE_CHANNEL]:
                    await self._process_youtube_item(item)
                
                # Mark as completed
                item.status = ProcessingStatus.COMPLETED
                item.processed_at = time.time()
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats.total_processed += 1
                self.stats.last_processing = time.time()
                
                # Update average processing time
                if self.stats.processing_time_avg == 0:
                    self.stats.processing_time_avg = processing_time
                else:
                    self.stats.processing_time_avg = (
                        (self.stats.processing_time_avg * (self.stats.total_processed - 1) + processing_time) /
                        self.stats.total_processed
                    )
                
                logger.info(f"✅ Completed processing {item.title} in {processing_time:.1f}s")
                
            except Exception as e:
                item.status = ProcessingStatus.FAILED
                item.processed_at = time.time()
                item.error_message = str(e)
                
                self.stats.total_failed += 1
                logger.error(f"❌ Failed to process {item.title}: {e}")
    
    async def _process_pdf_item(self, item: ContentItem):
        """Process a PDF file"""
        file_path = Path(item.source_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Extract text from PDF
        extract_result = await self.pdf_agent.execute_action("extract_text", str(file_path))
        if not extract_result.success:
            raise Exception(f"Failed to extract text: {extract_result.error_message}")
        
        # Chunk the document
        chunk_result = await self.pdf_agent.execute_action(
            "chunk_document", 
            str(file_path),
            previous_results={"extract_text": extract_result.output}
        )
        if not chunk_result.success:
            raise Exception(f"Failed to chunk document: {chunk_result.error_message}")
        
        # Create embeddings
        embedding_result = await self.pdf_agent.execute_action(
            "create_embeddings",
            str(file_path), 
            previous_results={"chunk_document": chunk_result.output}
        )
        if not embedding_result.success:
            raise Exception(f"Failed to create embeddings: {embedding_result.error_message}")
        
        # Store in content store
        chunks = chunk_result.output.get("chunks", [])
        metadata = {
            'title': item.title,
            'file_path': str(file_path),
            'processing_pipeline': True,
            'discovered_at': item.discovered_at,
            **item.metadata
        }
        
        content_id = self.content_store.add_pdf_content(
            pdf_path=str(file_path),
            title=item.title,
            chunks=chunks,
            metadata=metadata
        )
        
        item.metadata['content_id'] = content_id
    
    async def _process_youtube_item(self, item: ContentItem):
        """Process a YouTube video"""
        # This would implement YouTube video processing
        # Using existing YouTube agent functionality
        logger.debug(f"YouTube processing not yet fully implemented for {item.title}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics"""
        return {
            'is_running': self.is_running,
            'monitored_sources': len(self.monitored_sources),
            'queue_size': self.processing_queue.qsize(),
            'processed_items': len(self.processed_items),
            'stats': asdict(self.stats),
            'recent_items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'status': item.status.value,
                    'discovered_at': item.discovered_at,
                    'processed_at': item.processed_at,
                    'error_message': item.error_message
                }
                for item in sorted(
                    self.processed_items.values(),
                    key=lambda x: x.discovered_at,
                    reverse=True
                )[:10]  # Last 10 items
            ]
        }
    
    def get_source_configs(self) -> Dict[str, Any]:
        """Get current source configurations"""
        return dict(self.monitored_sources)

# Global pipeline instance
_pipeline_instance: Optional[ContentIngestionPipeline] = None

def get_pipeline() -> ContentIngestionPipeline:
    """Get global pipeline instance (singleton pattern)"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ContentIngestionPipeline()
    return _pipeline_instance

async def main():
    """Main function for testing the pipeline"""
    pipeline = ContentIngestionPipeline()
    
    # Add a test PDF directory
    pipeline.add_monitored_source("test_pdfs", {
        "type": "pdf_directory",
        "path": "test_content/",
        "enabled": True,
        "recursive": True,
        "file_patterns": ["*.pdf"]
    })
    
    print("🚀 Starting Content Ingestion Pipeline Test")
    
    try:
        await pipeline.start_pipeline()
    except KeyboardInterrupt:
        print("⏹️ Pipeline stopped by user")
        await pipeline.stop_pipeline()

if __name__ == "__main__":
    asyncio.run(main())