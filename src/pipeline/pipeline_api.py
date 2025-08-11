#!/usr/bin/env python3
"""
FastAPI integration for Content Ingestion Pipeline
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import logging
from pathlib import Path

from .content_ingestion_pipeline import get_pipeline, ContentSourceType
from .youtube_monitor import YouTubeMonitorConfig, parse_youtube_url

logger = logging.getLogger(__name__)

# Pydantic models for API
class PipelineStatus(BaseModel):
    is_running: bool
    monitored_sources: int
    queue_size: int
    processed_items: int
    stats: Dict[str, Any]
    recent_items: List[Dict[str, Any]]

class SourceConfig(BaseModel):
    name: str
    type: str
    path_or_url: str
    enabled: bool = True
    recursive: bool = True
    file_patterns: Optional[List[str]] = None
    check_interval: Optional[int] = None

class YouTubeSourceConfig(BaseModel):
    name: str
    url: str
    enabled: bool = True
    check_interval: int = 3600
    max_results: int = 50
    include_shorts: bool = True
    min_duration_seconds: int = 60
    keywords_filter: Optional[List[str]] = None
    exclude_keywords: Optional[List[str]] = None

# Create FastAPI router
pipeline_router = APIRouter(prefix="/api/pipeline", tags=["Content Pipeline"])

# Global pipeline management
pipeline_task: Optional[asyncio.Task] = None

@pipeline_router.get("/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get current pipeline status and statistics"""
    try:
        pipeline = get_pipeline()
        status = pipeline.get_pipeline_status()
        return PipelineStatus(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.post("/start")
async def start_pipeline(background_tasks: BackgroundTasks):
    """Start the content ingestion pipeline"""
    global pipeline_task
    
    try:
        pipeline = get_pipeline()
        
        if pipeline.is_running:
            return {"message": "Pipeline is already running", "status": "running"}
        
        # Start pipeline in background
        pipeline_task = asyncio.create_task(pipeline.start_pipeline())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        return {
            "message": "Content ingestion pipeline started successfully",
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@pipeline_router.post("/stop")
async def stop_pipeline():
    """Stop the content ingestion pipeline"""
    global pipeline_task
    
    try:
        pipeline = get_pipeline()
        
        if not pipeline.is_running:
            return {"message": "Pipeline is not running", "status": "stopped"}
        
        await pipeline.stop_pipeline()
        
        # Cancel background task
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass
        
        return {
            "message": "Content ingestion pipeline stopped successfully",
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline: {str(e)}")

@pipeline_router.get("/sources")
async def get_monitored_sources():
    """Get all monitored sources"""
    try:
        pipeline = get_pipeline()
        return {
            "sources": pipeline.get_source_configs(),
            "total": len(pipeline.monitored_sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.post("/sources/pdf")
async def add_pdf_source(source: SourceConfig):
    """Add a new PDF directory to monitor"""
    try:
        pipeline = get_pipeline()
        
        # Validate directory exists or can be created
        watch_path = Path(source.path_or_url)
        if not watch_path.exists():
            try:
                watch_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot create directory: {e}")
        
        # Create source configuration
        source_config = {
            "type": "pdf_directory",
            "path": source.path_or_url,
            "enabled": source.enabled,
            "recursive": source.recursive,
            "file_patterns": source.file_patterns or ["*.pdf"]
        }
        
        pipeline.add_monitored_source(source.name, source_config)
        
        return {
            "message": f"Added PDF source: {source.name}",
            "source": source_config
        }
        
    except Exception as e:
        logger.error(f"Failed to add PDF source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.post("/sources/youtube")
async def add_youtube_source(source: YouTubeSourceConfig):
    """Add a YouTube playlist or channel to monitor"""
    try:
        # Parse URL to determine type
        parsed = parse_youtube_url(source.url)
        
        if parsed['type'] not in ['playlist', 'channel']:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL. Must be a playlist or channel.")
        
        pipeline = get_pipeline()
        
        # Create source configuration
        source_config = {
            "type": f"youtube_{parsed['type']}",
            "url": source.url,
            "id": parsed['id'],
            "enabled": source.enabled,
            "check_interval": source.check_interval,
            "max_results": source.max_results,
            "include_shorts": source.include_shorts,
            "min_duration_seconds": source.min_duration_seconds,
            "keywords_filter": source.keywords_filter,
            "exclude_keywords": source.exclude_keywords
        }
        
        pipeline.add_monitored_source(source.name, source_config)
        
        return {
            "message": f"Added YouTube {parsed['type']}: {source.name}",
            "source": source_config,
            "parsed_info": parsed
        }
        
    except Exception as e:
        logger.error(f"Failed to add YouTube source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.delete("/sources/{source_name}")
async def remove_source(source_name: str):
    """Remove a monitored source"""
    try:
        pipeline = get_pipeline()
        
        if source_name not in pipeline.monitored_sources:
            raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")
        
        pipeline.remove_monitored_source(source_name)
        
        return {
            "message": f"Removed source: {source_name}",
            "status": "removed"
        }
        
    except Exception as e:
        logger.error(f"Failed to remove source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.put("/sources/{source_name}/toggle")
async def toggle_source(source_name: str):
    """Enable or disable a monitored source"""
    try:
        pipeline = get_pipeline()
        
        if source_name not in pipeline.monitored_sources:
            raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")
        
        current_status = pipeline.monitored_sources[source_name].get('enabled', True)
        new_status = not current_status
        
        pipeline.monitored_sources[source_name]['enabled'] = new_status
        pipeline.save_configuration()
        
        return {
            "message": f"Source '{source_name}' {'enabled' if new_status else 'disabled'}",
            "enabled": new_status
        }
        
    except Exception as e:
        logger.error(f"Failed to toggle source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.get("/config")
async def get_pipeline_config():
    """Get current pipeline configuration"""
    try:
        pipeline = get_pipeline()
        return {
            "config": {
                "max_concurrent_processors": pipeline.max_concurrent_processors,
                "discovery_interval": pipeline.discovery_interval,
                "retry_failed_after": pipeline.retry_failed_after,
                "config_path": pipeline.config_path
            },
            "sources": pipeline.get_source_configs()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.put("/config")
async def update_pipeline_config(config_data: Dict[str, Any]):
    """Update pipeline configuration"""
    try:
        pipeline = get_pipeline()
        
        # Update configurable parameters
        if 'max_concurrent_processors' in config_data:
            pipeline.max_concurrent_processors = max(1, min(10, config_data['max_concurrent_processors']))
        
        if 'discovery_interval' in config_data:
            pipeline.discovery_interval = max(10, config_data['discovery_interval'])
        
        if 'retry_failed_after' in config_data:
            pipeline.retry_failed_after = max(300, config_data['retry_failed_after'])
        
        # Save configuration
        pipeline.save_configuration()
        
        return {
            "message": "Pipeline configuration updated successfully",
            "config": {
                "max_concurrent_processors": pipeline.max_concurrent_processors,
                "discovery_interval": pipeline.discovery_interval,
                "retry_failed_after": pipeline.retry_failed_after
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.get("/recent-items")
async def get_recent_items(limit: int = 20):
    """Get recently processed items"""
    try:
        pipeline = get_pipeline()
        
        # Get recent items sorted by discovery time
        recent_items = sorted(
            pipeline.processed_items.values(),
            key=lambda x: x.discovered_at,
            reverse=True
        )[:limit]
        
        return {
            "items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "source_type": item.source_type.value,
                    "source_path": item.source_path,
                    "status": item.status.value,
                    "discovered_at": item.discovered_at,
                    "processed_at": item.processed_at,
                    "error_message": item.error_message,
                    "metadata": item.metadata
                }
                for item in recent_items
            ],
            "total_items": len(pipeline.processed_items)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.post("/process-folder")
async def trigger_folder_scan(folder_path: str, background_tasks: BackgroundTasks):
    """Manually trigger processing of all PDFs in a folder"""
    try:
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=400, detail="Invalid folder path")
        
        # Count PDF files
        pdf_files = list(folder.rglob("*.pdf"))
        
        if not pdf_files:
            return {
                "message": "No PDF files found in folder",
                "folder": str(folder),
                "files_found": 0
            }
        
        # Add as temporary source for immediate processing
        pipeline = get_pipeline()
        temp_source_name = f"manual_scan_{int(asyncio.get_event_loop().time())}"
        
        source_config = {
            "type": "pdf_directory",
            "path": str(folder),
            "enabled": True,
            "recursive": True,
            "file_patterns": ["*.pdf"]
        }
        
        pipeline.add_monitored_source(temp_source_name, source_config)
        
        # Schedule removal of temporary source after processing
        background_tasks.add_task(
            _remove_temp_source_after_delay,
            temp_source_name,
            300  # 5 minutes
        )
        
        return {
            "message": f"Started processing {len(pdf_files)} PDF files",
            "folder": str(folder),
            "files_found": len(pdf_files),
            "temp_source": temp_source_name
        }
        
    except Exception as e:
        logger.error(f"Failed to process folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _remove_temp_source_after_delay(source_name: str, delay_seconds: int):
    """Remove temporary source after delay"""
    await asyncio.sleep(delay_seconds)
    try:
        pipeline = get_pipeline()
        if source_name in pipeline.monitored_sources:
            pipeline.remove_monitored_source(source_name)
            logger.info(f"Removed temporary source: {source_name}")
    except Exception as e:
        logger.error(f"Failed to remove temporary source {source_name}: {e}")

@pipeline_router.get("/health")
async def pipeline_health_check():
    """Health check for the pipeline system"""
    try:
        pipeline = get_pipeline()
        status = pipeline.get_pipeline_status()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if status['stats']['total_failed'] > status['stats']['total_processed'] * 0.5:
            health_status = "unhealthy"
            issues.append("High failure rate")
        
        if not status['is_running'] and len(status['recent_items']) > 0:
            health_status = "warning"
            issues.append("Pipeline not running but has pending items")
        
        return {
            "status": health_status,
            "issues": issues,
            "pipeline_running": status['is_running'],
            "monitored_sources": status['monitored_sources'],
            "success_rate": status['stats'].get('success_rate', 0),
            "last_activity": status['stats'].get('last_processing'),
            "uptime_hours": status['stats'].get('uptime_hours', 0)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "issues": [str(e)],
            "pipeline_running": False,
            "monitored_sources": 0,
            "success_rate": 0
        }