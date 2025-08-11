#!/usr/bin/env python3
"""
Content Ingestion Pipeline Daemon
Runs the discovery and processing loops continuously in the background
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from pipeline.content_ingestion_pipeline import ContentIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class PipelineDaemon:
    def __init__(self):
        self.pipeline = ContentIngestionPipeline()
        self.running = False
        
    async def start(self):
        """Start the pipeline daemon"""
        self.running = True
        logger.info("Starting Content Ingestion Pipeline Daemon")
        
        try:
            # Setup signal handlers for graceful shutdown (Unix only)
            try:
                loop = asyncio.get_event_loop()
                for sig in [signal.SIGINT, signal.SIGTERM]:
                    loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                # Windows doesn't support signal handlers in asyncio
                logger.info("Signal handlers not supported on this platform")
            
            # Start the pipeline (discovery, processing, and cleanup loops)
            await self.pipeline.start_pipeline()
        except Exception as e:
            logger.error(f"Pipeline daemon error: {e}")
        finally:
            self.running = False
            logger.info("Pipeline daemon stopped")
    
    def stop(self):
        """Stop the pipeline daemon"""
        logger.info("Received stop signal - shutting down pipeline daemon")
        self.running = False
        asyncio.create_task(self.pipeline.stop_pipeline())

async def main():
    """Main function"""
    daemon = PipelineDaemon()
    
    try:
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
    except Exception as e:
        logger.error(f"Daemon failed: {e}")

if __name__ == "__main__":
    print("Content Ingestion Pipeline Daemon")
    print("=================================")
    print("This daemon will continuously monitor watched directories")
    print("and automatically process new PDF files.")
    print("Press Ctrl+C to stop.")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDaemon stopped by user")
    except Exception as e:
        print(f"Daemon failed: {e}")