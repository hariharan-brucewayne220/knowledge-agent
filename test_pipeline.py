#!/usr/bin/env python3
"""
Test script for the Real-Time Content Ingestion Pipeline
"""

import asyncio
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.pipeline import get_pipeline

async def test_pipeline():
    """Test the content ingestion pipeline"""
    print("🧪 Testing Real-Time Content Ingestion Pipeline")
    print("=" * 60)
    
    pipeline = get_pipeline()
    
    # Display current configuration
    print("\n📋 Current Configuration:")
    config = pipeline.get_source_configs()
    if config:
        for name, source_config in config.items():
            print(f"  📁 {name}: {source_config['type']} - {source_config.get('path', source_config.get('url', 'N/A'))}")
    else:
        print("  No sources configured")
    
    # Add a test source if none exist
    if not config:
        print("\n➕ Adding test PDF directory...")
        pipeline.add_monitored_source("test_pdfs", {
            "type": "pdf_directory",
            "path": "watched_pdfs/",
            "enabled": True,
            "recursive": True,
            "file_patterns": ["*.pdf"]
        })
    
    # Show status
    print("\n📊 Pipeline Status:")
    status = pipeline.get_pipeline_status()
    print(f"  Running: {status['is_running']}")
    print(f"  Sources: {status['monitored_sources']}")
    print(f"  Queue size: {status['queue_size']}")
    print(f"  Processed items: {status['processed_items']}")
    
    # Instructions for testing
    print("\n🔍 Testing Instructions:")
    print("1. The pipeline will monitor the following directories:")
    for name, source_config in pipeline.get_source_configs().items():
        if source_config['type'] == 'pdf_directory':
            path = Path(source_config['path'])
            path.mkdir(parents=True, exist_ok=True)
            print(f"   📂 {path} ({name})")
    
    print("\n2. To test the pipeline:")
    print("   - Copy PDF files to any of the watched directories")
    print("   - The pipeline will automatically detect and process them")
    print("   - Check the console for processing logs")
    
    print("\n3. API Endpoints (when FastAPI is running):")
    print("   GET  /api/pipeline/status     - Get pipeline status")
    print("   POST /api/pipeline/start      - Start pipeline")
    print("   POST /api/pipeline/stop       - Stop pipeline")
    print("   GET  /api/pipeline/sources    - List monitored sources")
    print("   POST /api/pipeline/sources/pdf - Add PDF directory")
    
    print("\n🚀 Starting pipeline for 60 seconds...")
    
    try:
        # Start pipeline for demonstration
        task = asyncio.create_task(pipeline.start_pipeline())
        
        # Let it run for 60 seconds
        await asyncio.sleep(60)
        
        print("\n⏹️ Stopping pipeline...")
        await pipeline.stop_pipeline()
        
        # Wait for task to complete
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Show final status
        print("\n📈 Final Statistics:")
        final_status = pipeline.get_pipeline_status()
        stats = final_status['stats']
        print(f"  Total discovered: {stats['total_discovered']}")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Total failed: {stats['total_failed']}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
        print(f"  Uptime: {stats.get('uptime_hours', 0):.2f} hours")
        
        if final_status['recent_items']:
            print("\n📄 Recent Items:")
            for item in final_status['recent_items'][:5]:
                print(f"  - {item['title']} ({item['status']})")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline stopped by user")
        await pipeline.stop_pipeline()
    
    print("\n✅ Pipeline test completed!")

async def test_api_integration():
    """Test API integration with FastAPI"""
    print("\n🌐 API Integration Test")
    print("=" * 40)
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test pipeline status endpoint
            try:
                response = await client.get(f"{base_url}/api/pipeline/status")
                if response.status_code == 200:
                    print("✅ Pipeline API is accessible")
                    data = response.json()
                    print(f"   Status: {'Running' if data['is_running'] else 'Stopped'}")
                    print(f"   Sources: {data['monitored_sources']}")
                else:
                    print(f"❌ API returned status {response.status_code}")
            except Exception as e:
                print(f"⚠️ API not accessible: {e}")
                print("   Make sure FastAPI server is running: python app.py")
    
    except ImportError:
        print("⚠️ httpx not available for API testing")
        print("   Install with: pip install httpx")

def create_test_pdf():
    """Create a simple test PDF for demonstration"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        test_file = Path("watched_pdfs/test_document.pdf")
        test_file.parent.mkdir(exist_ok=True)
        
        if not test_file.exists():
            c = canvas.Canvas(str(test_file), pagesize=letter)
            c.drawString(100, 750, "Test Document for Pipeline")
            c.drawString(100, 730, f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(100, 710, "This is a test PDF to demonstrate the")
            c.drawString(100, 690, "Real-Time Content Ingestion Pipeline.")
            c.save()
            print(f"📄 Created test PDF: {test_file}")
            return True
    except ImportError:
        print("⚠️ reportlab not available for PDF creation")
        print("   Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Failed to create test PDF: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 REAL-TIME CONTENT INGESTION PIPELINE TEST")
    print("=" * 80)
    
    # Create a test PDF
    create_test_pdf()
    
    # Test the pipeline
    await test_pipeline()
    
    # Test API integration
    await test_api_integration()
    
    print("\n" + "=" * 80)
    print("🎉 TESTING COMPLETE!")
    print("\nNext steps:")
    print("1. Start your FastAPI server: python app.py")
    print("2. Visit http://localhost:8000/api/pipeline/status")
    print("3. Drop PDF files in watched_pdfs/ to see real-time processing")
    print("4. Use the API endpoints to manage the pipeline")

if __name__ == "__main__":
    asyncio.run(main())