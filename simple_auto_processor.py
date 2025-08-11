#!/usr/bin/env python3
"""
Simple Auto PDF Processor
Continuously monitors watched_pdfs/ and processes new files
"""

import os
import sys
import time
import asyncio
from pathlib import Path
import json

# Add src to path
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from agents.pdf_agent import PDFAgent

class SimpleAutoProcessor:
    def __init__(self):
        self.watch_dir = Path("watched_pdfs")
        self.processed_files = set()
        self.content_store = None
        self.pdf_agent = None
        
        # Load already processed files
        self._load_processed_files()
        
    def _load_processed_files(self):
        """Load list of already processed files"""
        try:
            if Path("unified_content/index.json").exists():
                with open("unified_content/index.json", 'r') as f:
                    data = json.load(f)
                    for item in data.get('content_items', []):
                        if 'watched_pdfs' in item.get('source_path', ''):
                            filename = Path(item['source_path']).name
                            self.processed_files.add(filename)
            print(f"Loaded {len(self.processed_files)} already processed files")
        except Exception as e:
            print(f"Error loading processed files: {e}")
    
    async def initialize_agents(self):
        """Initialize content store and PDF agent"""
        if self.content_store is None:
            self.content_store = UnifiedContentStore()
            self.pdf_agent = PDFAgent()
            print("Agents initialized")
    
    async def scan_and_process(self):
        """Scan directory and process new PDF files"""
        if not self.watch_dir.exists():
            print(f"Watch directory {self.watch_dir} does not exist")
            return
        
        # Find all PDF files
        pdf_files = list(self.watch_dir.glob("*.pdf"))
        new_files = []
        
        for pdf_file in pdf_files:
            if pdf_file.name not in self.processed_files:
                new_files.append(pdf_file)
        
        if new_files:
            print(f"Found {len(new_files)} new files to process:")
            for file in new_files:
                print(f"  - {file.name}")
            
            # Initialize agents if needed
            await self.initialize_agents()
            
            # Process each new file
            for pdf_file in new_files:
                try:
                    await self.process_pdf(pdf_file)
                    self.processed_files.add(pdf_file.name)
                    print(f"✓ Successfully processed: {pdf_file.name}")
                except Exception as e:
                    print(f"✗ Error processing {pdf_file.name}: {e}")
        
        return len(new_files)
    
    async def process_pdf(self, pdf_path):
        """Process a single PDF file"""
        print(f"Processing: {pdf_path.name}")
        
        # Extract text
        extract_result = await self.pdf_agent.execute_action('extract_text', str(pdf_path))
        if not extract_result.success:
            raise Exception(f"Text extraction failed: {extract_result.error_message}")
        
        # Chunk document
        chunk_result = await self.pdf_agent.execute_action(
            'chunk_document', 
            str(pdf_path),
            previous_results={'extract_text': extract_result.output}
        )
        if not chunk_result.success:
            raise Exception(f"Chunking failed: {chunk_result.error_message}")
        
        # Create embeddings
        embedding_result = await self.pdf_agent.execute_action(
            'create_embeddings',
            str(pdf_path), 
            previous_results={'chunk_document': chunk_result.output}
        )
        if not embedding_result.success:
            raise Exception(f"Embedding creation failed: {embedding_result.error_message}")
        
        # Store in content store
        chunks = chunk_result.output.get('chunks', [])
        title = pdf_path.stem
        
        content_id = self.content_store.add_pdf_content(
            pdf_path=str(pdf_path),
            title=title,
            chunks=chunks,
            metadata={
                'file_path': str(pdf_path),
                'auto_processed': True,
                'file_size': pdf_path.stat().st_size
            }
        )
        
        return content_id

async def main():
    print("SIMPLE AUTO PDF PROCESSOR")
    print("=" * 30)
    print("Monitoring watched_pdfs/ directory")
    print("Checking every 10 seconds for new files")
    print("Press Ctrl+C to stop")
    print()
    
    processor = SimpleAutoProcessor()
    
    try:
        while True:
            try:
                processed_count = await processor.scan_and_process()
                if processed_count > 0:
                    print(f"Processed {processed_count} new files")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No new files found")
                
                # Wait 10 seconds before next scan
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        print("\nStopped by user")

if __name__ == "__main__":
    asyncio.run(main())