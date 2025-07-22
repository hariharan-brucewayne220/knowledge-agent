"""
Process the new complex PDFs with NER enhancement
"""

import sys, os, asyncio
sys.path.append('src')

from agents.pdf_agent import PDFAgent
from storage.unified_content_store import UnifiedContentStore

async def process_complex_pdfs():
    """Process the two new complex PDFs through NER pipeline"""
    
    print("Processing new complex PDFs with NER enhancement...")
    
    # Initialize components
    pdf_agent = PDFAgent()
    store = UnifiedContentStore()
    
    new_pdfs = [
        'uploads/Advanced_Battery_Technologies_and_Grid_Storage_Systems.pdf',
        'uploads/Atmospheric_Carbon_Removal_and_Climate_Engineering.pdf'
    ]
    
    for pdf_path in new_pdfs:
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue
            
        print(f"\nProcessing: {pdf_path}")
        
        try:
            # Extract text
            extract_result = await pdf_agent.execute_action("extract_text", pdf_path)
            if not extract_result.success:
                print(f"Text extraction failed: {extract_result.error_message}")
                continue
            
            # Create chunks 
            chunk_result = await pdf_agent.execute_action(
                "chunk_document", 
                pdf_path,
                previous_results={"extract_text": extract_result.output}
            )
            if not chunk_result.success:
                print(f"Chunking failed: {chunk_result.error_message}")
                continue
            
            # Create embeddings
            embed_result = await pdf_agent.execute_action(
                "create_embeddings",
                pdf_path,
                previous_results={"chunk_document": chunk_result.output}
            )
            if not embed_result.success:
                print(f"Embeddings failed: {embed_result.error_message}")
                continue
            
            # Extract title from content
            extracted_text = extract_result.output.get("extracted_text", "")
            lines = extracted_text.split('\n')
            title = None
            for line in lines[:10]:
                line = line.strip()
                if len(line) > 20 and len(line) < 100 and not line.startswith('['):
                    title = line
                    break
            
            if not title:
                title = os.path.basename(pdf_path).replace('.pdf', '').replace('_', ' ')
            
            # Store in unified content store with NER processing
            chunks_with_embeddings = embed_result.output.get("enriched_chunks", [])
            content_metadata = {
                "source_path": pdf_path,
                "total_pages": extract_result.output.get("total_pages", 0),
                "total_chunks": chunk_result.output.get("total_chunks", 0),
                "embedding_count": embed_result.output.get("embedding_count", 0)
            }
            
            content_id = store.add_pdf_content(
                pdf_path=pdf_path,
                title=title,
                chunks=chunks_with_embeddings,
                metadata=content_metadata
            )
            
            print(f"Successfully processed: {title}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Show results
    print("\n=== PROCESSING COMPLETE ===")
    all_content = store.get_all_content()
    print(f"Total content items: {len(all_content)}")
    
    # Show NER analysis for each item
    for item in all_content:
        print(f"\n{item.title}")
        print(f"  Keywords: {item.keywords[:10]}")
        print(f"  Topics: {item.topic_assignments}")
        print(f"  Confidence: {[f'{score:.3f}' for score in item.confidence_scores]}")
        
        if hasattr(item.metadata, 'get') and item.metadata.get('ner_topics'):
            print(f"  NER Topics:")
            for topic, score in item.metadata['ner_topics'].items():
                if score > 0.1:
                    print(f"    {topic}: {score:.3f}")

if __name__ == "__main__":
    asyncio.run(process_complex_pdfs())