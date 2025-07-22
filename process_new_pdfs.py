"""
Process and test the two new complex PDFs with NER system
"""

import asyncio
import sys
import os
sys.path.append('src')

from agents.enhanced_research_executor import EnhancedResearchExecutor
from storage.unified_content_store import UnifiedContentStore

async def process_new_pdfs():
    """Process the two new complex PDFs and test NER extraction"""
    
    print("=== PROCESSING NEW COMPLEX PDFs ===\n")
    
    # Initialize executor
    executor = EnhancedResearchExecutor()
    
    # New PDF files to process
    new_pdfs = [
        "uploads/Advanced_Battery_Technologies_and_Grid_Storage_Systems.pdf",
        "uploads/Atmospheric_Carbon_Removal_and_Climate_Engineering.pdf"
    ]
    
    # Check if files exist
    for pdf_path in new_pdfs:
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return False
        else:
            print(f"✅ File found: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    
    print("\n=== PROCESSING PDFs THROUGH NER PIPELINE ===\n")
    
    try:
        # Process PDFs through the enhanced research pipeline
        result = await executor.execute_research_query(
            query="Process these new technical documents and analyze their content",
            pdf_files=new_pdfs,
            use_openai=False,
            max_steps=5
        )
        
        print(f"Processing result: {result.final_answer[:200]}...")
        print(f"Sources processed: {result.sources_processed}")
        print(f"Execution time: {result.total_execution_time:.2f}s")
        
        # Check if content was stored
        store = UnifiedContentStore()
        all_content = store.get_all_content()
        
        print(f"\n=== CONTENT ANALYSIS ===")
        print(f"Total content items in store: {len(all_content)}")
        
        # Find the new content items
        new_items = [item for item in all_content 
                    if any(pdf_name.replace('uploads/', '').replace('.pdf', '').replace('_', ' ').lower() in item.title.lower() 
                          for pdf_name in new_pdfs)]
        
        if new_items:
            print(f"✅ Found {len(new_items)} new content items:")
            for item in new_items:
                print(f"\n📄 {item.title}")
                print(f"   Type: {item.content_type}")
                print(f"   Keywords: {item.keywords[:8]}")
                print(f"   Topics: {item.topic_assignments}")
                print(f"   Confidence: {[f'{score:.3f}' for score in item.confidence_scores]}")
                print(f"   Content length: {len(item.full_text)} chars")
                
                # Show NER topic analysis
                if hasattr(item.metadata, 'get') and item.metadata.get('ner_topics'):
                    print(f"   NER Analysis:")
                    for topic, score in item.metadata['ner_topics'].items():
                        if score > 0.1:
                            print(f"     {topic}: {score:.3f}")
        else:
            print("❌ No new content items found - PDFs may not have processed correctly")
            
            # Show all items for debugging
            print("\nAll content items in store:")
            for item in all_content:
                print(f"  - {item.title} ({item.content_type})")
        
        return len(new_items) > 0
        
    except Exception as e:
        print(f"❌ Error processing PDFs: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_complex_queries():
    """Test complex multi-resource queries with the new PDFs"""
    
    print("\n=== TESTING COMPLEX MULTI-RESOURCE QUERIES ===\n")
    
    store = UnifiedContentStore()
    
    from storage.simple_smart_router import SimpleSmartRouter
    router = SimpleSmartRouter(store)
    
    # Complex test queries that should use multiple new PDFs
    test_queries = [
        {
            "query": "How much battery storage capacity would be needed to power large-scale atmospheric carbon removal operations?",
            "expected": "Should find both battery and atmospheric removal PDFs"
        },
        {
            "query": "What are the temperature requirements for solid-state batteries versus biochar pyrolysis processes?", 
            "expected": "Should detect temperature-related processes in both PDFs"
        },
        {
            "query": "Could vanadium redox flow batteries provide the energy needed for direct air capture artificial trees?",
            "expected": "Should link battery technology with carbon removal energy needs"
        },
        {
            "query": "Compare energy density of lithium-ion batteries with CO2 fixation rates in algae systems",
            "expected": "Should find technical specifications from both domains"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"{i}. Complex Query Test")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected']}")
        
        try:
            pdfs, videos, explanation = router.route_query(test_case['query'])
            
            print(f"PDFs found: {len(pdfs)}")
            print(f"PDF files: {[os.path.basename(p) for p in pdfs]}")
            print(f"Routing explanation: {explanation}")
            
            # Check if it found the new PDFs specifically
            new_pdf_names = ["Advanced_Battery_Technologies", "Atmospheric_Carbon_Removal"]
            found_new_pdfs = [name for name in new_pdf_names 
                             if any(name in pdf for pdf in pdfs)]
            
            print(f"New PDFs detected: {found_new_pdfs}")
            print(f"Status: {'✅ SUCCESS' if len(found_new_pdfs) >= 2 else '⚠️ PARTIAL' if found_new_pdfs else '❌ FAILED'}")
            
        except Exception as e:
            print(f"❌ Error in query: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    async def main():
        # First process the new PDFs
        success = await process_new_pdfs()
        
        if success:
            # Then test complex queries
            await test_complex_queries()
        else:
            print("❌ PDF processing failed, skipping query tests")
    
    asyncio.run(main())