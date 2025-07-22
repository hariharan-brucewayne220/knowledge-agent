"""
Debug what chunks are actually being fed to the synthesizer
"""

import sys
import os
sys.path.append('src')
import asyncio

async def debug_chunk_retrieval():
    """Debug what chunks are retrieved and fed to synthesizer"""
    
    print("DEBUGGING CHUNK RETRIEVAL AND FEEDING")
    print("=" * 50)
    
    from agents.enhanced_research_executor import EnhancedResearchExecutor
    from storage.unified_content_store import UnifiedContentStore
    from storage.simple_smart_router import SimpleSmartRouter
    
    # Initialize components
    executor = EnhancedResearchExecutor()
    store = UnifiedContentStore()
    router = SimpleSmartRouter(store)
    
    # Test query that should return specific battery data
    test_query = "What is the energy density of lithium-ion batteries?"
    
    print(f"Test Query: {test_query}")
    print()
    
    # Step 1: Check what PDFs are routed
    print("STEP 1: PDF Routing")
    print("-" * 20)
    
    pdfs, videos, explanation = router.route_query(test_query)
    print(f"PDFs routed: {len(pdfs)}")
    for pdf in pdfs:
        print(f"  - {os.path.basename(pdf)}")
    print(f"Routing explanation: {explanation}")
    print()
    
    # Step 2: Check actual content in battery PDF
    print("STEP 2: Battery PDF Content Check")
    print("-" * 30)
    
    battery_pdf = None
    for item in store.get_all_content():
        if "Advanced Battery Technologies" in item.title:
            battery_pdf = item
            break
    
    if battery_pdf:
        print(f"Battery PDF Found: {battery_pdf.title}")
        print(f"Full text length: {len(battery_pdf.full_text)} chars")
        
        # Clean text for display
        clean_text = battery_pdf.full_text.replace('₂', '2').replace('°', ' degrees')
        print(f"Content preview:")
        print(clean_text[:400])
        
        # Check for specific data
        if "250-300" in battery_pdf.full_text:
            print("\n✅ CONFIRMED: '250-300 Wh/kg' IS in the PDF content")
        else:
            print("\n❌ ERROR: '250-300 Wh/kg' NOT found in PDF content")
        
        # Show chunks that would be retrieved
        print(f"\nChunks in Battery PDF: {len(battery_pdf.chunks)}")
        for i, chunk in enumerate(battery_pdf.chunks):
            if hasattr(chunk, 'text'):
                chunk_text = chunk.text.replace('₂', '2').replace('°', ' degrees')
                print(f"  Chunk {i}: {chunk_text[:200]}...")
                
                # Check if this chunk contains the key data
                if "250-300" in chunk.text:
                    print(f"    ✅ This chunk contains '250-300 Wh/kg'")
            elif isinstance(chunk, dict) and 'text' in chunk:
                chunk_text = chunk['text'].replace('₂', '2').replace('°', ' degrees')
                print(f"  Chunk {i}: {chunk_text[:200]}...")
                
                if "250-300" in chunk['text']:
                    print(f"    ✅ This chunk contains '250-300 Wh/kg'")
    else:
        print("❌ ERROR: Battery PDF not found!")
    
    print("\n" + "=" * 50)
    
    # Step 3: Run actual query and see what gets fed to synthesizer
    print("STEP 3: Execute Query and Check Synthesizer Input")
    print("-" * 45)
    
    try:
        # Execute the query
        result = await executor.execute_research_query(
            query=test_query,
            pdf_files=pdfs,  # Use the routed PDFs
            use_openai=False,  # Use fallback to see what content is passed
            max_steps=5
        )
        
        print(f"Query executed successfully")
        print(f"Final answer length: {len(result.final_answer)}")
        print(f"Sources processed: {result.sources_processed}")
        
        # The answer should contain the actual PDF content
        print(f"\nFinal Answer:")
        print(result.final_answer[:500])
        
        # Check if the answer contains the correct data
        if "250-300" in result.final_answer:
            print("\n✅ SUCCESS: Answer contains correct PDF data (250-300 Wh/kg)")
        else:
            print("\n❌ PROBLEM: Answer does not contain correct PDF data")
            
        # Check for hallucinated data
        if "100-265" in result.final_answer or "generally falls within" in result.final_answer:
            print("❌ HALLUCINATION: Answer contains generic/incorrect data")
        
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_chunk_retrieval())