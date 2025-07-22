"""
Regenerate existing content with NER enhancement
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def regenerate_content():
    """Regenerate existing content with advanced NER processing"""
    
    print("Regenerating content with NER enhancement...")
    
    try:
        from storage.unified_content_store import UnifiedContentStore
        
        # Initialize store
        store = UnifiedContentStore()
        
        print(f"Current content: {len(store.get_all_content())} items")
        
        # Sample data with enhanced structure for testing
        climate_pdfs = [
            {
                "title": "Solar Panel Efficiency and Grid Integration Challenges",
                "content": "Solar Panel Efficiency and Grid Integration Challenges\n\nSilicon solar cell efficiency: theoretical limit 29%, current commercial 20-22%\nPerovskite tandem cells: laboratory achievements 31.3% efficiency\nTemperature coefficients: -0.4%/°C power loss for silicon panels\nGrid stability: frequency regulation, voltage support requirements\nEnergy storage integration: lithium-ion costs $137/kWh, cycle life 6,000+ cycles\nInverter technologies: string vs. power optimizers vs. microinverters\nDuck curve challenges: ramping rates 13,000 MW/hour in California\nDegradation rates: 0.5-0.8% annual power loss, 25-year warranties",
                "path": "uploads/Solar_Panel_Efficiency_and_Grid_Integration_Challenges.pdf",
                "metadata": {"file_size": 1551, "page_count": 1, "language": "en"}
            },
            {
                "title": "Carbon Sequestration Technologies",
                "content": "Carbon Sequestration Technologies - Geological and Biological Approaches\n\nDirect Air Capture (DAC): energy requirements 1,500-2,000 kWh per ton CO₂\nGeological storage: capacity estimates 2,000-20,000 Gt CO₂ in sedimentary basins\nEnhanced weathering: olivine dissolution rates and CO₂ absorption\nForest carbon sequestration: 2.6 tons CO₂ per hectare per year average\nOcean alkalinization: pH buffering and marine ecosystem impacts\nBiochar production: pyrolysis temperatures 300-700°C, carbon retention rates\nCost analysis: $150-600 per ton CO₂ for various technologies\nMonitoring techniques: seismic surveys, well logging, atmospheric measurements",
                "path": "uploads/Carbon_Sequestration_Technologies.pdf", 
                "metadata": {"file_size": 1562, "page_count": 1, "language": "en"}
            }
        ]
        
        # Clear existing content
        store.content_items.clear()
        
        # Add content with NER processing
        for pdf_data in climate_pdfs:
            # Create chunks for the content
            chunks = [{
                'text': pdf_data['content'],
                'embedding': [0.1] * 384,  # Mock embedding
            }]
            
            content_id = store.add_pdf_content(
                pdf_path=pdf_data['path'],
                title=pdf_data['title'],
                chunks=chunks,
                metadata=pdf_data['metadata']
            )
            
            # Show what NER extracted
            item = store.content_items[content_id]
            print(f"\nNER Analysis for: {item.title}")
            print(f"  Keywords: {item.keywords[:10]}")  # First 10 keywords
            print(f"  Topic assignments: {item.topic_assignments}")
            print(f"  Confidence scores: {[f'{score:.2f}' for score in item.confidence_scores]}")
            
            if hasattr(item.metadata, 'get') and item.metadata.get('ner_topics'):
                ner_topics = item.metadata['ner_topics']
                print(f"  Topic scores:")
                for topic, score in ner_topics.items():
                    if score > 0.1:  # Only show significant topics
                        print(f"    {topic}: {score:.3f}")
        
        print(f"\nRegeneration complete!")
        print(f"Total content items: {len(store.get_all_content())}")
        
        # Test the enhanced smart routing
        print(f"\nTesting enhanced smart routing...")
        
        from storage.simple_smart_router import SimpleSmartRouter
        router = SimpleSmartRouter(store)
        
        test_query = "Could solar panels provide enough energy for carbon sequestration projects?"
        pdfs, videos, explanation = router.route_query(test_query)
        
        print(f"Query: {test_query}")
        print(f"PDFs found: {len(pdfs)} - {[Path(p).name for p in pdfs]}")
        print(f"Explanation: {explanation}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = regenerate_content()
    exit(0 if success else 1)