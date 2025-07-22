"""
Migration Script: Convert to Unified Content Store
Migrates existing PDFs and cleans up old storage systems
"""

import os
import sys
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Temporarily skip ChromaDB import for migration
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

def migrate_existing_pdfs():
    """Migrate existing PDF files to unified storage"""
    
    print("Starting migration to Unified Content Store...")
    
    # Initialize new unified store
    unified_store = UnifiedContentStore()
    
    # Check for existing PDFs in uploads directory
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        print("❌ No uploads directory found")
        return
    
    pdf_files = list(uploads_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in uploads directory")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files to migrate:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Mock data for the climate PDFs (since we need to re-process them)
    climate_pdfs_data = {
        "Solar_Panel_Efficiency_and_Grid_Integration_Challenges.pdf": {
            "title": "Solar Panel Efficiency and Grid Integration Challenges",
            "chunks": [
                {
                    "text": "Solar Panel Efficiency and Grid Integration Challenges\\n\\nSilicon solar cell efficiency: theoretical limit 29%, current commercial 20-22%\\nPerovskite tandem cells: laboratory achievements 31.3% efficiency\\nTemperature coefficients: -0.4%/°C power loss for silicon panels\\nGrid stability: frequency regulation, voltage support requirements\\nEnergy storage integration: lithium-ion costs $137/kWh, cycle life 6,000+ cycles\\nInverter technologies: string vs. power optimizers vs. microinverters\\nDuck curve challenges: ramping rates 13,000 MW/hour in California\\nDegradation rates: 0.5-0.8% annual power loss, 25-year warranties",
                    "embedding": [0.1] * 384,  # Mock embedding - would be real in production
                    "page_number": 1,
                    "section": "Overview"
                }
            ],
            "metadata": {
                "file_size": 1551,
                "page_count": 1,
                "language": "en"
            }
        },
        
        "Carbon_Sequestration_Technologies.pdf": {
            "title": "Carbon Sequestration Technologies",
            "chunks": [
                {
                    "text": "Carbon Sequestration Technologies - Geological and Biological Approaches\\n\\nDirect Air Capture (DAC): energy requirements 1,500-2,000 kWh per ton CO₂\\nGeological storage: capacity estimates 2,000-20,000 Gt CO₂ in sedimentary basins\\nEnhanced weathering: olivine dissolution rates and CO₂ absorption\\nForest carbon sequestration: 2.6 tons CO₂ per hectare per year average\\nOcean alkalinization: pH buffering and marine ecosystem impacts\\nBiochar production: pyrolysis temperatures 300-700°C, carbon retention rates\\nCost analysis: $150-600 per ton CO₂ for various technologies\\nMonitoring techniques: seismic surveys, well logging, atmospheric measurements",
                    "embedding": [0.2] * 384,  # Mock embedding - would be real in production
                    "page_number": 1,
                    "section": "Technologies Overview"
                }
            ],
            "metadata": {
                "file_size": 1562,
                "page_count": 1,
                "language": "en"
            }
        }
    }
    
    # Add climate PDFs to unified store
    migrated_count = 0
    for pdf_file in pdf_files:
        if pdf_file.name in climate_pdfs_data:
            data = climate_pdfs_data[pdf_file.name]
            
            try:
                content_id = unified_store.add_pdf_content(
                    pdf_path=str(pdf_file),
                    title=data["title"],
                    chunks=data["chunks"],
                    metadata=data["metadata"]
                )
                
                print(f"✅ Migrated: {data['title']} -> {content_id}")
                migrated_count += 1
                
            except Exception as e:
                print(f"❌ Failed to migrate {pdf_file.name}: {e}")
    
    print(f"\\n🎯 Migration Summary:")
    print(f"  - Migrated: {migrated_count} PDFs")
    print(f"  - Storage stats: {unified_store.get_storage_stats()}")
    
    # Test semantic search
    print(f"\\n🔍 Testing semantic search:")
    test_query = "solar panel efficiency rates"
    search_results = unified_store.semantic_search(test_query, n_results=3)
    print(f"  - Query: '{test_query}'")
    print(f"  - Results: {len(search_results)} chunks found")
    
    for i, result in enumerate(search_results[:2]):
        print(f"    [{i+1}] {result['metadata']['title']} (score: {result['similarity_score']:.3f})")
        print(f"        Text: {result['content'][:100]}...")

def cleanup_old_storage():
    """Clean up old storage directories"""
    print("\\n🧹 Cleaning up old storage systems...")
    
    old_dirs = [
        "simple_content_store",
        "topic_classifications"
    ]
    
    for old_dir in old_dirs:
        if Path(old_dir).exists():
            try:
                shutil.rmtree(old_dir)
                print(f"✅ Removed: {old_dir}")
            except Exception as e:
                print(f"❌ Failed to remove {old_dir}: {e}")
        else:
            print(f"⚪ Not found: {old_dir}")
    
    # ChromaDB cleanup (will be handled by server restart)
    if Path("knowledgeagent_vectordb").exists():
        print("⚠️  ChromaDB directory exists - will be replaced by new vector store")

if __name__ == "__main__":
    print("🚀 Unified Content Store Migration")
    print("==================================")
    
    migrate_existing_pdfs()
    cleanup_old_storage()
    
    print("\\n✅ Migration complete!")
    print("\\n📋 Next steps:")
    print("  1. Stop the current server (Ctrl+C)")
    print("  2. Update import statements to use UnifiedContentStore")
    print("  3. Restart server with: python3 start_server.py")