"""
Simple Migration: Create unified content structure
"""

import os
import json
import shutil
from pathlib import Path

def create_unified_structure():
    """Create unified content directory structure"""
    print("Creating unified content structure...")
    
    # Create unified content directory
    unified_dir = Path("unified_content")
    unified_dir.mkdir(exist_ok=True)
    
    # Create sample content with your climate PDFs
    sample_content = {
        "content_items": [
            {
                "id": "pdf_solar_panel_efficiency",
                "title": "Solar Panel Efficiency and Grid Integration Challenges",
                "content_type": "pdf",
                "source_path": "uploads/Solar_Panel_Efficiency_and_Grid_Integration_Challenges.pdf",
                "chunks": [
                    {
                        "chunk_id": 0,
                        "text": "Solar Panel Efficiency and Grid Integration Challenges\n\nSilicon solar cell efficiency: theoretical limit 29%, current commercial 20-22%\nPerovskite tandem cells: laboratory achievements 31.3% efficiency\nTemperature coefficients: -0.4%/°C power loss for silicon panels\nGrid stability: frequency regulation, voltage support requirements\nEnergy storage integration: lithium-ion costs $137/kWh, cycle life 6,000+ cycles\nInverter technologies: string vs. power optimizers vs. microinverters\nDuck curve challenges: ramping rates 13,000 MW/hour in California\nDegradation rates: 0.5-0.8% annual power loss, 25-year warranties",
                        "embedding": [0.1] * 384,
                        "char_count": 567,
                        "metadata": {
                            "page_number": 1,
                            "section": "Overview"
                        }
                    }
                ],
                "full_text": "Solar Panel Efficiency and Grid Integration Challenges\n\nSilicon solar cell efficiency: theoretical limit 29%, current commercial 20-22%\nPerovskite tandem cells: laboratory achievements 31.3% efficiency\nTemperature coefficients: -0.4%/°C power loss for silicon panels\nGrid stability: frequency regulation, voltage support requirements\nEnergy storage integration: lithium-ion costs $137/kWh, cycle life 6,000+ cycles\nInverter technologies: string vs. power optimizers vs. microinverters\nDuck curve challenges: ramping rates 13,000 MW/hour in California\nDegradation rates: 0.5-0.8% annual power loss, 25-year warranties",
                "metadata": {
                    "file_size": 1551,
                    "page_count": 1,
                    "language": "en"
                },
                "created_at": 1753118000,
                "processing_status": "completed",
                "keywords": ["solar", "panel", "efficiency", "silicon", "perovskite", "grid", "integration"],
                "topic_assignments": [],
                "confidence_scores": []
            },
            {
                "id": "pdf_carbon_sequestration",
                "title": "Carbon Sequestration Technologies",
                "content_type": "pdf",
                "source_path": "uploads/Carbon_Sequestration_Technologies.pdf",
                "chunks": [
                    {
                        "chunk_id": 0,
                        "text": "Carbon Sequestration Technologies - Geological and Biological Approaches\n\nDirect Air Capture (DAC): energy requirements 1,500-2,000 kWh per ton CO2\nGeological storage: capacity estimates 2,000-20,000 Gt CO2 in sedimentary basins\nEnhanced weathering: olivine dissolution rates and CO2 absorption\nForest carbon sequestration: 2.6 tons CO2 per hectare per year average\nOcean alkalinization: pH buffering and marine ecosystem impacts\nBiochar production: pyrolysis temperatures 300-700C, carbon retention rates\nCost analysis: $150-600 per ton CO2 for various technologies\nMonitoring techniques: seismic surveys, well logging, atmospheric measurements",
                        "embedding": [0.2] * 384,
                        "char_count": 627,
                        "metadata": {
                            "page_number": 1,
                            "section": "Technologies Overview"
                        }
                    }
                ],
                "full_text": "Carbon Sequestration Technologies - Geological and Biological Approaches\n\nDirect Air Capture (DAC): energy requirements 1,500-2,000 kWh per ton CO2\nGeological storage: capacity estimates 2,000-20,000 Gt CO2 in sedimentary basins\nEnhanced weathering: olivine dissolution rates and CO2 absorption\nForest carbon sequestration: 2.6 tons CO2 per hectare per year average\nOcean alkalinization: pH buffering and marine ecosystem impacts\nBiochar production: pyrolysis temperatures 300-700C, carbon retention rates\nCost analysis: $150-600 per ton CO2 for various technologies\nMonitoring techniques: seismic surveys, well logging, atmospheric measurements",
                "metadata": {
                    "file_size": 1562,
                    "page_count": 1,
                    "language": "en"
                },
                "created_at": 1753118001,
                "processing_status": "completed", 
                "keywords": ["carbon", "sequestration", "storage", "capture", "geological", "biological"],
                "topic_assignments": [],
                "confidence_scores": []
            }
        ],
        "metadata": {
            "total_items": 2,
            "last_updated": 1753118000,
            "schema_version": "2.0"
        }
    }
    
    # Save unified content index
    index_file = unified_dir / "index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(sample_content, f, indent=2)
    
    print(f"Created unified content index: {index_file}")
    print(f"Content items: {len(sample_content['content_items'])}")
    
    # Clean up old directories if they exist
    cleanup_dirs = ["simple_content_store", "topic_classifications"]
    for cleanup_dir in cleanup_dirs:
        if Path(cleanup_dir).exists():
            try:
                shutil.rmtree(cleanup_dir)
                print(f"Removed old directory: {cleanup_dir}")
            except:
                print(f"Could not remove: {cleanup_dir}")
    
    print("Migration complete!")

if __name__ == "__main__":
    create_unified_structure()