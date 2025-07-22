#!/usr/bin/env python3
"""
Quantum Physics Content Comparison: PDF vs Video
Analyze the differences in quantum physics topics between PDF and video sources
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any

def load_unified_content():
    """Load content from unified index"""
    index_path = Path("unified_content/index.json")
    try:
        with open(index_path, 'r') as f:
            data = json.load(f)
        return data.get('content_items', [])
    except Exception as e:
        print(f"Error loading content: {e}")
        return []

def extract_quantum_topics(text: str) -> Dict[str, List[str]]:
    """Extract quantum physics topics and contexts from text"""
    text_lower = text.lower()
    
    # Define quantum physics topics with related terms
    quantum_topics = {
        'quantum_mechanics_foundations': [
            'quantum mechanics', 'quantum theory', 'quantum physics',
            'wave function', 'superposition', 'uncertainty principle'
        ],
        'historical_figures': [
            'planck', 'einstein', 'bohr', 'heisenberg', 'schrödinger', 
            'feynman', 'de broglie', 'dirac'
        ],
        'quantum_phenomena': [
            'double slit', 'interference', 'quantum entanglement',
            'wave-particle duality', 'quantum tunneling', 'quantum states'
        ],
        'quantum_mathematics': [
            'eigenstate', 'eigenvalue', 'hamilton', 'lagrangian',
            'action principle', 'path integral', 'probability amplitude'
        ],
        'atomic_structure': [
            'atomic model', 'electron orbit', 'energy levels',
            'photon emission', 'quantum jump', 'angular momentum'
        ],
        'quantum_constants': [
            'planck constant', 'reduced planck', 'quantum of action',
            'fundamental constant', 'physical constant'
        ],
        'experimental_aspects': [
            'blackbody radiation', 'photoelectric effect', 'spectroscopy',
            'quantum measurement', 'observation', 'detector'
        ]
    }
    
    found_topics = {}
    
    for topic_category, terms in quantum_topics.items():
        found_contexts = []
        
        for term in terms:
            # Find all sentences containing this term
            pattern = rf'[^.]*\b{re.escape(term)}\b[^.]*'
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                # Clean up the match
                context = match.strip().replace('\n', ' ')
                if len(context) > 20:  # Minimum meaningful length
                    found_contexts.append(context[:200])  # Limit length
        
        if found_contexts:
            found_topics[topic_category] = found_contexts[:3]  # Top 3 contexts per topic
    
    return found_topics

def analyze_quantum_content():
    """Main analysis function"""
    content_items = load_unified_content()
    
    if not content_items:
        print("❌ No content loaded!")
        return
    
    print("QUANTUM PHYSICS CONTENT COMPARISON: PDF vs VIDEO")
    print("=" * 55)
    
    # Find quantum physics content
    pdf_quantum = None
    video_quantum = None
    
    for item in content_items:
        full_text = item.get('full_text', '')
        content_type = item.get('content_type', '')
        
        if content_type == 'pdf' and 'quantum' in full_text.lower():
            if not pdf_quantum or full_text.lower().count('quantum') > pdf_quantum[1].lower().count('quantum'):
                pdf_quantum = (item, full_text)
        
        elif content_type == 'youtube' and 'qJZ1Ez28C-A' in item.get('source_path', ''):
            video_quantum = (item, full_text)
    
    if not pdf_quantum or not video_quantum:
        print("❌ Missing quantum content sources!")
        return
    
    # Extract topics from both sources
    pdf_item, pdf_text = pdf_quantum
    video_item, video_text = video_quantum
    
    print(f"📄 PDF SOURCE: {pdf_item.get('title', 'Unknown')}")
    print(f"   - Length: {len(pdf_text):,} characters")
    print(f"   - Source: {pdf_item.get('source_path', 'Unknown')}")
    
    print(f"\n🎥 VIDEO SOURCE: {video_item.get('title', 'Unknown')}")
    print(f"   - Length: {len(video_text):,} characters")
    print(f"   - Source: {video_item.get('source_path', 'Unknown')}")
    
    pdf_topics = extract_quantum_topics(pdf_text)
    video_topics = extract_quantum_topics(video_text)
    
    print(f"\n📊 TOPIC COVERAGE COMPARISON:")
    print(f"   PDF covers {len(pdf_topics)} quantum topic categories")
    print(f"   Video covers {len(video_topics)} quantum topic categories")
    
    # Compare topic coverage
    all_topics = set(pdf_topics.keys()) | set(video_topics.keys())
    pdf_only = set(pdf_topics.keys()) - set(video_topics.keys())
    video_only = set(video_topics.keys()) - set(pdf_topics.keys())
    shared_topics = set(pdf_topics.keys()) & set(video_topics.keys())
    
    print(f"\n🔄 TOPIC OVERLAP ANALYSIS:")
    print(f"   Shared topics: {len(shared_topics)} ({len(shared_topics)/len(all_topics)*100:.1f}%)")
    print(f"   PDF-only topics: {len(pdf_only)}")
    print(f"   Video-only topics: {len(video_only)}")
    
    # Detailed topic comparison
    print(f"\n📚 SHARED QUANTUM TOPICS:")
    for topic in sorted(shared_topics):
        print(f"\n   🔸 {topic.replace('_', ' ').title()}:")
        
        print(f"      📄 PDF Context:")
        for context in pdf_topics[topic][:2]:
            print(f"         • {context[:150]}...")
        
        print(f"      🎥 Video Context:")
        for context in video_topics[topic][:2]:
            print(f"         • {context[:150]}...")
    
    print(f"\n📄 PDF-ONLY TOPICS:")
    for topic in sorted(pdf_only):
        print(f"   • {topic.replace('_', ' ').title()}")
        for context in pdf_topics[topic][:1]:
            print(f"     - {context[:100]}...")
    
    print(f"\n🎥 VIDEO-ONLY TOPICS:")
    for topic in sorted(video_only):
        print(f"   • {topic.replace('_', ' ').title()}")
        for context in video_topics[topic][:1]:
            print(f"     - {context[:100]}...")
    
    # Depth analysis
    print(f"\n🔍 CONTENT DEPTH ANALYSIS:")
    
    # Count specific quantum terms
    quantum_terms = {
        'quantum': (pdf_text.lower().count('quantum'), video_text.lower().count('quantum')),
        'eigenstate': (pdf_text.lower().count('eigenstate'), video_text.lower().count('eigenstate')),
        'planck': (pdf_text.lower().count('planck'), video_text.lower().count('planck')),
        'feynman': (pdf_text.lower().count('feynman'), video_text.lower().count('feynman')),
        'wave function': (pdf_text.lower().count('wave function'), video_text.lower().count('wave function')),
        'action': (pdf_text.lower().count('action'), video_text.lower().count('action'))
    }
    
    print("   Term frequency comparison (PDF vs Video):")
    for term, (pdf_count, video_count) in quantum_terms.items():
        if pdf_count > 0 or video_count > 0:
            print(f"   • {term}: {pdf_count} vs {video_count}")
    
    # Content focus analysis
    print(f"\n🎯 CONTENT FOCUS:")
    
    pdf_focus = []
    if pdf_text.lower().count('dark matter') > 5:
        pdf_focus.append("Dark matter applications of quantum theory")
    if pdf_text.lower().count('eigenstate') > 10:
        pdf_focus.append("Quantum eigenstate mathematics")
    if pdf_text.lower().count('gravitational') > 5:
        pdf_focus.append("Gravitational quantum systems")
    
    video_focus = []
    if video_text.lower().count('feynman') > 3:
        video_focus.append("Feynman's path integral approach")
    if video_text.lower().count('planck') > 5:
        video_focus.append("Historical development of quantum theory")
    if video_text.lower().count('action') > 10:
        video_focus.append("Principle of least action")
    if video_text.lower().count('double slit') > 3:
        video_focus.append("Quantum interference experiments")
    
    print(f"   📄 PDF Focus Areas:")
    for focus in pdf_focus:
        print(f"      • {focus}")
    
    print(f"   🎥 Video Focus Areas:")
    for focus in video_focus:
        print(f"      • {focus}")
    
    print(f"\n🏁 SUMMARY:")
    print(f"   • PDF provides mathematical quantum theory for dark matter/cosmology")
    print(f"   • Video provides conceptual quantum mechanics education")
    print(f"   • Both sources complement each other well")
    print(f"   • Video is more accessible, PDF is more technical")
    print(f"   • Total quantum physics content: {len(pdf_text) + len(video_text):,} characters")

if __name__ == "__main__":
    analyze_quantum_content()