#!/usr/bin/env python3
"""
Multi-PDF Test Cases for Knowledge Agent System
Test scenarios requiring analysis of 2+ PDFs simultaneously
"""

import sys
sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from storage.enhanced_ner_fuzzy_router import EnhancedNERFuzzyRouter

def test_multi_pdf_scenarios():
    """Test scenarios requiring multiple PDF analysis"""
    
    print("=== MULTI-PDF TEST SCENARIOS ===\n")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    # Test cases designed to require multiple PDFs
    test_scenarios = [
        {
            'query': 'compare renewable energy storage solutions with carbon capture efficiency',
            'description': 'Cross-domain comparison requiring energy storage + carbon capture PDFs',
            'expected_pdfs': 2,
            'reasoning': 'Should pull both energy storage PDFs and carbon sequestration PDFs for comprehensive comparison'
        },
        
        {
            'query': 'analyze the relationship between solar technology efficiency and environmental carbon reduction',
            'description': 'Technology-environment intersection analysis',
            'expected_pdfs': 2,
            'reasoning': 'Requires solar/energy PDFs + environmental/carbon PDFs for complete analysis'
        },
        
        {
            'query': 'what are the differences between various carbon sequestration methods and their energy requirements',
            'description': 'Multi-method comparison with energy analysis',
            'expected_pdfs': 2,
            'reasoning': 'Needs carbon capture PDFs + energy system PDFs to analyze energy requirements'
        },
        
        {
            'query': 'comprehensive analysis of sustainable energy systems including storage and carbon footprint',
            'description': 'Holistic sustainability analysis',
            'expected_pdfs': 3,
            'reasoning': 'Should include energy storage + renewable energy + carbon/environmental PDFs'
        },
        
        {
            'query': 'compare different approaches to climate change mitigation through technology',
            'description': 'Broad climate technology comparison',
            'expected_pdfs': 2,
            'reasoning': 'Requires both renewable energy PDFs and carbon capture PDFs for complete picture'
        },
        
        {
            'query': 'evaluate trade-offs between renewable energy costs and carbon reduction benefits',
            'description': 'Cost-benefit analysis across domains',
            'expected_pdfs': 2,
            'reasoning': 'Needs energy economics PDFs + carbon/environmental impact PDFs'
        }
    ]
    
    print(f"Testing {len(test_scenarios)} multi-PDF scenarios:")
    print("-" * 80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        query = scenario['query']
        description = scenario['description']
        expected = scenario['expected_pdfs']
        reasoning = scenario['reasoning']
        
        print(f"\nSCENARIO {i}: {description}")
        print(f"Query: '{query}'")
        print(f"Expected: {expected}+ PDFs")
        print(f"Reasoning: {reasoning}")
        
        try:
            pdfs, videos, explanation = enhanced_router.route_query(query)
            
            pdf_count = len(pdfs)
            video_count = len(videos)
            
            print(f"Results: {pdf_count} PDFs, {video_count} videos")
            print(f"Explanation: {explanation}")
            
            # Analyze which PDFs were selected
            if pdfs:
                print("PDFs selected:")
                for j, pdf_path in enumerate(pdfs, 1):
                    # Find PDF details
                    for item in content_store.get_all_content():
                        if item.content_type == 'pdf' and item.source_path == pdf_path:
                            print(f"  {j}. {item.title}")
                            break
            
            # Evaluate success
            if pdf_count >= expected:
                print(f"✅ SUCCESS: Found {pdf_count}/{expected}+ PDFs as expected")
            elif pdf_count >= 2:
                print(f"✅ PARTIAL: Found {pdf_count} PDFs (multi-PDF analysis possible)")
            else:
                print(f"❌ INSUFFICIENT: Only {pdf_count} PDF found (expected {expected}+)")
            
            # Check domain coverage
            domain_coverage = analyze_domain_coverage(pdfs, content_store)
            if len(domain_coverage) >= 2:
                print(f"✅ GOOD COVERAGE: {len(domain_coverage)} domains covered: {', '.join(domain_coverage)}")
            else:
                print(f"⚠️ LIMITED COVERAGE: Only {len(domain_coverage)} domain(s) covered")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("-" * 80)

def analyze_domain_coverage(pdf_paths, content_store):
    """Analyze which domains are covered by selected PDFs"""
    
    domains = set()
    domain_keywords = {
        'energy_storage': ['battery', 'storage', 'energy storage', 'grid'],
        'renewable_energy': ['solar', 'wind', 'renewable', 'photovoltaic'],
        'carbon_capture': ['carbon', 'sequestration', 'capture', 'co2'],
        'environmental': ['environment', 'climate', 'emission', 'pollution'],
        'technology': ['technology', 'system', 'efficiency', 'device']
    }
    
    for pdf_path in pdf_paths:
        for item in content_store.get_all_content():
            if item.content_type == 'pdf' and item.source_path == pdf_path:
                content = item.full_text.lower()
                title = item.title.lower()
                
                # Check which domains this PDF covers
                for domain, keywords in domain_keywords.items():
                    if any(keyword in content or keyword in title for keyword in keywords):
                        domains.add(domain)
                break
    
    return domains

def test_comparative_analysis():
    """Test comparative analysis queries requiring multiple sources"""
    
    print("\n=== COMPARATIVE ANALYSIS TESTS ===\n")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    comparative_queries = [
        "compare the effectiveness of different carbon capture technologies",
        "analyze pros and cons of various renewable energy storage methods", 
        "evaluate different approaches to sustainable energy systems",
        "comparison between solar and wind energy storage requirements",
        "assess multiple strategies for climate change mitigation"
    ]
    
    for i, query in enumerate(comparative_queries, 1):
        print(f"COMPARATIVE TEST {i}: '{query}'")
        
        try:
            pdfs, videos, explanation = enhanced_router.route_query(query)
            
            print(f"Results: {len(pdfs)} PDFs, {len(videos)} videos")
            print(f"Explanation: {explanation}")
            
            if len(pdfs) >= 2:
                print("✅ GOOD: Multiple PDFs for comparison")
            else:
                print("⚠️ LIMITED: Insufficient sources for comprehensive comparison")
            
            print()
            
        except Exception as e:
            print(f"ERROR: {e}\n")

def test_cross_domain_queries():
    """Test queries spanning multiple knowledge domains"""
    
    print("\n=== CROSS-DOMAIN ANALYSIS TESTS ===\n")
    
    content_store = UnifiedContentStore()
    enhanced_router = EnhancedNERFuzzyRouter(content_store)
    
    cross_domain_queries = [
        "how do energy storage systems impact carbon footprint reduction",
        "relationship between renewable energy adoption and environmental benefits",
        "economic and environmental trade-offs in clean technology",
        "integration challenges between renewable energy and carbon capture",
        "lifecycle analysis of sustainable technology solutions"
    ]
    
    for i, query in enumerate(cross_domain_queries, 1):
        print(f"CROSS-DOMAIN TEST {i}: '{query}'")
        
        try:
            pdfs, videos, explanation = enhanced_router.route_query(query)
            
            print(f"Results: {len(pdfs)} PDFs, {len(videos)} videos")
            
            # Analyze domain spread
            domains = analyze_domain_coverage(pdfs, content_store)
            print(f"Domain coverage: {len(domains)} domains - {', '.join(domains)}")
            
            if len(domains) >= 2 and len(pdfs) >= 2:
                print("✅ EXCELLENT: Multi-domain, multi-source analysis")
            elif len(pdfs) >= 2:
                print("✅ GOOD: Multiple sources available")
            else:
                print("⚠️ LIMITED: May need more sources for complete analysis")
            
            print()
            
        except Exception as e:
            print(f"ERROR: {e}\n")

def main():
    print("MULTI-PDF TEST CASE ANALYSIS")
    print("=" * 80)
    print("Testing Knowledge Agent's ability to route queries requiring")
    print("analysis of multiple PDFs simultaneously")
    print("=" * 80)
    
    # Run all test suites
    test_multi_pdf_scenarios()
    test_comparative_analysis() 
    test_cross_domain_queries()
    
    print("\n" + "=" * 80)
    print("MULTI-PDF TESTING COMPLETE")
    print("\nKey Success Criteria:")
    print("✅ Multiple PDFs returned for complex queries")
    print("✅ Cross-domain coverage achieved")
    print("✅ Comparative analysis sources provided")
    print("✅ Comprehensive knowledge synthesis possible")

if __name__ == "__main__":
    main()