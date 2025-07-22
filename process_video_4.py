#!/usr/bin/env python3
"""
Process video #4 and analyze content
"""

import asyncio
import sys

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from agents.youtube_agent import YouTubeAgent
from storage.ner_fuzzy_router import NERFuzzyRouter

async def process_video_4():
    """Process video #4 with URL: https://www.youtube.com/watch?v=qJZ1Ez28C-A"""
    
    youtube_url = "https://www.youtube.com/watch?v=qJZ1Ez28C-A"
    
    print(f"=== Processing Video #4 ===")
    print(f"URL: {youtube_url}")
    
    # Check current state
    print(f"\n1. Checking current content...")
    content_store = UnifiedContentStore()
    initial_content = content_store.get_all_content()
    initial_videos = [item for item in initial_content if item.content_type == 'youtube']
    
    print(f"Current content: {len(initial_content)} items")
    print(f"Current videos: {len(initial_videos)}")
    for i, video in enumerate(initial_videos, 1):
        print(f"  {i}. {video.title}")
    
    # Step 2: Get video transcript
    print(f"\n2. Getting video transcript...")
    youtube_agent = YouTubeAgent()
    
    try:
        captions_result = await youtube_agent.execute_action('get_captions', {'url': youtube_url})
        
        if not captions_result.success:
            print(f"ERROR: Failed to get captions: {captions_result.error}")
            return None
        
        output = captions_result.output
        video_info = output.get('video_info', {})
        transcript = output.get('transcript', '')
        segments = output.get('segments', [])
        
        print(f"SUCCESS: Got video transcript")
        print(f"Title: {video_info.get('title', 'Unknown')}")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Segments: {len(segments)}")
        
        # Show content preview
        print(f"\nContent preview:")
        print(f"'{transcript[:400]}...'")
        
    except Exception as e:
        print(f"ERROR: Exception getting captions: {e}")
        return None
    
    # Step 3: Analyze content themes
    print(f"\n3. Analyzing content themes...")
    
    # Look for key topics in transcript
    words = transcript.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 4 and word.isalpha():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"Top keywords: {[word for word, count in top_words[:10]]}")
    
    # Check for specific scientific/technical topics
    tech_topics = {
        'energy': transcript.lower().count('energy'),
        'solar': transcript.lower().count('solar'),
        'battery': transcript.lower().count('battery'),
        'carbon': transcript.lower().count('carbon'),
        'climate': transcript.lower().count('climate'),
        'technology': transcript.lower().count('technology'),
        'renewable': transcript.lower().count('renewable'),
        'electricity': transcript.lower().count('electricity'),
        'power': transcript.lower().count('power'),
        'physics': transcript.lower().count('physics'),
        'science': transcript.lower().count('science'),
        'research': transcript.lower().count('research'),
        'engineering': transcript.lower().count('engineering'),
        'design': transcript.lower().count('design'),
        'system': transcript.lower().count('system'),
        'computer': transcript.lower().count('computer'),
        'software': transcript.lower().count('software'),
        'algorithm': transcript.lower().count('algorithm'),
        'data': transcript.lower().count('data'),
        'artificial': transcript.lower().count('artificial'),
        'intelligence': transcript.lower().count('intelligence'),
        'machine': transcript.lower().count('machine'),
        'learning': transcript.lower().count('learning')
    }
    
    relevant_topics = {topic: count for topic, count in tech_topics.items() if count > 0}
    print(f"Tech topics found: {relevant_topics}")
    
    # Look for specific domain indicators
    domains = {
        'AI/ML': ['artificial', 'intelligence', 'machine', 'learning', 'algorithm', 'neural', 'model'],
        'Software': ['software', 'programming', 'code', 'computer', 'application'],
        'Engineering': ['engineering', 'design', 'system', 'technical', 'development'],
        'Energy': ['energy', 'power', 'solar', 'battery', 'renewable'],
        'Science': ['science', 'research', 'physics', 'chemistry', 'biology']
    }
    
    domain_scores = {}
    for domain, keywords in domains.items():
        score = sum(transcript.lower().count(keyword) for keyword in keywords)
        if score > 0:
            domain_scores[domain] = score
    
    print(f"Domain analysis: {domain_scores}")
    
    # Step 4: Store in content store
    print(f"\n4. Storing in content store...")
    
    try:
        title = video_info.get('title', f'Video qJZ1Ez28C-A')
        if not title or 'Unknown' in title:
            title = "Video qJZ1Ez28C-A"
        
        success = content_store.add_youtube_content(
            url=youtube_url,
            title=title,
            transcript_segments=segments,
            metadata={
                'video_id': 'qJZ1Ez28C-A',
                'duration': video_info.get('duration', 'Unknown'),
                'source': 'youtube_captions',
                'transcript_length': len(transcript),
                'batch_number': 4,
                'tech_topics': relevant_topics,
                'domain_scores': domain_scores
            }
        )
        
        if success:
            print(f"SUCCESS: Stored video in content store")
        else:
            print(f"ERROR: Failed to store video")
            return None
            
    except Exception as e:
        print(f"ERROR: Exception storing video: {e}")
        return None
    
    # Step 5: Verify storage and analyze stored content
    print(f"\n5. Verifying storage...")
    
    content_store = UnifiedContentStore()  # Reload
    final_content = content_store.get_all_content()
    final_videos = [item for item in final_content if item.content_type == 'youtube']
    
    print(f"Final content: {len(final_content)} items")
    print(f"Final videos: {len(final_videos)}")
    
    new_video = None
    for video in final_videos:
        if 'qJZ1Ez28C-A' in video.source_path:
            new_video = video
            break
    
    if new_video:
        print(f"SUCCESS: Found new video in content store")
        print(f"Title: {new_video.title}")
        print(f"Content length: {len(new_video.full_text)} chars")
        print(f"Keywords: {len(new_video.keywords)} extracted")
        print(f"Topics: {len(new_video.topic_assignments)} assigned")
        print(f"Sample keywords: {new_video.keywords[:10]}")
        
        # Show actual content from stored video
        stored_content = new_video.full_text.lower()
        print(f"\nContent verification in stored video:")
        for topic, expected_count in relevant_topics.items():
            actual_count = stored_content.count(topic)
            match_status = "MATCH" if actual_count == expected_count else f"DIFF ({actual_count} vs {expected_count})"
            print(f"  '{topic}': {match_status}")
    else:
        print(f"WARNING: New video not found in content store")
        return None
    
    # Step 6: Generate targeted queries based on content analysis
    print(f"\n6. Generating targeted queries...")
    
    queries = []
    
    # Add domain-based queries
    for domain, score in domain_scores.items():
        if score > 2:
            queries.extend([
                f"{domain.lower()} technology",
                f"what is {domain.lower()}",
                f"{domain.lower()} explained"
            ])
    
    # Add topic-based queries
    for topic, count in relevant_topics.items():
        if count > 2:
            queries.extend([
                f"{topic} systems",
                f"{topic} information"
            ])
    
    # Add general queries
    queries.extend([
        "video content explanation",
        "what does this video cover",
        "according to the video",
        "technical tutorial",
        "educational content"
    ])
    
    # Add keyword-based queries from most frequent terms
    for word, count in top_words[:5]:
        if count > 3:
            queries.append(f"{word} tutorial")
    
    # Remove duplicates and limit
    queries = list(set(queries))[:15]
    
    print(f"Generated {len(queries)} targeted queries:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")
    
    return {
        'video_id': 'qJZ1Ez28C-A',
        'title': title,
        'url': youtube_url,
        'transcript_length': len(transcript),
        'segments': len(segments),
        'keywords': [word for word, count in top_words[:12]],
        'tech_topics': relevant_topics,
        'domain_scores': domain_scores,
        'test_queries': queries,
        'content_preview': transcript[:600]
    }

async def test_video_4_routing(video_info):
    """Test routing for video #4"""
    
    print(f"\n=== Testing Video #4 Routing ===")
    
    content_store = UnifiedContentStore()
    ner_router = NERFuzzyRouter(content_store)
    
    # Test queries based on detected content
    test_queries = [
        "what does the video explain",
        "video tutorial content",
        "educational video",
        "technical information"
    ]
    
    # Add domain-specific queries
    if video_info['domain_scores']:
        top_domain = max(video_info['domain_scores'].items(), key=lambda x: x[1])
        test_queries.extend([
            f"{top_domain[0].lower()} tutorial",
            f"{top_domain[0].lower()} explained",
            f"learn {top_domain[0].lower()}"
        ])
    
    # Add topic-specific queries
    if video_info['tech_topics']:
        for topic in list(video_info['tech_topics'].keys())[:3]:
            test_queries.append(f"{topic} content")
    
    print(f"Testing {len(test_queries)} queries for video #4:")
    
    for i, query in enumerate(test_queries, 1):
        try:
            pdf_files, youtube_urls, explanation = ner_router.route_query(query)
            
            video_found = len(youtube_urls) > 0
            video_4_found = any('qJZ1Ez28C-A' in url for url in youtube_urls)
            
            status = "INCLUDES VIDEO" if video_found else "PDF ONLY"
            if video_4_found:
                status += " (Video #4)"
            
            print(f"  {i}. {status}: '{query}'")
            print(f"     -> {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
            print(f"     -> {explanation}")
            print()
            
        except Exception as e:
            print(f"  {i}. ERROR: '{query}' - {e}")
            print()

async def show_complete_system_status():
    """Show complete system status with all videos"""
    
    print(f"=== Complete System Status ===")
    
    content_store = UnifiedContentStore()
    all_content = content_store.get_all_content()
    
    pdfs = [item for item in all_content if item.content_type == 'pdf']
    videos = [item for item in all_content if item.content_type == 'youtube']
    
    print(f"Total Content: {len(all_content)} items")
    print(f"PDFs: {len(pdfs)}")
    print(f"Videos: {len(videos)}")
    
    print(f"\nPDF Collection:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"  {i}. {pdf.title}")
    
    print(f"\nVideo Collection:")
    for i, video in enumerate(videos, 1):
        marker = " <- NEW" if 'qJZ1Ez28C-A' in video.source_path else ""
        print(f"  {i}. {video.title} ({len(video.full_text)} chars){marker}")

async def main():
    print("Processing Video #4...")
    video_info = await process_video_4()
    
    if video_info:
        print(f"\n" + "="*60)
        print(f"VIDEO #4 PROCESSING COMPLETE")
        print(f"Title: {video_info['title']}")
        print(f"Content: {video_info['transcript_length']} chars, {video_info['segments']} segments")
        
        if video_info['domain_scores']:
            top_domain = max(video_info['domain_scores'].items(), key=lambda x: x[1])
            print(f"Primary domain: {top_domain[0]} (score: {top_domain[1]})")
        
        if video_info['tech_topics']:
            print(f"Tech topics: {list(video_info['tech_topics'].keys())[:5]}")
        
        # Test routing
        await test_video_4_routing(video_info)
        
        # Show complete system status
        await show_complete_system_status()
        
        print(f"\nReady for video #5 (final video)!")
        
    else:
        print(f"\nVideo #4 processing failed")

if __name__ == "__main__":
    asyncio.run(main())