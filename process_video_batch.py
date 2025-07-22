#!/usr/bin/env python3
"""
Process video batch - analyze transcript and create targeted queries
"""

import asyncio
import sys

sys.path.append('src')

from storage.unified_content_store import UnifiedContentStore
from agents.youtube_agent import YouTubeAgent
from storage.ner_fuzzy_router import NERFuzzyRouter

async def process_single_video(youtube_url, video_number):
    """Process a single video and analyze its content"""
    
    print(f"=== Processing Video #{video_number} ===")
    print(f"URL: {youtube_url}")
    
    # Step 1: Get video transcript
    print(f"\n1. Getting video transcript...")
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
        print(f"'{transcript[:300]}...'")
        
    except Exception as e:
        print(f"ERROR: Exception getting captions: {e}")
        return None
    
    # Step 2: Store in content store
    print(f"\n2. Storing in content store...")
    
    try:
        content_store = UnifiedContentStore()
        
        title = video_info.get('title', f'Video {video_number}')
        if not title or 'Unknown' in title:
            video_id = youtube_url.split('v=')[1][:11] if 'v=' in youtube_url else f'video_{video_number}'
            title = f"Video {video_id}"
        
        success = content_store.add_youtube_content(
            url=youtube_url,
            title=title,
            transcript_segments=segments,
            metadata={
                'video_id': youtube_url.split('v=')[1][:11] if 'v=' in youtube_url else f'video_{video_number}',
                'duration': video_info.get('duration', 'Unknown'),
                'source': 'youtube_captions',
                'transcript_length': len(transcript),
                'batch_number': video_number
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
    
    # Step 3: Analyze content and extract key topics
    print(f"\n3. Analyzing content for key topics...")
    
    # Split transcript into sentences and analyze
    sentences = transcript.replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    print(f"Key content themes found:")
    
    # Look for main topics in first few sentences
    intro_content = ' '.join(sentences[:5])
    print(f"  Introduction: {intro_content[:200]}...")
    
    # Look for recurring keywords
    words = transcript.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 4 and word.isalpha():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"  Top keywords: {[word for word, count in top_words[:8]]}")
    
    # Step 4: Generate targeted test queries
    print(f"\n4. Generating targeted test queries...")
    
    # Base queries
    queries = [
        f"video content about {title.lower().split()[0] if title else 'topic'}",
        "what does this video discuss",
        "according to the video"
    ]
    
    # Add keyword-based queries
    if top_words:
        for word, count in top_words[:5]:
            if count > 2:  # Only words that appear multiple times
                queries.append(f"{word} explained")
                queries.append(f"what is {word}")
    
    # Add content-based queries from intro
    intro_words = intro_content.lower().split()
    important_terms = [w for w in intro_words if len(w) > 5 and w.isalpha()]
    if important_terms:
        queries.append(f"{important_terms[0]} information")
        if len(important_terms) > 1:
            queries.append(f"{important_terms[0]} and {important_terms[1]}")
    
    print(f"Generated {len(queries)} test queries:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")
    
    return {
        'video_number': video_number,
        'url': youtube_url,
        'title': title,
        'transcript_length': len(transcript),
        'segments': len(segments),
        'keywords': [word for word, count in top_words[:10]],
        'test_queries': queries,
        'content_preview': transcript[:500]
    }

async def test_video_routing(video_info):
    """Test routing for processed video"""
    
    print(f"\n=== Testing Routing for Video #{video_info['video_number']} ===")
    
    try:
        content_store = UnifiedContentStore()
        ner_router = NERFuzzyRouter(content_store)
        
        print(f"Testing {len(video_info['test_queries'])} queries...")
        
        for i, query in enumerate(video_info['test_queries'], 1):
            try:
                pdf_files, youtube_urls, explanation = ner_router.route_query(query)
                
                # Check if our video is included
                video_found = any(video_info['url'] in url or video_info['video_number'] == video_info['video_number'] 
                                for url in youtube_urls)
                
                status = "✅ INCLUDES VIDEO" if len(youtube_urls) > 0 else "📄 PDF ONLY"
                
                print(f"  {i}. {status}: '{query}'")
                print(f"     → {len(pdf_files)} PDFs, {len(youtube_urls)} videos")
                
                if len(youtube_urls) > 0:
                    print(f"     → {explanation}")
                
            except Exception as e:
                print(f"  {i}. ❌ ERROR: '{query}' - {e}")
        
    except Exception as e:
        print(f"ERROR: Routing test failed: {e}")

async def main():
    # Process video
    video_url = "https://www.youtube.com/watch?v=rJLtT0QXoPo"
    video_info = await process_single_video(video_url, 2)  # Video #2 (after the space video)
    
    if video_info:
        print(f"\n" + "="*60)
        print(f"VIDEO #{video_info['video_number']} PROCESSING COMPLETE")
        print(f"Title: {video_info['title']}")
        print(f"Content: {video_info['transcript_length']} chars, {video_info['segments']} segments")
        print(f"Top keywords: {video_info['keywords'][:5]}")
        print(f"Generated {len(video_info['test_queries'])} test queries")
        
        # Test routing
        await test_video_routing(video_info)
        
        # Show current content store status
        print(f"\n=== Content Store Status ===")
        content_store = UnifiedContentStore()
        all_content = content_store.get_all_content()
        videos = [item for item in all_content if item.content_type == 'youtube']
        
        print(f"Total content: {len(all_content)} items")
        print(f"Videos: {len(videos)}")
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video.title}")
        
        print(f"\n✅ Ready for next video!")
        
    else:
        print(f"\n❌ Video processing failed")

if __name__ == "__main__":
    asyncio.run(main())