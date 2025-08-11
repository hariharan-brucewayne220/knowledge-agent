#!/usr/bin/env python3
"""
YouTube Content Monitor
Monitors YouTube playlists and channels for new videos
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
import re
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

@dataclass
class YouTubeVideoInfo:
    """Information about a YouTube video"""
    video_id: str
    title: str
    description: str
    published_at: str
    channel_id: str
    channel_title: str
    playlist_id: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    thumbnail_url: Optional[str] = None
    
    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"
    
    @property
    def published_timestamp(self) -> float:
        """Convert published_at to timestamp"""
        try:
            dt = datetime.fromisoformat(self.published_at.replace('Z', '+00:00'))
            return dt.timestamp()
        except:
            return time.time()

@dataclass
class YouTubeMonitorConfig:
    """Configuration for YouTube monitoring"""
    api_key: str
    check_interval: int = 3600  # 1 hour
    max_results: int = 50
    include_shorts: bool = True
    min_duration_seconds: int = 60  # Skip videos shorter than 1 minute
    keywords_filter: Optional[List[str]] = None  # Only include videos with these keywords
    exclude_keywords: Optional[List[str]] = None  # Exclude videos with these keywords

class YouTubeAPIClient:
    """YouTube Data API client with rate limiting and error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 10000  # Daily quota limit
        self.last_rate_limit_reset = time.time()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with error handling and rate limiting"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Check rate limit
        if self.rate_limit_remaining <= 0:
            if time.time() - self.last_rate_limit_reset < 86400:  # 24 hours
                raise Exception("YouTube API quota exceeded")
            else:
                self.rate_limit_remaining = 10000  # Reset daily quota
                self.last_rate_limit_reset = time.time()
        
        params['key'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                # Update rate limit info
                if 'quota_user' in response.headers:
                    self.rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', 0))
                
                if response.status != 200:
                    error_msg = data.get('error', {}).get('message', 'Unknown error')
                    raise Exception(f"YouTube API error: {error_msg}")
                
                return data
                
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {e}")
    
    async def get_playlist_videos(self, playlist_id: str, max_results: int = 50) -> List[YouTubeVideoInfo]:
        """Get videos from a playlist"""
        videos = []
        next_page_token = None
        
        while len(videos) < max_results:
            params = {
                'part': 'snippet,contentDetails',
                'playlistId': playlist_id,
                'maxResults': min(50, max_results - len(videos)),
            }
            
            if next_page_token:
                params['pageToken'] = next_page_token
            
            try:
                data = await self._make_request('playlistItems', params)
                
                for item in data.get('items', []):
                    snippet = item['snippet']
                    content_details = item.get('contentDetails', {})
                    
                    video = YouTubeVideoInfo(
                        video_id=snippet['resourceId']['videoId'],
                        title=snippet['title'],
                        description=snippet['description'],
                        published_at=snippet['publishedAt'],
                        channel_id=snippet['channelId'],
                        channel_title=snippet['channelTitle'],
                        playlist_id=playlist_id,
                        thumbnail_url=snippet.get('thumbnails', {}).get('medium', {}).get('url')
                    )
                    videos.append(video)
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching playlist {playlist_id}: {e}")
                break
        
        return videos
    
    async def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[YouTubeVideoInfo]:
        """Get recent videos from a channel"""
        try:
            # First, get the uploads playlist ID
            params = {
                'part': 'contentDetails',
                'id': channel_id
            }
            
            data = await self._make_request('channels', params)
            
            if not data.get('items'):
                raise Exception(f"Channel {channel_id} not found")
            
            uploads_playlist_id = data['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from uploads playlist
            return await self.get_playlist_videos(uploads_playlist_id, max_results)
            
        except Exception as e:
            logger.error(f"Error fetching channel {channel_id}: {e}")
            return []
    
    async def get_video_details(self, video_ids: List[str]) -> Dict[str, Dict]:
        """Get detailed information for specific videos"""
        if not video_ids:
            return {}
        
        details = {}
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            
            params = {
                'part': 'contentDetails,statistics',
                'id': ','.join(batch)
            }
            
            try:
                data = await self._make_request('videos', params)
                
                for item in data.get('items', []):
                    video_id = item['id']
                    content_details = item.get('contentDetails', {})
                    statistics = item.get('statistics', {})
                    
                    details[video_id] = {
                        'duration': content_details.get('duration'),
                        'view_count': int(statistics.get('viewCount', 0)),
                        'like_count': int(statistics.get('likeCount', 0)),
                        'comment_count': int(statistics.get('commentCount', 0))
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching video details: {e}")
        
        return details

def parse_youtube_url(url: str) -> Dict[str, Optional[str]]:
    """Parse YouTube URL to extract playlist ID, channel ID, or video ID"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    
    result = {
        'type': None,
        'id': None,
        'playlist_id': None,
        'video_id': None,
        'channel_id': None
    }
    
    if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
        # Playlist URL
        if 'list' in query:
            result['type'] = 'playlist'
            result['id'] = query['list'][0]
            result['playlist_id'] = query['list'][0]
        
        # Channel URL
        elif '/channel/' in parsed.path:
            result['type'] = 'channel'
            result['id'] = parsed.path.split('/channel/')[-1]
            result['channel_id'] = result['id']
        
        elif '/c/' in parsed.path or '/user/' in parsed.path:
            # These would need additional API calls to resolve to channel ID
            result['type'] = 'channel_name'
            result['id'] = parsed.path.split('/')[-1]
        
        # Video URL
        elif 'v' in query:
            result['type'] = 'video'
            result['id'] = query['v'][0]
            result['video_id'] = result['id']
        
        elif 'youtu.be' in parsed.netloc:
            result['type'] = 'video'
            result['id'] = parsed.path.strip('/')
            result['video_id'] = result['id']
    
    return result

def parse_duration(duration_str: str) -> int:
    """Parse YouTube duration string (PT1H2M3S) to seconds"""
    if not duration_str:
        return 0
    
    # Remove PT prefix
    duration_str = duration_str[2:] if duration_str.startswith('PT') else duration_str
    
    hours = 0
    minutes = 0
    seconds = 0
    
    # Parse hours
    if 'H' in duration_str:
        hours = int(duration_str.split('H')[0])
        duration_str = duration_str.split('H')[1]
    
    # Parse minutes
    if 'M' in duration_str:
        minutes = int(duration_str.split('M')[0])
        duration_str = duration_str.split('M')[1]
    
    # Parse seconds
    if 'S' in duration_str:
        seconds = int(duration_str.split('S')[0])
    
    return hours * 3600 + minutes * 60 + seconds

class YouTubePlaylistMonitor:
    """
    Monitor YouTube playlists and channels for new videos
    """
    
    def __init__(self, config: YouTubeMonitorConfig, callback: Callable[[List[YouTubeVideoInfo]], None]):
        self.config = config
        self.callback = callback
        self.api_client = YouTubeAPIClient(config.api_key)
        
        # State tracking
        self.monitored_sources: Dict[str, Dict] = {}  # source_id -> config
        self.known_videos: Set[str] = set()  # video IDs we've already seen
        self.is_running = False
        self.last_check_times: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'new_videos_found': 0,
            'api_calls_made': 0,
            'errors_encountered': 0,
            'last_successful_check': None
        }
    
    def add_playlist(self, name: str, playlist_id: str, **kwargs):
        """Add a playlist to monitor"""
        self.monitored_sources[name] = {
            'type': 'playlist',
            'id': playlist_id,
            'url': f"https://www.youtube.com/playlist?list={playlist_id}",
            'enabled': True,
            **kwargs
        }
        logger.info(f"📺 Added playlist monitor: {name}")
    
    def add_channel(self, name: str, channel_id: str, **kwargs):
        """Add a channel to monitor"""
        self.monitored_sources[name] = {
            'type': 'channel',
            'id': channel_id,
            'url': f"https://www.youtube.com/channel/{channel_id}",
            'enabled': True,
            **kwargs
        }
        logger.info(f"📺 Added channel monitor: {name}")
    
    def add_source_from_url(self, name: str, url: str, **kwargs):
        """Add a source by parsing its URL"""
        parsed = parse_youtube_url(url)
        
        if parsed['type'] == 'playlist':
            self.add_playlist(name, parsed['playlist_id'], **kwargs)
        elif parsed['type'] == 'channel':
            self.add_channel(name, parsed['channel_id'], **kwargs)
        else:
            raise ValueError(f"Unsupported YouTube URL: {url}")
    
    def remove_source(self, name: str):
        """Remove a monitored source"""
        if name in self.monitored_sources:
            del self.monitored_sources[name]
            logger.info(f"🗑️ Removed monitor: {name}")
    
    async def start_monitoring(self):
        """Start monitoring all configured sources"""
        if self.is_running:
            logger.warning("YouTube monitor is already running")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting YouTube monitor for {len(self.monitored_sources)} sources")
        
        # Initialize known videos from first scan
        await self._initial_scan()
        
        # Start monitoring loop
        await self._monitoring_loop()
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("⏹️ Stopping YouTube monitor")
        self.is_running = False
    
    async def _initial_scan(self):
        """Initial scan to populate known videos"""
        logger.info("🔍 Performing initial scan of YouTube sources...")
        
        async with self.api_client:
            for source_name, source_config in self.monitored_sources.items():
                try:
                    videos = await self._fetch_source_videos(source_config)
                    
                    # Add to known videos without triggering callback
                    for video in videos:
                        self.known_videos.add(video.video_id)
                    
                    logger.info(f"📊 {source_name}: Found {len(videos)} existing videos")
                    
                except Exception as e:
                    logger.error(f"Error in initial scan of {source_name}: {e}")
                    self.stats['errors_encountered'] += 1
        
        logger.info(f"✅ Initial scan complete: {len(self.known_videos)} known videos")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                async with self.api_client:
                    await self._check_all_sources()
                
                # Wait before next check
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stats['errors_encountered'] += 1
                await asyncio.sleep(60)  # Brief pause on error
    
    async def _check_all_sources(self):
        """Check all monitored sources for new videos"""
        new_videos = []
        
        for source_name, source_config in self.monitored_sources.items():
            if not source_config.get('enabled', True):
                continue
            
            try:
                logger.debug(f"🔍 Checking {source_name}")
                
                videos = await self._fetch_source_videos(source_config)
                source_new_videos = []
                
                for video in videos:
                    if video.video_id not in self.known_videos:
                        # Apply filters
                        if self._should_include_video(video):
                            source_new_videos.append(video)
                            self.known_videos.add(video.video_id)
                
                if source_new_videos:
                    new_videos.extend(source_new_videos)
                    logger.info(f"🆕 {source_name}: Found {len(source_new_videos)} new videos")
                
                self.last_check_times[source_name] = time.time()
                
            except Exception as e:
                logger.error(f"Error checking {source_name}: {e}")
                self.stats['errors_encountered'] += 1
        
        self.stats['total_checks'] += 1
        
        if new_videos:
            self.stats['new_videos_found'] += len(new_videos)
            self.stats['last_successful_check'] = time.time()
            
            # Get additional details for new videos
            try:
                video_ids = [video.video_id for video in new_videos]
                details = await self.api_client.get_video_details(video_ids)
                
                # Update video info with details
                for video in new_videos:
                    if video.video_id in details:
                        detail = details[video.video_id]
                        video.duration = detail.get('duration')
                        video.view_count = detail.get('view_count')
                        video.like_count = detail.get('like_count')
                
            except Exception as e:
                logger.error(f"Error fetching video details: {e}")
            
            # Call callback with new videos
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(new_videos)
                else:
                    self.callback(new_videos)
            except Exception as e:
                logger.error(f"Error in new video callback: {e}")
    
    async def _fetch_source_videos(self, source_config: Dict) -> List[YouTubeVideoInfo]:
        """Fetch videos from a single source"""
        source_type = source_config['type']
        source_id = source_config['id']
        
        self.stats['api_calls_made'] += 1
        
        if source_type == 'playlist':
            return await self.api_client.get_playlist_videos(source_id, self.config.max_results)
        elif source_type == 'channel':
            return await self.api_client.get_channel_videos(source_id, self.config.max_results)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _should_include_video(self, video: YouTubeVideoInfo) -> bool:
        """Apply filters to determine if video should be included"""
        
        # Duration filter
        if video.duration:
            duration_seconds = parse_duration(video.duration)
            if duration_seconds < self.config.min_duration_seconds:
                return False
        
        # Keyword filters
        if self.config.keywords_filter:
            text = f"{video.title} {video.description}".lower()
            if not any(keyword.lower() in text for keyword in self.config.keywords_filter):
                return False
        
        if self.config.exclude_keywords:
            text = f"{video.title} {video.description}".lower()
            if any(keyword.lower() in text for keyword in self.config.exclude_keywords):
                return False
        
        # Shorts filter
        if not self.config.include_shorts and video.duration:
            duration_seconds = parse_duration(video.duration)
            if duration_seconds <= 60:  # Typical shorts duration
                return False
        
        return True
    
    def get_status(self) -> Dict:
        """Get monitor status and statistics"""
        return {
            'is_running': self.is_running,
            'monitored_sources': len(self.monitored_sources),
            'known_videos': len(self.known_videos),
            'config': {
                'check_interval': self.config.check_interval,
                'max_results': self.config.max_results,
                'include_shorts': self.config.include_shorts,
                'min_duration_seconds': self.config.min_duration_seconds
            },
            'statistics': self.stats,
            'last_check_times': dict(self.last_check_times),
            'sources': {
                name: {
                    'type': config['type'],
                    'id': config['id'],
                    'enabled': config.get('enabled', True),
                    'url': config.get('url')
                }
                for name, config in self.monitored_sources.items()
            }
        }

# Example usage
async def example_youtube_monitoring():
    """Example of YouTube monitoring usage"""
    
    # You would need a real YouTube API key
    API_KEY = "your-youtube-api-key-here"
    
    def handle_new_videos(videos: List[YouTubeVideoInfo]):
        """Handle newly discovered videos"""
        for video in videos:
            print(f"🆕 New video: {video.title}")
            print(f"   Channel: {video.channel_title}")
            print(f"   Published: {video.published_at}")
            print(f"   URL: {video.url}")
            print()
    
    config = YouTubeMonitorConfig(
        api_key=API_KEY,
        check_interval=300,  # 5 minutes
        max_results=20,
        include_shorts=False,
        min_duration_seconds=180,  # 3 minutes minimum
        keywords_filter=["tutorial", "education", "science", "research"]
    )
    
    monitor = YouTubePlaylistMonitor(config, handle_new_videos)
    
    # Add sources to monitor
    monitor.add_source_from_url("AI Research", "https://www.youtube.com/playlist?list=PLrAXtmrdJcz3YNrXe2kk8x8xJ5v0Y1WNS")
    monitor.add_channel("3Blue1Brown", "UCYO_jab_esuFRV4b17AJtAw")
    
    print("🚀 Starting YouTube monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitor...")
        await monitor.stop_monitoring()
        print("✅ Monitor stopped")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_youtube_monitoring())