"""
AWS Lambda Function for YouTube Monitoring
Runs daily to check YouTube channels for new videos
Stores results in S3 for local system to process
"""

import json
import boto3
import requests
import os
from datetime import datetime, timedelta
from typing import List, Dict

def lambda_handler(event, context):
    """
    AWS Lambda handler for YouTube monitoring
    Triggered daily by EventBridge
    """
    
    # Get environment variables (set in Lambda config)
    youtube_api_key = os.environ['YOUTUBE_API_KEY']
    s3_bucket = os.environ['S3_BUCKET_NAME']
    
    # Initialize AWS services
    s3_client = boto3.client('s3')
    
    try:
        # Load monitoring configuration
        monitored_channels = [
            {
                'name': 'MIT OpenCourseWare',
                'channel_id': 'UCEBb1b_L6zDS3xTUrIALZOw',
                'keywords': ['AI', 'machine learning', 'computer science']
            },
            {
                'name': 'DeepMind',
                'channel_id': 'UCP7jMXSY2xbc3KCAE0MHQ-A', 
                'keywords': ['AI', 'research', 'neural networks']
            }
        ]
        
        new_videos_found = []
        
        # Check each monitored channel
        for channel in monitored_channels:
            print(f"Checking channel: {channel['name']}")
            
            # Get recent videos from channel
            videos = get_recent_videos(
                youtube_api_key, 
                channel['channel_id'],
                days_back=1  # Only check last 24 hours
            )
            
            # Filter by keywords if specified
            if channel.get('keywords'):
                videos = filter_videos_by_keywords(videos, channel['keywords'])
            
            # Add channel info to videos
            for video in videos:
                video['monitoring_source'] = channel['name']
                video['found_at'] = datetime.utcnow().isoformat()
                
            new_videos_found.extend(videos)
        
        # Save results to S3
        if new_videos_found:
            result = {
                'timestamp': datetime.utcnow().isoformat(),
                'videos_found': len(new_videos_found),
                'videos': new_videos_found
            }
            
            # Save to S3 with timestamp in filename
            s3_key = f"youtube-monitoring/{datetime.utcnow().strftime('%Y/%m/%d')}/videos-{int(datetime.utcnow().timestamp())}.json"
            
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=json.dumps(result, indent=2),
                ContentType='application/json'
            )
            
            print(f"Found {len(new_videos_found)} new videos, saved to s3://{s3_bucket}/{s3_key}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Found {len(new_videos_found)} new videos',
                    'videos': new_videos_found,
                    's3_location': f's3://{s3_bucket}/{s3_key}'
                })
            }
        else:
            print("No new videos found")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No new videos found'})
            }
            
    except Exception as e:
        print(f"Error in YouTube monitoring: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def get_recent_videos(api_key: str, channel_id: str, days_back: int = 1) -> List[Dict]:
    """Get recent videos from a YouTube channel"""
    
    # Calculate date for filtering (24 hours ago)
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    cutoff_rfc3339 = cutoff_date.isoformat() + 'Z'
    
    # Get channel's uploads playlist
    channel_url = f"https://www.googleapis.com/youtube/v3/channels"
    channel_params = {
        'part': 'contentDetails',
        'id': channel_id,
        'key': api_key
    }
    
    channel_response = requests.get(channel_url, params=channel_params)
    channel_data = channel_response.json()
    
    if not channel_data.get('items'):
        return []
    
    uploads_playlist_id = channel_data['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    
    # Get recent videos from uploads playlist
    playlist_url = f"https://www.googleapis.com/youtube/v3/playlistItems"
    playlist_params = {
        'part': 'snippet',
        'playlistId': uploads_playlist_id,
        'maxResults': 10,  # Check last 10 videos
        'key': api_key
    }
    
    playlist_response = requests.get(playlist_url, params=playlist_params)
    playlist_data = playlist_response.json()
    
    recent_videos = []
    
    for item in playlist_data.get('items', []):
        snippet = item['snippet']
        published_at = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
        
        # Only include videos published in the last 24 hours
        if published_at.replace(tzinfo=None) > cutoff_date:
            video_info = {
                'video_id': snippet['resourceId']['videoId'],
                'title': snippet['title'],
                'description': snippet['description'][:500],  # Truncate description
                'published_at': snippet['publishedAt'],
                'channel_title': snippet['channelTitle'],
                'url': f"https://www.youtube.com/watch?v={snippet['resourceId']['videoId']}"
            }
            recent_videos.append(video_info)
    
    return recent_videos

def filter_videos_by_keywords(videos: List[Dict], keywords: List[str]) -> List[Dict]:
    """Filter videos that contain any of the specified keywords"""
    filtered_videos = []
    
    for video in videos:
        title_lower = video['title'].lower()
        description_lower = video['description'].lower()
        
        # Check if any keyword is in title or description
        for keyword in keywords:
            if keyword.lower() in title_lower or keyword.lower() in description_lower:
                filtered_videos.append(video)
                break  # Found match, no need to check other keywords
    
    return filtered_videos