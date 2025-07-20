"""
YouTube Processing Agent for KnowAgent

This agent handles all YouTube/video-related actions from our knowledge base:
- download_audio: Extract audio from YouTube videos
- transcribe_audio: Convert audio to text using local Whisper
- extract_timestamps: Create timestamped segments
- identify_speakers: Basic speaker diarization

This implements the "local Whisper" approach we discussed - no API costs!
"""

import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import json

# YouTube downloading
import yt_dlp

# Local Whisper for transcription
import whisper

# YouTube transcript API for captions
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    print("YouTube transcript API not available - install youtube-transcript-api")

# Audio processing
import numpy as np

# Our base agent
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent, ExecutionResult

# Transcript cleaning
try:
    from preprocessing.transcript_cleaner import TranscriptCleaner
    TRANSCRIPT_CLEANER_AVAILABLE = True
except ImportError:
    TRANSCRIPT_CLEANER_AVAILABLE = False
    print("Transcript cleaner not available")

# Topic classification
try:
    from classification.topic_classifier import DynamicTopicClassifier
    TOPIC_CLASSIFICATION_AVAILABLE = True
except ImportError:
    TOPIC_CLASSIFICATION_AVAILABLE = False
    print("Topic classification not available - install sklearn")

# Embeddings for topic classification
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Sentence transformers not available - install sentence-transformers")

class YouTubeAgent(BaseAgent):
    """
    Specialized agent for YouTube video processing.
    
    This agent implements the video processing actions from our
    ActionKnowledgeBase using LOCAL Whisper (no API costs).
    """
    
    def __init__(self):
        # Initialize local Whisper model
        print("Loading local Whisper model...")
        self.whisper_model = whisper.load_model("base")  # Good balance of speed/accuracy
        print("Whisper model loaded! (base model)")
        
        # Initialize embedding model for topic classification
        if EMBEDDINGS_AVAILABLE:
            print("Loading sentence transformer model for topic classification...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded!")
        else:
            self.embedding_model = None
        
        # Initialize topic classifier
        if TOPIC_CLASSIFICATION_AVAILABLE and self.embedding_model:
            self.topic_classifier = DynamicTopicClassifier(
                similarity_threshold=0.7,  # 70% similarity threshold as requested
                storage_path="topic_classifications"
            )
            print("Topic classifier initialized!")
        else:
            self.topic_classifier = None
        
        # Initialize transcript cleaner
        if TRANSCRIPT_CLEANER_AVAILABLE:
            self.transcript_cleaner = TranscriptCleaner()
            print("Transcript cleaner initialized!")
        else:
            self.transcript_cleaner = None
        
        # Storage for processed videos
        self.video_transcripts = {}
        self.audio_files = {}
        
        # yt-dlp configuration
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'mp3',  # Changed to mp3 for better Whisper compatibility
            'outtmpl': 'audio_%(id)s.%(ext)s',  # Simpler filename without special chars
            'quiet': True,
            'no_warnings': True,
        }
        
        super().__init__("YouTubeAgent")
    
    def _get_supported_actions(self) -> List[str]:
        """YouTube Agent supports these actions from the knowledge base"""
        actions = [
            "download_audio",
            "transcribe_audio", 
            "extract_timestamps",
            "identify_speakers",
            "classify_topics"
        ]
        
        if YOUTUBE_TRANSCRIPT_AVAILABLE:
            actions.append("get_captions")
            
        return actions
    
    async def execute_action(self, action: str, target: str, **kwargs) -> ExecutionResult:
        """
        Execute YouTube processing actions.
        """
        start_time = time.time()
        
        try:
            if action == "download_audio":
                url = target.get("url") if isinstance(target, dict) else target
                result = await self._download_audio(url, **kwargs)
            elif action == "transcribe_audio":
                url = target.get("url") if isinstance(target, dict) else target
                result = await self._transcribe_audio(url, **kwargs)
            elif action == "extract_timestamps":
                result = await self._extract_timestamps(target, **kwargs)
            elif action == "identify_speakers":
                result = await self._identify_speakers(target, **kwargs)
            elif action == "classify_topics":
                result = await self._classify_video_topics(target, **kwargs)
            elif action == "get_captions":
                url = target.get("url") if isinstance(target, dict) else target
                result = await self._get_captions(url)
            else:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message=f"Unknown action: {action}"
                )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Error in {action}: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _download_audio(self, youtube_url: str, **kwargs) -> ExecutionResult:
        """
        Download audio from YouTube video.
        
        This implements the 'download_audio' action from our knowledge base.
        """
        print(f"Downloading audio from: {youtube_url}")
        
        # Validate URL
        if not self._is_valid_youtube_url(youtube_url):
            return ExecutionResult(
                success=False,
                output=None,
                error_message="Invalid YouTube URL"
            )
        
        try:
            # Create temporary directory for downloads
            temp_dir = tempfile.mkdtemp()
            
            # Configure yt-dlp for this download
            ydl_opts = {
                **self.ydl_opts,
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            }
            
            video_info = {}
            audio_file_path = None
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first
                print("Extracting video information...")
                info = ydl.extract_info(youtube_url, download=False)
                
                video_info = {
                    "title": info.get('title', 'Unknown'),
                    "duration": info.get('duration', 0),
                    "uploader": info.get('uploader', 'Unknown'),
                    "upload_date": info.get('upload_date', 'Unknown'),
                    "view_count": info.get('view_count', 0),
                    "description": info.get('description', '')[:500],  # First 500 chars
                }
                
                # Check duration (limit to reasonable length)
                if video_info['duration'] > 3600:  # 1 hour limit
                    return ExecutionResult(
                        success=False,
                        output=None,
                        error_message=f"Video too long: {video_info['duration']/60:.1f} minutes (max 60 minutes)"
                    )
                
                print(f"Downloading: {video_info['title'][:50]}...")
                
                # Download the audio
                ydl.download([youtube_url])
                
                # Find the downloaded file
                for file in os.listdir(temp_dir):
                    if file.endswith(('.mp3', '.wav', '.m4a', '.webm')):
                        audio_file_path = os.path.join(temp_dir, file)
                        print(f"Found audio file: {audio_file_path}")
                        break
            
            if not audio_file_path or not os.path.exists(audio_file_path):
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message="Failed to download audio file"
                )
            
            # Store audio file info
            video_id = self._extract_video_id(youtube_url)
            self.audio_files[video_id] = {
                "file_path": audio_file_path,
                "video_info": video_info,
                "url": youtube_url,
                "download_timestamp": time.time()
            }
            
            result_data = {
                "video_id": video_id,
                "audio_file_path": audio_file_path,
                "video_info": video_info,
                "file_size_mb": os.path.getsize(audio_file_path) / (1024 * 1024),
                "download_timestamp": time.time()
            }
            
            print(f"Downloaded audio: {os.path.getsize(audio_file_path)/(1024*1024):.1f} MB")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "download_audio",
                    "duration_seconds": video_info['duration'],
                    "file_size_mb": result_data['file_size_mb']
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Audio download failed: {str(e)}"
            )
    
    async def _get_captions(self, youtube_url: str) -> ExecutionResult:
        """
        Try to get captions/subtitles from YouTube directly.
        
        This is much faster than audio transcription when available.
        """
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="YouTube transcript API not available"
            )
        
        video_id = self._extract_video_id(youtube_url)
        if not video_id:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="Could not extract video ID from URL"
            )
        
        try:
            print(f"Attempting to get captions for video: {video_id}")
            
            # Try to get transcript
            transcript_entries = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Convert to our format
            raw_segments = []
            for entry in transcript_entries:
                start_time = entry['start']
                duration = entry['duration']
                end_time = start_time + duration
                text = entry['text'].strip()
                
                raw_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "words": []  # Captions don't have word-level timestamps
                })
            
            # Clean transcript segments if cleaner is available
            if self.transcript_cleaner:
                print("Cleaning caption segments...")
                cleaned_segments = self.transcript_cleaner.clean_transcript(raw_segments)
                
                # Merge cleaned segments into optimal chunks
                merged_chunks = self.transcript_cleaner.merge_cleaned_segments(cleaned_segments, target_length=150)
                
                # Convert merged chunks back to segments format
                segments = []
                full_transcript = ""
                
                for chunk in merged_chunks:
                    segments.append({
                        "start": chunk['start_time'],
                        "end": chunk['end_time'],
                        "text": chunk['text'],
                        "words": [],
                        "cleaned": True,
                        "confidence": chunk['confidence'],
                        "segment_count": chunk['segment_count']
                    })
                    full_transcript += chunk['text'] + " "
                
                # Get cleaning stats
                cleaning_stats = self.transcript_cleaner.get_cleaning_stats(raw_segments, cleaned_segments)
                print(f"Transcript cleaning complete: {cleaning_stats['retention_ratio']:.1%} words retained, {cleaning_stats['average_confidence']:.2f} avg confidence")
            else:
                # Fallback to original format
                segments = raw_segments
                full_transcript = ""
                for segment in segments:
                    full_transcript += segment['text'] + " "
                cleaning_stats = None
            
            # Get video info
            video_info = {
                "title": f"Video {video_id}",
                "duration": segments[-1]["end"] if segments else 0,
                "source": "youtube_captions"
            }
            
            result_data = {
                "video_id": video_id,
                "transcript": full_transcript.strip(),
                "segments": segments,
                "language": "auto-detected",
                "total_duration": segments[-1]["end"] if segments else 0,
                "segment_count": len(segments),
                "word_count": len(full_transcript.split()),
                "transcription_timestamp": time.time(),
                "video_info": video_info,
                "source": "youtube_captions",
                "transcript_cleaned": self.transcript_cleaner is not None,
                "cleaning_stats": cleaning_stats if self.transcript_cleaner else None
            }
            
            print(f"Successfully got captions: {len(segments)} segments, {len(full_transcript.split())} words")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "get_captions",
                    "segments": len(segments),
                    "words": len(full_transcript.split()),
                    "source": "youtube_captions"
                }
            )
            
        except Exception as e:
            print(f"Captions not available: {e}")
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Captions not available: {str(e)}"
            )

    async def _transcribe_audio(self, target: str, **kwargs) -> ExecutionResult:
        """
        Transcribe audio to text using hybrid approach:
        1. Try YouTube captions first (instant, no download needed)
        2. Fallback to local Whisper if captions not available
        
        This implements the 'transcribe_audio' action from our knowledge base.
        """
        print(f"Transcribing audio: {target}")
        
        # Step 1: Try to get captions first (much faster!)
        if YOUTUBE_TRANSCRIPT_AVAILABLE and ('youtube.com' in target or 'youtu.be' in target):
            print("Trying YouTube captions first (faster than audio transcription)...")
            captions_result = await self._get_captions(target)
            
            if captions_result.success:
                print("Got captions from YouTube - no audio download needed!")
                
                # Add topic classification to captions
                if self.topic_classifier and self.embedding_model:
                    transcript_data = captions_result.output
                    segments = transcript_data.get("segments", [])
                    video_id = transcript_data.get("video_id")
                    
                    print("Classifying caption segments into topics...")
                    topic_assignments = {}
                    
                    for segment in segments:
                        content_id = f"{video_id}_segment_{segment['start']:.1f}"
                        segment_text = segment['text']
                        
                        # Generate embedding for segment
                        segment_embedding = self.embedding_model.encode([segment_text])[0]
                        
                        assigned_topics = self.topic_classifier.add_content(
                            content_id=content_id,
                            content_type="video",
                            source_id=video_id,
                            text=segment_text,
                            embedding=segment_embedding,
                            metadata={
                                "start_time": segment['start'],
                                "end_time": segment['end'],
                                "duration": segment['end'] - segment['start'],
                                "word_count": len(segment_text.split()),
                                "source": "youtube_captions",
                                "full_transcript": transcript_data.get("transcript", "")
                            }
                        )
                        topic_assignments[content_id] = assigned_topics
                        segment['topics'] = assigned_topics
                    
                    # Update result with topic assignments
                    captions_result.output["topic_assignments"] = topic_assignments
                    captions_result.metadata["audio_cleaned"] = True  # No audio file to clean
                    
                    print(f"Classified {len(segments)} caption segments into topics")
                
                return captions_result
            else:
                print("Captions not available, falling back to audio transcription...")
        
        # Step 2: Fallback to local Whisper transcription
        print("Using local Whisper transcription...")
        
        # Get previous results or audio file info
        previous_results = kwargs.get('previous_results', {})
        
        # Find the audio file
        audio_file_path = None
        video_info = {}
        video_id = None
        
        # Check previous results for download_audio output
        for key, result in previous_results.items():
            if 'download_audio' in key and isinstance(result, dict):
                audio_file_path = result.get('audio_file_path')
                video_info = result.get('video_info', {})
                video_id = result.get('video_id')
                break
        
        # Fallback: check our cache
        if not audio_file_path:
            video_id = self._extract_video_id(target) if 'youtube.com' in target or 'youtu.be' in target else target
            if video_id in self.audio_files:
                cached_data = self.audio_files[video_id]
                audio_file_path = cached_data['file_path']
                video_info = cached_data['video_info']
        
        if not audio_file_path or not os.path.exists(audio_file_path):
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No audio file found for transcription"
            )
        
        try:
            print("Running local Whisper transcription...")
            print("(This may take a few minutes depending on audio length)")
            print(f"Audio file path: {audio_file_path}")
            print(f"File exists: {os.path.exists(audio_file_path)}")
            
            if not os.path.exists(audio_file_path):
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message=f"Audio file not found: {audio_file_path}"
                )
            
            # Normalize the path for Windows compatibility
            normalized_path = os.path.normpath(audio_file_path)
            print(f"Normalized path: {normalized_path}")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                normalized_path,
                verbose=False,
                word_timestamps=True  # Get word-level timestamps
            )
            
            # Extract transcript and segments
            raw_segments = []
            for segment in result.get('segments', []):
                raw_segments.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text'].strip(),
                    "words": segment.get('words', [])
                })
            
            # Clean transcript segments if cleaner is available
            if self.transcript_cleaner:
                print("Cleaning transcript segments...")
                cleaned_segments = self.transcript_cleaner.clean_transcript(raw_segments)
                
                # Merge cleaned segments into optimal chunks
                merged_chunks = self.transcript_cleaner.merge_cleaned_segments(cleaned_segments, target_length=150)
                
                # Convert merged chunks back to segments format
                segments = []
                full_transcript = ""
                
                for chunk in merged_chunks:
                    segments.append({
                        "start": chunk['start_time'],
                        "end": chunk['end_time'],
                        "text": chunk['text'],
                        "words": [],  # Word-level timestamps lost in cleaning
                        "cleaned": True,
                        "confidence": chunk['confidence'],
                        "segment_count": chunk['segment_count']
                    })
                    full_transcript += chunk['text'] + " "
                
                # Get cleaning stats
                cleaning_stats = self.transcript_cleaner.get_cleaning_stats(raw_segments, cleaned_segments)
                print(f"Transcript cleaning complete: {cleaning_stats['retention_ratio']:.1%} words retained, {cleaning_stats['average_confidence']:.2f} avg confidence")
            else:
                # Fallback to original format
                segments = raw_segments
                full_transcript = result['text']
                cleaning_stats = None
            
            # Store transcript
            if video_id:
                self.video_transcripts[video_id] = {
                    "transcript": full_transcript,
                    "segments": segments,
                    "language": result.get('language', 'unknown'),
                    "transcription_timestamp": time.time()
                }
            
            # Automatically classify topics if topic classifier is available
            topic_assignments = {}
            if self.topic_classifier and self.embedding_model:
                print("Classifying transcript segments into topics...")
                for segment in segments:
                    content_id = f"{video_id}_segment_{segment['start']:.1f}"
                    segment_text = segment['text']
                    
                    # Generate embedding for segment
                    segment_embedding = self.embedding_model.encode([segment_text])[0]
                    
                    assigned_topics = self.topic_classifier.add_content(
                        content_id=content_id,
                        content_type="video",
                        source_id=video_id,
                        text=segment_text,
                        embedding=segment_embedding,
                        metadata={
                            "start_time": segment['start'],
                            "end_time": segment['end'],
                            "duration": segment['end'] - segment['start'],
                            "word_count": len(segment_text.split())
                        }
                    )
                    topic_assignments[content_id] = assigned_topics
                    # Add topic info to segment
                    segment['topics'] = assigned_topics
                
                print(f"Classified {len(segments)} segments into topics")
            
            result_data = {
                "video_id": video_id,
                "transcript": full_transcript,
                "segments": segments,
                "language": result.get('language', 'unknown'),
                "total_duration": segments[-1]['end'] if segments else 0,
                "segment_count": len(segments),
                "word_count": len(full_transcript.split()),
                "topic_assignments": topic_assignments,
                "transcription_timestamp": time.time(),
                "video_info": video_info,
                "transcript_cleaned": self.transcript_cleaner is not None,
                "cleaning_stats": cleaning_stats if self.transcript_cleaner else None
            }
            
            print(f"Transcription complete: {len(segments)} segments, {len(full_transcript.split())} words")
            
            # IMPORTANT: Clean up audio file to save space
            try:
                if os.path.exists(audio_file_path):
                    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
                    os.remove(audio_file_path)
                    print(f"Cleaned up audio file: {audio_file_path} ({file_size_mb:.1f}MB saved)")
                    
                    # Remove from audio cache since file is deleted
                    if video_id in self.audio_files:
                        del self.audio_files[video_id]
                        print("Removed audio file from cache - keeping only text data")
                        
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up audio file: {cleanup_error}")
                # Continue anyway - transcription was successful
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "transcribe_audio",
                    "segments": len(segments),
                    "words": len(full_transcript.split()),
                    "language": result.get('language', 'unknown'),
                    "audio_cleaned": True  # Indicate we cleaned up the file
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            if "cannot find the file specified" in error_msg.lower():
                error_msg = ("Transcription failed: ffmpeg is required for audio processing. "
                           "Please install ffmpeg and add it to your system PATH. "
                           "Download from: https://ffmpeg.org/download.html")
            
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Transcription failed: {error_msg}"
            )
    
    async def _extract_timestamps(self, target: str, **kwargs) -> ExecutionResult:
        """
        Create timestamped segments from transcript.
        
        This implements the 'extract_timestamps' action from our knowledge base.
        """
        print(f"Extracting timestamps for: {target}")
        
        # Get previous results
        previous_results = kwargs.get('previous_results', {})
        
        # Find transcript data
        transcript_data = None
        video_id = None
        
        for key, result in previous_results.items():
            if 'transcribe_audio' in key and isinstance(result, dict):
                transcript_data = result
                video_id = result.get('video_id')
                break
        
        if not transcript_data:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No transcript found for timestamp extraction"
            )
        
        try:
            segments = transcript_data.get('segments', [])
            
            # Create topic-based timestamp segments (group by content similarity)
            timestamped_segments = []
            current_topic = ""
            current_start = 0
            current_text = ""
            
            for i, segment in enumerate(segments):
                segment_text = segment['text'].strip()
                
                # Simple topic detection (you could enhance this with NLP)
                if (len(current_text) > 200 or  # Max segment length
                    self._is_topic_boundary(current_text, segment_text) or
                    i == len(segments) - 1):  # Last segment
                    
                    if current_text:
                        timestamped_segments.append({
                            "segment_id": len(timestamped_segments),
                            "start_time": current_start,
                            "end_time": segment['end'],
                            "duration": segment['end'] - current_start,
                            "text": current_text.strip(),
                            "word_count": len(current_text.split()),
                            "topic_keywords": self._extract_keywords(current_text)
                        })
                    
                    # Start new segment
                    current_start = segment['start']
                    current_text = segment_text
                else:
                    # Add to current segment
                    current_text += " " + segment_text
            
            result_data = {
                "video_id": video_id,
                "timestamped_segments": timestamped_segments,
                "total_segments": len(timestamped_segments),
                "total_duration": segments[-1]['end'] if segments else 0,
                "average_segment_length": sum(s['duration'] for s in timestamped_segments) / len(timestamped_segments) if timestamped_segments else 0,
                "timestamp_extraction_time": time.time()
            }
            
            print(f"Created {len(timestamped_segments)} timestamped segments")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "extract_timestamps",
                    "segment_count": len(timestamped_segments),
                    "total_duration": result_data['total_duration']
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Timestamp extraction failed: {str(e)}"
            )
    
    async def _identify_speakers(self, target: str, **kwargs) -> ExecutionResult:
        """
        Basic speaker identification.
        
        This implements the 'identify_speakers' action from our knowledge base.
        Note: This is a simplified implementation. Real speaker diarization
        would require specialized models.
        """
        print(f"Identifying speakers in: {target}")
        
        # For this demo, we'll do basic speaker detection based on audio patterns
        # In production, you'd use pyannote.audio or similar
        
        return ExecutionResult(
            success=True,
            output={
                "speaker_count": 1,  # Simplified: assume single speaker
                "speakers": [{"speaker_id": "SPEAKER_00", "confidence": 0.9}],
                "note": "Basic speaker detection - would use pyannote.audio in production"
            },
            metadata={
                "action": "identify_speakers",
                "method": "simplified"
            }
        )
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL"""
        youtube_patterns = [
            'youtube.com/watch',
            'youtu.be/',
            'youtube.com/embed/',
            'youtube.com/v/'
        ]
        return any(pattern in url for pattern in youtube_patterns)
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        if 'youtu.be/' in url:
            return url.split('youtu.be/')[-1].split('?')[0]
        elif 'youtube.com' in url:
            if 'v=' in url:
                return url.split('v=')[-1].split('&')[0]
        return url  # Fallback: use full URL as ID
    
    def _is_topic_boundary(self, current_text: str, new_text: str) -> bool:
        """Simple topic boundary detection"""
        # Look for topic transition indicators
        transition_words = ['however', 'meanwhile', 'next', 'now', 'moving on', 'in conclusion']
        new_lower = new_text.lower()
        return any(word in new_lower for word in transition_words)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text"""
        # Basic keyword extraction (would use NLP in production)
        words = text.lower().split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        # Return top 5 most frequent
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(5)]
    
    async def _classify_video_topics(self, target: str, **kwargs) -> ExecutionResult:
        """
        Classify video transcript segments into topics using the topic classifier.
        
        This implements the 'classify_topics' action from our knowledge base.
        """
        print(f"Classifying video topics for: {target}")
        
        if not self.topic_classifier or not self.embedding_model:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="Topic classifier or embedding model not available"
            )
        
        # Get previous results
        previous_results = kwargs.get('previous_results', {})
        
        # Find transcript data (from either transcribe_audio or get_captions)
        transcript_data = None
        video_id = None
        
        for key, result in previous_results.items():
            if ('transcribe_audio' in key or 'get_captions' in key) and isinstance(result, dict):
                transcript_data = result
                video_id = result.get('video_id')
                break
        
        if not transcript_data:
            return ExecutionResult(
                success=False,
                output=None,
                error_message="No transcript found for topic classification"
            )
        
        try:
            segments = transcript_data.get('segments', [])
            topic_assignments = {}
            
            # If segments don't already have topics, classify them
            if not any('topics' in segment for segment in segments):
                print("Classifying transcript segments into topics...")
                for segment in segments:
                    content_id = f"{video_id}_segment_{segment['start']:.1f}"
                    segment_text = segment['text']
                    
                    # Generate embedding for segment
                    segment_embedding = self.embedding_model.encode([segment_text])[0]
                    
                    assigned_topics = self.topic_classifier.add_content(
                        content_id=content_id,
                        content_type="video",
                        source_id=video_id,
                        text=segment_text,
                        embedding=segment_embedding,
                        metadata={
                            "start_time": segment['start'],
                            "end_time": segment['end'],
                            "duration": segment['end'] - segment['start'],
                            "word_count": len(segment_text.split())
                        }
                    )
                    topic_assignments[content_id] = assigned_topics
                    segment['topics'] = assigned_topics
            else:
                # Extract existing topic assignments
                for segment in segments:
                    content_id = f"{video_id}_segment_{segment['start']:.1f}"
                    topic_assignments[content_id] = segment.get('topics', [])
            
            # Get topic hierarchy and statistics
            topic_hierarchy = self.topic_classifier.get_topic_hierarchy()
            cross_topic_content = self.topic_classifier.get_cross_topic_content()
            classification_stats = self.topic_classifier.get_classification_stats()
            
            result_data = {
                "video_id": video_id,
                "topic_assignments": topic_assignments,
                "topic_hierarchy": topic_hierarchy,
                "cross_topic_content": cross_topic_content,
                "classification_stats": classification_stats,
                "total_segments_classified": len(topic_assignments),
                "classification_timestamp": time.time()
            }
            
            print(f"Classified {len(topic_assignments)} segments into {len(topic_hierarchy)} topics")
            
            return ExecutionResult(
                success=True,
                output=result_data,
                metadata={
                    "action": "classify_topics",
                    "segments_classified": len(topic_assignments),
                    "topics_discovered": len(topic_hierarchy),
                    "cross_topic_items": len(cross_topic_content)
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"Video topic classification failed: {str(e)}"
            )

# Test the YouTube agent
if __name__ == "__main__":
    async def test_youtube_agent():
        agent = YouTubeAgent()
        print(f"YouTube Agent initialized with actions: {agent.supported_actions}")
        
        print("\nYouTube Agent ready for testing!")
        print("To test with real YouTube video:")
        print("1. Find a short YouTube video URL")
        print("2. Call: result = await agent.execute_action('download_audio', 'https://youtube.com/watch?v=...')")
    
    asyncio.run(test_youtube_agent())