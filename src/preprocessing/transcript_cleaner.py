"""
Transcript Cleaning Pipeline
Removes noise, filler words, and improves chunking quality
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class CleanedSegment:
    """Cleaned transcript segment"""
    original_start: float
    original_end: float
    cleaned_text: str
    confidence: float
    words_removed: int
    original_text: str

class TranscriptCleaner:
    """
    Clean noisy video transcripts for better chunking and search
    """
    
    def __init__(self):
        # Common filler words and false starts
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 
            'sort of', 'kind of', 'basically', 'actually', 'literally',
            'well', 'so', 'right', 'okay', 'alright', 'yeah', 'yes',
            'no', 'hmm', 'mmm', 'oh', 'wow', 'cool', 'nice', 'great'
        }
        
        # Repetition patterns
        self.repetition_patterns = [
            r'\b(\w+)\s+\1\b',  # word word (direct repetition)
            r'\b(\w+)\s+(\w+)\s+\1\s+\2\b',  # word1 word2 word1 word2
        ]
        
        # False start patterns
        self.false_start_patterns = [
            r'\b(\w+)\s*-+\s*(\w+)',  # word - word (correction)
            r'\b(\w+)\s+no\s+(\w+)',  # word no word (correction)
            r'\b(\w+)\s+I\s+mean\s+(\w+)',  # word I mean word
        ]
        
        # Incomplete sentence patterns
        self.incomplete_patterns = [
            r'\.\.\.',  # trailing dots
            r'-+$',  # trailing dashes
            r'\b(and|but|so|then)\s*$',  # trailing connectors
        ]

    def clean_transcript(self, segments: List[Dict]) -> List[CleanedSegment]:
        """
        Clean raw transcript segments
        """
        cleaned_segments = []
        
        for segment in segments:
            original_text = segment.get('text', '').strip()
            if not original_text:
                continue
            
            # Step 1: Basic cleaning
            cleaned_text = self._basic_clean(original_text)
            
            # Step 2: Remove filler words
            cleaned_text, fillers_removed = self._remove_filler_words(cleaned_text)
            
            # Step 3: Fix repetitions
            cleaned_text = self._fix_repetitions(cleaned_text)
            
            # Step 4: Handle false starts
            cleaned_text = self._fix_false_starts(cleaned_text)
            
            # Step 5: Complete sentences
            cleaned_text = self._complete_sentences(cleaned_text)
            
            # Step 6: Final cleanup
            cleaned_text = self._final_cleanup(cleaned_text)
            
            # Only keep if meaningful content remains
            if len(cleaned_text.split()) >= 3:  # At least 3 words
                confidence = self._calculate_confidence(original_text, cleaned_text)
                
                cleaned_segment = CleanedSegment(
                    original_start=segment.get('start', 0),
                    original_end=segment.get('end', 0),
                    cleaned_text=cleaned_text,
                    confidence=confidence,
                    words_removed=fillers_removed,
                    original_text=original_text
                )
                
                cleaned_segments.append(cleaned_segment)
        
        return cleaned_segments

    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove transcription artifacts
        text = re.sub(r'\[.*?\]', '', text)  # [Music], [Laughter]
        text = re.sub(r'\(.*?\)', '', text)  # (inaudible)
        
        # Fix common transcription errors
        text = re.sub(r'\buh\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bum\s+', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def _remove_filler_words(self, text: str) -> tuple:
        """Remove filler words and count removals"""
        words = text.split()
        original_count = len(words)
        
        # Remove single filler words
        filtered_words = []
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if word_clean not in self.filler_words:
                filtered_words.append(word)
        
        # Remove filler phrases
        cleaned_text = ' '.join(filtered_words)
        for filler in ['you know', 'I mean', 'kind of', 'sort of']:
            cleaned_text = re.sub(rf'\b{re.escape(filler)}\b', '', cleaned_text, flags=re.IGNORECASE)
        
        words_removed = original_count - len(cleaned_text.split())
        return cleaned_text, words_removed

    def _fix_repetitions(self, text: str) -> str:
        """Fix word repetitions"""
        for pattern in self.repetition_patterns:
            text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        
        return text

    def _fix_false_starts(self, text: str) -> str:
        """Fix false starts and corrections"""
        for pattern in self.false_start_patterns:
            # Keep the correction, remove the false start
            text = re.sub(pattern, r'\2', text, flags=re.IGNORECASE)
        
        return text

    def _complete_sentences(self, text: str) -> str:
        """Complete incomplete sentences"""
        # Remove incomplete endings
        for pattern in self.incomplete_patterns:
            text = re.sub(pattern, '', text)
        
        # Add periods to complete sentences if missing
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Remove very short fragments
        words = text.split()
        if len(words) < 3:
            return ""
        
        return text.strip()

    def _calculate_confidence(self, original: str, cleaned: str) -> float:
        """Calculate confidence in the cleaning"""
        original_words = len(original.split())
        cleaned_words = len(cleaned.split())
        
        if original_words == 0:
            return 0.0
        
        # Confidence based on retention ratio
        retention_ratio = cleaned_words / original_words
        
        # Good cleaning should retain 30-80% of words
        if 0.3 <= retention_ratio <= 0.8:
            return min(1.0, retention_ratio * 1.5)
        elif retention_ratio > 0.8:
            return 0.8  # Might not have cleaned enough
        else:
            return retention_ratio  # Cleaned too aggressively
    
    def merge_cleaned_segments(self, cleaned_segments: List[CleanedSegment], 
                             target_length: int = 150) -> List[Dict]:
        """
        Merge cleaned segments into optimal chunks for search
        """
        if not cleaned_segments:
            return []
        
        merged_chunks = []
        current_chunk = {
            'text': '',
            'start_time': cleaned_segments[0].original_start,
            'end_time': cleaned_segments[0].original_end,
            'segment_count': 0,
            'confidence': 0.0
        }
        
        for segment in cleaned_segments:
            # Check if adding this segment exceeds target length
            potential_text = current_chunk['text'] + ' ' + segment.cleaned_text
            
            if len(potential_text.split()) > target_length and current_chunk['text']:
                # Finalize current chunk
                current_chunk['confidence'] = current_chunk['confidence'] / current_chunk['segment_count']
                merged_chunks.append(current_chunk.copy())
                
                # Start new chunk
                current_chunk = {
                    'text': segment.cleaned_text,
                    'start_time': segment.original_start,
                    'end_time': segment.original_end,
                    'segment_count': 1,
                    'confidence': segment.confidence
                }
            else:
                # Add to current chunk
                if current_chunk['text']:
                    current_chunk['text'] += ' ' + segment.cleaned_text
                else:
                    current_chunk['text'] = segment.cleaned_text
                    current_chunk['start_time'] = segment.original_start
                
                current_chunk['end_time'] = segment.original_end
                current_chunk['segment_count'] += 1
                current_chunk['confidence'] += segment.confidence
        
        # Don't forget the last chunk
        if current_chunk['text']:
            current_chunk['confidence'] = current_chunk['confidence'] / current_chunk['segment_count']
            merged_chunks.append(current_chunk)
        
        return merged_chunks

    def get_cleaning_stats(self, original_segments: List[Dict], 
                          cleaned_segments: List[CleanedSegment]) -> Dict:
        """Get statistics about the cleaning process"""
        original_words = sum(len(seg.get('text', '').split()) for seg in original_segments)
        cleaned_words = sum(len(seg.cleaned_text.split()) for seg in cleaned_segments)
        
        total_removed = sum(seg.words_removed for seg in cleaned_segments)
        avg_confidence = sum(seg.confidence for seg in cleaned_segments) / len(cleaned_segments) if cleaned_segments else 0
        
        return {
            'original_segments': len(original_segments),
            'cleaned_segments': len(cleaned_segments),
            'original_words': original_words,
            'cleaned_words': cleaned_words,
            'words_removed': total_removed,
            'retention_ratio': cleaned_words / original_words if original_words > 0 else 0,
            'average_confidence': avg_confidence,
            'compression_ratio': len(cleaned_segments) / len(original_segments) if original_segments else 0
        }

# Example usage and testing
def test_cleaning():
    """Test the transcript cleaner"""
    sample_segments = [
        {
            'start': 0,
            'end': 5,
            'text': 'Um, so like, machine learning is, uh, basically when computers learn'
        },
        {
            'start': 5,
            'end': 10,
            'text': 'you know, from data and, um, they can make predictions and stuff'
        },
        {
            'start': 10,
            'end': 15,
            'text': 'Neural networks are, well, they are like brain neurons but artificial'
        }
    ]
    
    cleaner = TranscriptCleaner()
    cleaned = cleaner.clean_transcript(sample_segments)
    merged = cleaner.merge_cleaned_segments(cleaned)
    stats = cleaner.get_cleaning_stats(sample_segments, cleaned)
    
    print("=== TRANSCRIPT CLEANING TEST ===")
    print(f"Original segments: {len(sample_segments)}")
    print(f"Cleaned segments: {len(cleaned)}")
    print(f"Merged chunks: {len(merged)}")
    print(f"Word retention: {stats['retention_ratio']:.1%}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    
    print("\n=== BEFORE vs AFTER ===")
    for i, (original, clean) in enumerate(zip(sample_segments, cleaned)):
        print(f"Original {i}: {original['text']}")
        print(f"Cleaned {i}: {clean.cleaned_text}")
        print(f"Confidence: {clean.confidence:.2f}")
        print()

if __name__ == "__main__":
    test_cleaning()