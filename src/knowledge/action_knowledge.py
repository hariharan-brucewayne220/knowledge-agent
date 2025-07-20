"""
KnowAgent Action Knowledge Base

This is the CORE of KnowAgent's intelligence. Instead of letting the LLM
hallucinate random actions, we provide a structured knowledge base of:
1. What actions are valid for each task type
2. What prerequisites each action needs
3. What outcomes each action produces
4. What actions are forbidden/dangerous

This prevents the agent from doing stupid things like:
- Deleting files when asked to research
- Running system commands for document analysis
- Making API calls without proper authentication
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    """Different types of research tasks"""
    DOCUMENT_ANALYSIS = "document_analysis"
    VIDEO_PROCESSING = "video_processing"
    CROSS_MODAL_RESEARCH = "cross_modal_research"
    SYNTHESIS = "synthesis"
    FACT_CHECKING = "fact_checking"

@dataclass
class ActionInfo:
    """Information about a specific action"""
    name: str
    description: str
    prerequisites: List[str]
    expected_outcomes: List[str]
    risk_level: str  # "safe", "caution", "forbidden"
    execution_time: str  # "fast", "medium", "slow"

class ActionKnowledgeBase:
    """
    The brain of our KnowAgent system.
    
    This class stores all the knowledge about what actions are valid,
    safe, and effective for different types of research tasks.
    """
    
    def __init__(self):
        self.knowledge = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[TaskType, Dict[str, ActionInfo]]:
        """
        Build the complete action knowledge base.
        
        This is where we encode HUMAN EXPERTISE about research workflows.
        Think of this as a senior researcher's knowledge about how to
        approach different types of research tasks.
        """
        
        return {
            TaskType.DOCUMENT_ANALYSIS: {
                "extract_text": ActionInfo(
                    name="extract_text",
                    description="Extract text content from PDF documents",
                    prerequisites=["valid_pdf_file", "read_permissions"],
                    expected_outcomes=["raw_text", "document_structure"],
                    risk_level="safe",
                    execution_time="fast"
                ),
                "chunk_document": ActionInfo(
                    name="chunk_document",
                    description="Split document into semantic chunks",
                    prerequisites=["extracted_text"],
                    expected_outcomes=["text_chunks", "chunk_metadata"],
                    risk_level="safe",
                    execution_time="fast"
                ),
                "create_embeddings": ActionInfo(
                    name="create_embeddings",
                    description="Generate vector embeddings for text chunks",
                    prerequisites=["text_chunks", "embedding_model"],
                    expected_outcomes=["vector_embeddings", "embedding_index"],
                    risk_level="safe",
                    execution_time="medium"
                ),
                "extract_citations": ActionInfo(
                    name="extract_citations",
                    description="Find and parse citations in academic documents",
                    prerequisites=["extracted_text", "academic_document"],
                    expected_outcomes=["citation_list", "reference_metadata"],
                    risk_level="safe",
                    execution_time="medium"
                ),
                # FORBIDDEN ACTIONS - things agents might hallucinate
                "delete_file": ActionInfo(
                    name="delete_file",
                    description="Delete files from filesystem",
                    prerequisites=[],
                    expected_outcomes=["data_loss"],
                    risk_level="forbidden",
                    execution_time="fast"
                ),
            },
            
            TaskType.VIDEO_PROCESSING: {
                "download_audio": ActionInfo(
                    name="download_audio",
                    description="Extract audio from YouTube video",
                    prerequisites=["valid_youtube_url", "network_access"],
                    expected_outcomes=["audio_file", "video_metadata"],
                    risk_level="safe",
                    execution_time="slow"
                ),
                "transcribe_audio": ActionInfo(
                    name="transcribe_audio",
                    description="Convert audio to text using Whisper",
                    prerequisites=["audio_file", "whisper_model"],
                    expected_outcomes=["transcript", "timestamps"],
                    risk_level="safe",
                    execution_time="slow"
                ),
                "extract_timestamps": ActionInfo(
                    name="extract_timestamps",
                    description="Create timestamped segments from transcript",
                    prerequisites=["transcript"],
                    expected_outcomes=["timestamped_segments", "topic_boundaries"],
                    risk_level="safe",
                    execution_time="fast"
                ),
                "identify_speakers": ActionInfo(
                    name="identify_speakers",
                    description="Perform speaker diarization on audio",
                    prerequisites=["audio_file", "speaker_model"],
                    expected_outcomes=["speaker_segments", "speaker_count"],
                    risk_level="caution",
                    execution_time="slow"
                ),
            },
            
            TaskType.CROSS_MODAL_RESEARCH: {
                "semantic_search": ActionInfo(
                    name="semantic_search",
                    description="Search across PDF and video content semantically",
                    prerequisites=["embeddings_ready", "search_query"],
                    expected_outcomes=["relevant_passages", "relevance_scores"],
                    risk_level="safe",
                    execution_time="fast"
                ),
                "temporal_alignment": ActionInfo(
                    name="temporal_alignment",
                    description="Align concepts across time in different sources",
                    prerequisites=["timestamped_content", "concept_extraction"],
                    expected_outcomes=["temporal_relationships", "concept_timeline"],
                    risk_level="safe",
                    execution_time="medium"
                ),
                "cross_reference": ActionInfo(
                    name="cross_reference",
                    description="Find references between PDF citations and video content",
                    prerequisites=["citation_list", "video_transcript"],
                    expected_outcomes=["cross_references", "source_connections"],
                    risk_level="safe",
                    execution_time="medium"
                ),
            },
            
            TaskType.SYNTHESIS: {
                "generate_summary": ActionInfo(
                    name="generate_summary",
                    description="Create comprehensive summary from multiple sources",
                    prerequisites=["processed_content", "user_query"],
                    expected_outcomes=["synthesized_summary", "source_attributions"],
                    risk_level="safe",
                    execution_time="medium"
                ),
                "create_comparison": ActionInfo(
                    name="create_comparison",
                    description="Compare and contrast information across sources",
                    prerequisites=["multiple_sources", "comparison_criteria"],
                    expected_outcomes=["comparison_table", "key_differences"],
                    risk_level="safe",
                    execution_time="medium"
                ),
                "generate_insights": ActionInfo(
                    name="generate_insights",
                    description="Extract novel insights from combined sources",
                    prerequisites=["synthesized_content", "domain_knowledge"],
                    expected_outcomes=["novel_insights", "insight_confidence"],
                    risk_level="caution",
                    execution_time="slow"
                ),
            }
        }
    
    def get_valid_actions(self, task_type: TaskType) -> List[ActionInfo]:
        """
        Get all valid (safe + caution) actions for a task type.
        
        This is what the planning agent uses to constrain its action space.
        """
        if task_type not in self.knowledge:
            return []
        
        actions = self.knowledge[task_type]
        return [action for action in actions.values() 
                if action.risk_level in ["safe", "caution"]]
    
    def is_action_valid(self, task_type: TaskType, action_name: str) -> bool:
        """
        Check if a specific action is valid for a task type.
        
        This is used during plan validation to catch hallucinated actions.
        """
        if task_type not in self.knowledge:
            return False
        
        if action_name not in self.knowledge[task_type]:
            return False
        
        action = self.knowledge[task_type][action_name]
        return action.risk_level != "forbidden"
    
    def get_action_info(self, task_type: TaskType, action_name: str) -> Optional[ActionInfo]:
        """Get detailed information about a specific action."""
        if task_type not in self.knowledge:
            return None
        return self.knowledge[task_type].get(action_name)
    
    def get_prerequisites(self, task_type: TaskType, action_name: str) -> List[str]:
        """
        Get prerequisites for an action.
        
        This helps the planner understand what needs to be done before
        this action can be executed.
        """
        action_info = self.get_action_info(task_type, action_name)
        if action_info:
            return action_info.prerequisites
        return []
    
    def get_forbidden_actions(self, task_type: TaskType) -> List[str]:
        """
        Get list of forbidden actions for a task type.
        
        This is used to actively prevent dangerous hallucinations.
        """
        if task_type not in self.knowledge:
            return []
        
        actions = self.knowledge[task_type]
        return [name for name, action in actions.items() 
                if action.risk_level == "forbidden"]

# Example usage for testing
if __name__ == "__main__":
    kb = ActionKnowledgeBase()
    
    # Test: Get valid actions for document analysis
    valid_actions = kb.get_valid_actions(TaskType.DOCUMENT_ANALYSIS)
    print("Valid actions for document analysis:")
    for action in valid_actions:
        print(f"  - {action.name}: {action.description}")
    
    # Test: Check if a dangerous action is forbidden
    is_delete_valid = kb.is_action_valid(TaskType.DOCUMENT_ANALYSIS, "delete_file")
    print(f"\nIs 'delete_file' valid for document analysis? {is_delete_valid}")
    
    # Test: Get prerequisites for an action
    prereqs = kb.get_prerequisites(TaskType.DOCUMENT_ANALYSIS, "create_embeddings")
    print(f"\nPrerequisites for 'create_embeddings': {prereqs}")