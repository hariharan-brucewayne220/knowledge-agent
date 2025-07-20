"""
KnowAgent Research Planner

This is where the MAGIC happens. The planner takes a user's research question
and creates a step-by-step plan using ONLY actions that are validated by
the knowledge base.

The key insight: Instead of letting the LLM plan freely (and hallucinate
bad actions), we constrain it to choose from a curated set of safe,
effective actions.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge.action_knowledge import ActionKnowledgeBase, TaskType, ActionInfo

@dataclass
class PlanStep:
    """A single step in a research plan"""
    action: str
    target: str  # What this action operates on
    description: str
    prerequisites_met: List[str]
    expected_output: str
    estimated_time: str

@dataclass
class ResearchPlan:
    """A complete research plan"""
    query: str
    task_type: TaskType
    steps: List[PlanStep]
    total_estimated_time: str
    confidence_score: float

class QueryClassifier:
    """
    Classifies user queries into task types.
    
    This is important because different types of research questions
    require different approaches and action sequences.
    """
    
    def classify_query(self, query: str) -> TaskType:
        """
        Classify a research query into a task type.
        
        In a full implementation, this would use an LLM or trained classifier.
        For our demo, we'll use keyword matching.
        """
        query_lower = query.lower()
        
        # Multi-modal queries (PDF + video)
        if any(keyword in query_lower for keyword in 
               ['compare', 'contrast', 'difference', 'similarity', 'both']):
            return TaskType.CROSS_MODAL_RESEARCH
        
        # Video-specific queries
        elif any(keyword in query_lower for keyword in 
                 ['video', 'youtube', 'watch', 'transcript', 'spoken']):
            return TaskType.VIDEO_PROCESSING
        
        # Document-specific queries
        elif any(keyword in query_lower for keyword in 
                 ['pdf', 'document', 'paper', 'article', 'text']):
            return TaskType.DOCUMENT_ANALYSIS
        
        # Synthesis queries
        elif any(keyword in query_lower for keyword in 
                 ['summarize', 'synthesis', 'overview', 'insights']):
            return TaskType.SYNTHESIS
        
        # Default to cross-modal research
        else:
            return TaskType.CROSS_MODAL_RESEARCH

class ResearchPlanner:
    """
    The core planning agent that creates intelligent, validated research plans.
    
    This is what makes KnowAgent special - instead of random action sequences,
    we create logical, safe plans based on expert knowledge.
    """
    
    def __init__(self):
        self.knowledge_base = ActionKnowledgeBase()
        self.classifier = QueryClassifier()
    
    def create_plan(self, query: str, available_sources: Dict[str, List[str]]) -> ResearchPlan:
        """
        Create a research plan for a given query.
        
        Args:
            query: The user's research question
            available_sources: Dict like {"pdfs": ["file1.pdf"], "videos": ["url1"]}
        
        Returns:
            A validated research plan
        """
        # Step 1: Classify the query
        task_type = self.classifier.classify_query(query)
        print(f"Query classified as: {task_type.value}")
        
        # Step 2: Get valid actions for this task type
        valid_actions = self.knowledge_base.get_valid_actions(task_type)
        print(f"Available actions: {[a.name for a in valid_actions]}")
        
        # Step 3: Create the plan based on available sources
        steps = self._plan_for_task_type(task_type, query, available_sources, valid_actions)
        
        # Step 4: Validate and optimize the plan
        validated_steps = self._validate_plan_steps(steps)
        
        # Step 5: Calculate confidence and timing
        confidence = self._calculate_confidence(validated_steps, available_sources)
        total_time = self._estimate_total_time(validated_steps)
        
        return ResearchPlan(
            query=query,
            task_type=task_type,
            steps=validated_steps,
            total_estimated_time=total_time,
            confidence_score=confidence
        )
    
    def _plan_for_task_type(self, task_type: TaskType, query: str, 
                           sources: Dict[str, List[str]], 
                           valid_actions: List[ActionInfo]) -> List[PlanStep]:
        """
        Create task-specific plans.
        
        This is where we encode the LOGIC of how to approach different
        types of research tasks.
        """
        
        if task_type == TaskType.DOCUMENT_ANALYSIS:
            return self._plan_document_analysis(query, sources.get("pdfs", []), valid_actions)
        
        elif task_type == TaskType.VIDEO_PROCESSING:
            return self._plan_video_processing(query, sources.get("videos", []), valid_actions)
        
        elif task_type == TaskType.CROSS_MODAL_RESEARCH:
            return self._plan_cross_modal_research(query, sources, valid_actions)
        
        elif task_type == TaskType.SYNTHESIS:
            return self._plan_synthesis(query, sources, valid_actions)
        
        else:
            # Fallback to cross-modal
            return self._plan_cross_modal_research(query, sources, valid_actions)
    
    def _plan_document_analysis(self, query: str, pdf_files: List[str], 
                               valid_actions: List[ActionInfo]) -> List[PlanStep]:
        """Plan for analyzing PDF documents."""
        steps = []
        
        # Step 1: Process each PDF
        for pdf_file in pdf_files:
            steps.append(PlanStep(
                action="extract_text",
                target=pdf_file,
                description=f"Extract text content from {pdf_file}",
                prerequisites_met=["valid_pdf_file"],
                expected_output="raw_text",
                estimated_time="30 seconds"
            ))
            
            steps.append(PlanStep(
                action="chunk_document",
                target=f"{pdf_file}_text",
                description=f"Split {pdf_file} into semantic chunks",
                prerequisites_met=["extracted_text"],
                expected_output="text_chunks",
                estimated_time="10 seconds"
            ))
            
            steps.append(PlanStep(
                action="create_embeddings",
                target=f"{pdf_file}_chunks",
                description=f"Generate embeddings for {pdf_file} chunks",
                prerequisites_met=["text_chunks"],
                expected_output="vector_embeddings",
                estimated_time="2 minutes"
            ))
        
        # Step 2: Search and synthesize
        steps.append(PlanStep(
            action="semantic_search",
            target="all_embeddings",
            description=f"Search for content relevant to: {query}",
            prerequisites_met=["embeddings_ready"],
            expected_output="relevant_passages",
            estimated_time="5 seconds"
        ))
        
        return steps
    
    def _plan_video_processing(self, query: str, video_urls: List[str], 
                              valid_actions: List[ActionInfo]) -> List[PlanStep]:
        """Plan for processing YouTube videos."""
        steps = []
        
        for video_url in video_urls:
            steps.append(PlanStep(
                action="download_audio",
                target=video_url,
                description=f"Extract audio from {video_url}",
                prerequisites_met=["valid_youtube_url"],
                expected_output="audio_file",
                estimated_time="1 minute"
            ))
            
            steps.append(PlanStep(
                action="transcribe_audio",
                target=video_url,
                description=f"Transcribe audio to text",
                prerequisites_met=["audio_file"],
                expected_output="transcript",
                estimated_time="3 minutes"
            ))
            
            steps.append(PlanStep(
                action="extract_timestamps",
                target=video_url,
                description="Create timestamped segments",
                prerequisites_met=["transcript"],
                expected_output="timestamped_segments",
                estimated_time="30 seconds"
            ))
        
        return steps
    
    def _plan_cross_modal_research(self, query: str, sources: Dict[str, List[str]], 
                                  valid_actions: List[ActionInfo]) -> List[PlanStep]:
        """Plan for research across both PDFs and videos."""
        steps = []
        
        # Process PDFs first
        if "pdfs" in sources:
            steps.extend(self._plan_document_analysis(query, sources["pdfs"], valid_actions))
        
        # Then process videos
        if "videos" in sources:
            steps.extend(self._plan_video_processing(query, sources["videos"], valid_actions))
        
        # Finally, cross-modal analysis
        if "pdfs" in sources and "videos" in sources:
            steps.append(PlanStep(
                action="cross_reference",
                target="all_content",
                description="Find connections between PDF and video content",
                prerequisites_met=["processed_content"],
                expected_output="cross_references",
                estimated_time="1 minute"
            ))
            
            steps.append(PlanStep(
                action="temporal_alignment",
                target="timestamped_content",
                description="Align concepts across time and sources",
                prerequisites_met=["timestamped_content"],
                expected_output="temporal_relationships",
                estimated_time="2 minutes"
            ))
        
        # Synthesis step
        steps.append(PlanStep(
            action="generate_summary",
            target="all_processed_content",
            description=f"Generate comprehensive answer to: {query}",
            prerequisites_met=["processed_content"],
            expected_output="synthesized_answer",
            estimated_time="1 minute"
        ))
        
        return steps
    
    def _plan_synthesis(self, query: str, sources: Dict[str, List[str]], 
                       valid_actions: List[ActionInfo]) -> List[PlanStep]:
        """Plan for synthesizing information from multiple sources."""
        # Start with cross-modal processing
        steps = self._plan_cross_modal_research(query, sources, valid_actions)
        
        # Add synthesis-specific steps
        steps.append(PlanStep(
            action="create_comparison",
            target="processed_sources",
            description="Compare and contrast information across sources",
            prerequisites_met=["multiple_sources"],
            expected_output="comparison_analysis",
            estimated_time="2 minutes"
        ))
        
        steps.append(PlanStep(
            action="generate_insights",
            target="comparison_analysis",
            description="Extract novel insights from combined analysis",
            prerequisites_met=["synthesized_content"],
            expected_output="novel_insights",
            estimated_time="3 minutes"
        ))
        
        return steps
    
    def _validate_plan_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """
        Validate that all plan steps are safe and logical.
        
        This is a KEY safety mechanism - we double-check that every
        action in the plan is actually valid.
        """
        validated_steps = []
        
        for step in steps:
            # Check if action exists in knowledge base
            action_info = None
            for task_type in TaskType:
                action_info = self.knowledge_base.get_action_info(task_type, step.action)
                if action_info:
                    break
            
            if action_info and action_info.risk_level != "forbidden":
                validated_steps.append(step)
                print(f"Validated step: {step.action}")
            else:
                print(f"Rejected step: {step.action} (invalid or forbidden)")
        
        return validated_steps
    
    def _calculate_confidence(self, steps: List[PlanStep], sources: Dict[str, List[str]]) -> float:
        """Calculate confidence score for the plan."""
        if not steps:
            return 0.0
        
        # Base confidence on:
        # 1. Number of sources available
        # 2. Completeness of the plan
        # 3. Quality of action sequence
        
        source_count = sum(len(source_list) for source_list in sources.values())
        source_score = min(1.0, source_count / 3.0)  # Optimal with 3+ sources
        
        plan_completeness = len(steps) / 8.0  # Expect ~8 steps for complete plan
        completeness_score = min(1.0, plan_completeness)
        
        # Average the scores
        confidence = (source_score + completeness_score) / 2.0
        return round(confidence, 2)
    
    def _estimate_total_time(self, steps: List[PlanStep]) -> str:
        """Estimate total execution time for the plan."""
        # Simple time estimation based on step count and types
        total_seconds = len(steps) * 30  # Average 30 seconds per step
        
        if total_seconds < 60:
            return f"{total_seconds} seconds"
        elif total_seconds < 3600:
            return f"{total_seconds // 60} minutes"
        else:
            return f"{total_seconds // 3600} hours"

# Example usage
if __name__ == "__main__":
    planner = ResearchPlanner()
    
    # Test query
    test_query = "What are the key differences between machine learning approaches mentioned in my documents and the practical implementations shown in these videos?"
    
    # Test sources
    test_sources = {
        "pdfs": ["ml_theory_paper.pdf", "deep_learning_survey.pdf"],
        "videos": ["https://youtube.com/watch?v=practical_ml", "https://youtube.com/watch?v=ml_implementation"]
    }
    
    # Create plan
    plan = planner.create_plan(test_query, test_sources)
    
    # Display the plan
    print(f"\nResearch Plan for: {plan.query}")
    print(f"Task Type: {plan.task_type.value}")
    print(f"Estimated Time: {plan.total_estimated_time}")
    print(f"Confidence: {plan.confidence_score}")
    print(f"\nExecution Steps:")
    
    for i, step in enumerate(plan.steps, 1):
        print(f"{i}. {step.action}: {step.description}")
        print(f"   Time: {step.estimated_time}")
        print(f"   Output: {step.expected_output}")
        print()