"""
Enhanced KnowledgeAgent Action Graph Planning System
Combines official KnowledgeAgent methodology with our multi-modal capabilities
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Available action types in our enhanced research system"""
    START = "Start"
    SEARCH_PDF = "SearchPDF"
    SEARCH_YOUTUBE = "SearchYouTube" 
    SEARCH_VECTOR = "SearchVector"
    PROCESS_PDF = "ProcessPDF"
    PROCESS_YOUTUBE = "ProcessYouTube"
    RETRIEVE_CONTENT = "RetrieveContent"
    LOOKUP_CONTENT = "LookupContent"
    SYNTHESIZE = "Synthesize"
    FINISH = "Finish"

@dataclass
class ActionStep:
    """Represents a single step in the action path"""
    step_number: int
    action_type: ActionType
    argument: str
    observation: str = ""
    timestamp: float = 0.0
    success: bool = False
    error_message: str = ""

@dataclass
class ActionPath:
    """Complete action trajectory for a research query"""
    query: str
    steps: List[ActionStep]
    final_answer: str = ""
    total_execution_time: float = 0.0
    success: bool = False
    confidence_score: float = 0.0

class KnowledgeAgentActionGraph:
    """
    Enhanced action graph that combines KnowledgeAgent's planning with our multi-modal system
    """
    
    def __init__(self):
        # Define action transitions based on KnowledgeAgent methodology but adapted for our system
        self.action_graph = {
            ActionType.START: [
                ActionType.SEARCH_PDF, 
                ActionType.SEARCH_YOUTUBE, 
                ActionType.SEARCH_VECTOR
            ],
            ActionType.SEARCH_PDF: [
                ActionType.PROCESS_PDF,
                ActionType.SEARCH_YOUTUBE, 
                ActionType.SEARCH_VECTOR,
                ActionType.SYNTHESIZE
            ],
            ActionType.SEARCH_YOUTUBE: [
                ActionType.PROCESS_YOUTUBE,
                ActionType.SEARCH_PDF,
                ActionType.SEARCH_VECTOR, 
                ActionType.SYNTHESIZE
            ],
            ActionType.SEARCH_VECTOR: [
                ActionType.RETRIEVE_CONTENT,
                ActionType.LOOKUP_CONTENT,
                ActionType.SEARCH_PDF,
                ActionType.SEARCH_YOUTUBE,
                ActionType.SYNTHESIZE
            ],
            ActionType.PROCESS_PDF: [
                ActionType.SEARCH_YOUTUBE,
                ActionType.SEARCH_VECTOR,
                ActionType.LOOKUP_CONTENT,
                ActionType.SYNTHESIZE
            ],
            ActionType.PROCESS_YOUTUBE: [
                ActionType.SEARCH_PDF,
                ActionType.SEARCH_VECTOR, 
                ActionType.LOOKUP_CONTENT,
                ActionType.SYNTHESIZE
            ],
            ActionType.RETRIEVE_CONTENT: [
                ActionType.LOOKUP_CONTENT,
                ActionType.SEARCH_PDF,
                ActionType.SEARCH_YOUTUBE,
                ActionType.SYNTHESIZE
            ],
            ActionType.LOOKUP_CONTENT: [
                ActionType.RETRIEVE_CONTENT,
                ActionType.SEARCH_PDF,
                ActionType.SEARCH_YOUTUBE,
                ActionType.SYNTHESIZE
            ],
            ActionType.SYNTHESIZE: [
                ActionType.FINISH
            ],
            ActionType.FINISH: []
        }
        
        self.action_descriptions = {
            ActionType.SEARCH_PDF: "Search for PDF documents related to the query",
            ActionType.SEARCH_YOUTUBE: "Search for YouTube videos related to the query", 
            ActionType.SEARCH_VECTOR: "Search the vector database for relevant content",
            ActionType.PROCESS_PDF: "Process and extract information from PDF documents",
            ActionType.PROCESS_YOUTUBE: "Process and transcribe YouTube videos",
            ActionType.RETRIEVE_CONTENT: "Retrieve specific content from the knowledge base",
            ActionType.LOOKUP_CONTENT: "Look up specific keywords or phrases in retrieved content",
            ActionType.SYNTHESIZE: "Synthesize information from multiple sources into a coherent answer",
            ActionType.FINISH: "Complete the research task with final answer"
        }
        
    def get_valid_actions(self, current_action: ActionType) -> List[ActionType]:
        """Get valid next actions from current action"""
        return self.action_graph.get(current_action, [])
    
    def plan_research_path(self, query: str, available_resources: Dict[str, Any]) -> List[ActionType]:
        """
        Plan an optimal research path based on query and available resources
        Inspired by KnowledgeAgent's planning methodology
        """
        path = [ActionType.START]
        
        # Analyze query to determine initial actions
        query_lower = query.lower()
        
        # Determine starting strategy based on query and resources
        if available_resources.get('pdf_files'):
            path.append(ActionType.SEARCH_PDF)
        elif available_resources.get('youtube_urls'):
            path.append(ActionType.SEARCH_YOUTUBE)
        elif self._has_existing_knowledge(query):
            path.append(ActionType.SEARCH_VECTOR)
        else:
            # Default: start with PDF search for academic queries
            if any(keyword in query_lower for keyword in ['paper', 'research', 'study', 'author']):
                path.append(ActionType.SEARCH_PDF)
            else:
                path.append(ActionType.SEARCH_VECTOR)
        
        # Add processing steps
        if ActionType.SEARCH_PDF in path:
            path.append(ActionType.PROCESS_PDF)
        if ActionType.SEARCH_YOUTUBE in path:
            path.append(ActionType.PROCESS_YOUTUBE)
        if ActionType.SEARCH_VECTOR in path:
            path.append(ActionType.RETRIEVE_CONTENT)
            
        # Add cross-modal search if beneficial
        if len(available_resources.get('pdf_files', [])) > 0 and len(available_resources.get('youtube_urls', [])) > 0:
            if ActionType.SEARCH_PDF in path and ActionType.SEARCH_YOUTUBE not in path:
                path.append(ActionType.SEARCH_YOUTUBE)
                path.append(ActionType.PROCESS_YOUTUBE)
            elif ActionType.SEARCH_YOUTUBE in path and ActionType.SEARCH_PDF not in path:
                path.append(ActionType.SEARCH_PDF)
                path.append(ActionType.PROCESS_PDF)
        
        # Always end with synthesis and finish
        path.extend([ActionType.SYNTHESIZE, ActionType.FINISH])
        
        return path
    
    def _has_existing_knowledge(self, query: str) -> bool:
        """Check if we have existing knowledge for this query"""
        # TODO: Implement check against vector database
        return False
    
    def create_action_step(self, step_number: int, action_type: ActionType, argument: str) -> ActionStep:
        """Create a new action step"""
        return ActionStep(
            step_number=step_number,
            action_type=action_type,
            argument=argument,
            timestamp=time.time()
        )
    
    def validate_action_path(self, path: List[ActionType]) -> bool:
        """Validate that an action path follows the graph rules"""
        if not path or path[0] != ActionType.START:
            return False
            
        for i in range(len(path) - 1):
            current_action = path[i]
            next_action = path[i + 1]
            
            valid_next_actions = self.get_valid_actions(current_action)
            if next_action not in valid_next_actions:
                logger.warning(f"Invalid transition: {current_action} -> {next_action}")
                return False
                
        return True
    
    def get_action_description(self, action: ActionType) -> str:
        """Get human-readable description of an action"""
        return self.action_descriptions.get(action, f"Execute {action.value}")
    
    def export_graph_visualization(self) -> Dict[str, Any]:
        """Export graph structure for visualization"""
        nodes = []
        edges = []
        
        for action_type in ActionType:
            nodes.append({
                "id": action_type.value,
                "label": action_type.value,
                "description": self.get_action_description(action_type)
            })
        
        for source, targets in self.action_graph.items():
            for target in targets:
                edges.append({
                    "source": source.value,
                    "target": target.value
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "description": "Enhanced KnowAgent Action Graph for Multi-Modal Research"
        }

class ActionPathOptimizer:
    """
    Optimizes action paths based on success patterns
    Implements trajectory learning from KnowAgent
    """
    
    def __init__(self, storage_path: str = "action_trajectories"):
        self.storage_path = storage_path
        self.successful_paths = []
        self.failed_paths = []
        self.action_success_rates = {}
        
    def record_trajectory(self, action_path: ActionPath):
        """Record a completed action trajectory"""
        if action_path.success:
            self.successful_paths.append(action_path)
        else:
            self.failed_paths.append(action_path)
            
        # Update action success rates
        for step in action_path.steps:
            action_key = step.action_type.value
            if action_key not in self.action_success_rates:
                self.action_success_rates[action_key] = {"success": 0, "total": 0}
                
            self.action_success_rates[action_key]["total"] += 1
            if step.success:
                self.action_success_rates[action_key]["success"] += 1
    
    def get_action_success_rate(self, action: ActionType) -> float:
        """Get success rate for a specific action"""
        key = action.value
        if key not in self.action_success_rates:
            return 0.5  # Default neutral probability
            
        stats = self.action_success_rates[key]
        if stats["total"] == 0:
            return 0.5
            
        return stats["success"] / stats["total"]
    
    def suggest_path_improvements(self, failed_path: ActionPath) -> List[ActionType]:
        """Suggest improvements for a failed path"""
        # Analyze failure points and suggest alternatives
        improvements = []
        
        for step in failed_path.steps:
            if not step.success:
                # Find alternative actions with higher success rates
                current_success_rate = self.get_action_success_rate(step.action_type)
                
                # Suggest actions with better success rates
                for action_type in ActionType:
                    if (action_type != step.action_type and 
                        self.get_action_success_rate(action_type) > current_success_rate):
                        improvements.append(action_type)
                        
        return improvements
    
    def save_trajectories(self):
        """Save trajectory data to disk"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Save successful paths
        with open(f"{self.storage_path}/successful_paths.json", "w") as f:
            json.dump([self._serialize_path(path) for path in self.successful_paths], f, indent=2)
            
        # Save success rates
        with open(f"{self.storage_path}/action_success_rates.json", "w") as f:
            json.dump(self.action_success_rates, f, indent=2)
    
    def _serialize_path(self, path: ActionPath) -> Dict[str, Any]:
        """Serialize action path for storage"""
        return {
            "query": path.query,
            "steps": [
                {
                    "step_number": step.step_number,
                    "action_type": step.action_type.value,
                    "argument": step.argument,
                    "observation": step.observation,
                    "success": step.success,
                    "timestamp": step.timestamp
                }
                for step in path.steps
            ],
            "final_answer": path.final_answer,
            "total_execution_time": path.total_execution_time,
            "success": path.success,
            "confidence_score": path.confidence_score
        }

if __name__ == "__main__":
    # Test the action graph
    graph = KnowAgentActionGraph()
    
    # Test planning
    test_query = "Who authored this research paper?"
    test_resources = {
        "pdf_files": ["research_paper.pdf"],
        "youtube_urls": []
    }
    
    planned_path = graph.plan_research_path(test_query, test_resources)
    print("Planned Action Path:")
    for i, action in enumerate(planned_path):
        print(f"{i+1}. {action.value}: {graph.get_action_description(action)}")
    
    # Test validation
    is_valid = graph.validate_action_path(planned_path)
    print(f"\nPath validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Export graph visualization
    graph_data = graph.export_graph_visualization()
    print(f"\nGraph exported with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")