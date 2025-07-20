"""
Base Agent Class for KnowAgent System

This defines the interface that all specialized agents must implement.
Each agent is responsible for executing specific types of actions
from our knowledge base.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio
import logging
logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of executing an action"""
    success: bool
    output: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """
    Base class for all KnowAgent execution agents.
    
    Each specialized agent (PDFAgent, VideoAgent, etc.) inherits from this
    and implements the specific actions it can perform.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.supported_actions = self._get_supported_actions()
        logger.info(f"Initialized {self.name} agent with actions: {self.supported_actions}")
    
    @abstractmethod
    def _get_supported_actions(self) -> List[str]:
        """Return list of actions this agent can execute"""
        pass
    
    @abstractmethod
    async def execute_action(self, action: str, target: str, **kwargs) -> ExecutionResult:
        """
        Execute a specific action.
        
        Args:
            action: The action to execute (must be in supported_actions)
            target: The target to operate on (file path, URL, etc.)
            **kwargs: Additional parameters for the action
        
        Returns:
            ExecutionResult with success status and output
        """
        pass
    
    def can_execute(self, action: str) -> bool:
        """Check if this agent can execute a specific action"""
        return action in self.supported_actions
    
    async def validate_prerequisites(self, action: str, target: str, **kwargs) -> bool:
        """
        Validate that prerequisites for an action are met.
        
        This is called before executing an action to ensure
        it will succeed.
        """
        # Default implementation - override in subclasses
        return True
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of the agent"""
        return {
            "name": self.name,
            "supported_actions": self.supported_actions,
            "is_ready": True
        }

class AgentCoordinator:
    """
    Coordinates multiple agents to execute a research plan.
    
    This is the "conductor" that orchestrates all the different
    specialized agents to work together.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_history: List[Dict[str, Any]] = []
        logger.info("Initialized AgentCoordinator")
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent with the coordinator"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def find_agent_for_action(self, action: str) -> Optional[BaseAgent]:
        """Find which agent can execute a specific action"""
        for agent in self.agents.values():
            if agent.can_execute(action):
                return agent
        return None
    
    async def execute_plan_step(self, step, previous_results: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a single step from a research plan.
        
        Args:
            step: PlanStep object
            previous_results: Results from previous steps
        
        Returns:
            ExecutionResult
        """
        # Find the right agent for this action
        agent = self.find_agent_for_action(step.action)
        if not agent:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=f"No agent found for action: {step.action}"
            )
        
        logger.info(f"Executing {step.action} with {agent.name}")
        
        try:
            # Validate prerequisites
            prereqs_met = await agent.validate_prerequisites(
                step.action, 
                step.target,
                previous_results=previous_results
            )
            
            if not prereqs_met:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error_message=f"Prerequisites not met for {step.action}"
                )
            
            # Execute the action
            result = await agent.execute_action(
                step.action, 
                step.target,
                previous_results=previous_results
            )
            
            # Log the execution
            self.execution_history.append({
                "step": step.action,
                "agent": agent.name,
                "success": result.success,
                "execution_time": result.execution_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing {step.action}: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error_message=str(e)
            )
    
    async def execute_full_plan(self, plan) -> Dict[str, Any]:
        """
        Execute a complete research plan.
        
        This orchestrates the execution of all steps in sequence,
        passing results between steps as needed.
        """
        logger.info(f"Starting execution of plan with {len(plan.steps)} steps")
        
        results = {}
        step_results = []
        
        for i, step in enumerate(plan.steps):
            logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.action}")
            
            # Execute the step
            result = await self.execute_plan_step(step, results)
            
            # Store the result
            step_key = f"step_{i+1}_{step.action}"
            results[step_key] = result.output
            step_results.append({
                "step_number": i + 1,
                "action": step.action,
                "success": result.success,
                "output": result.output,
                "error": result.error_message,
                "execution_time": result.execution_time
            })
            
            # If a critical step fails, decide whether to continue
            if not result.success:
                logger.warning(f"Step {i+1} failed: {result.error_message}")
                # For now, continue with next steps
                # In production, you might want more sophisticated error handling
        
        return {
            "plan_query": plan.query,
            "total_steps": len(plan.steps),
            "step_results": step_results,
            "final_results": results,
            "execution_history": self.execution_history
        }
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get status of the coordinator and all agents"""
        return {
            "registered_agents": list(self.agents.keys()),
            "agent_statuses": {name: agent.get_agent_status() 
                             for name, agent in self.agents.items()},
            "execution_history_count": len(self.execution_history)
        }

# Example usage
if __name__ == "__main__":
    # This would be tested with actual agent implementations
    coordinator = AgentCoordinator()
    print("AgentCoordinator initialized successfully!")
    print(f"Status: {coordinator.get_coordinator_status()}")