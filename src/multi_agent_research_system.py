"""
Multi-Agent Research System
A modern approach to autonomous research using specialized agents that coordinate
to conduct comprehensive, high-quality research across multiple domains.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

import openai
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages agents can send to each other"""
    RESEARCH_REQUEST = "research_request"
    RESEARCH_RESULT = "research_result"
    COORDINATION = "coordination"
    FACT_CHECK = "fact_check"
    SYNTHESIS = "synthesis"
    STATUS_UPDATE = "status_update"
    QUESTION = "question"


class AgentStatus(Enum):
    """Current status of an agent"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 5=low
    requires_response: bool = True
    correlation_id: Optional[str] = None


@dataclass
class ResearchTask:
    """Represents a research task for agents"""
    id: str
    description: str
    domain: str
    priority: int
    assigned_agent: Optional[str] = None
    status: str = "pending"
    results: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class BaseAgent(ABC):
    """Base class for all research agents"""
    
    def __init__(self, agent_id: str, name: str, openai_api_key: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.IDLE
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Communication
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        self.message_handlers = {}
        
        # Task management
        self.current_task: Optional[ResearchTask] = None
        self.completed_tasks: List[ResearchTask] = []
        
        # Agent capabilities and specialization
        self.capabilities: Set[str] = set()
        self.specialization: str = "general"
        
        # Performance metrics
        self.tasks_completed = 0
        self.success_rate = 0.0
        self.average_response_time = 0.0
        
        # Initialize message handlers
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """Set up handlers for different message types"""
        self.message_handlers = {
            MessageType.RESEARCH_REQUEST: self._handle_research_request,
            MessageType.COORDINATION: self._handle_coordination,
            MessageType.FACT_CHECK: self._handle_fact_check,
            MessageType.QUESTION: self._handle_question,
            MessageType.STATUS_UPDATE: self._handle_status_update,
        }
    
    async def start(self):
        """Start the agent's main processing loop"""
        logger.info(f"Starting agent {self.name}")
        
        # Start message processing
        message_task = asyncio.create_task(self._process_messages())
        
        # Start main work loop
        work_task = asyncio.create_task(self._work_loop())
        
        # Wait for both tasks
        await asyncio.gather(message_task, work_task)
    
    async def _process_messages(self):
        """Process incoming messages"""
        while True:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                
                # Handle message based on type
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    await handler(message)
                else:
                    logger.warning(f"No handler for message type {message.message_type}")
                
                self.inbox.task_done()
                
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                logger.error(f"Error processing message in {self.name}: {e}")
    
    async def _work_loop(self):
        """Main work loop for the agent"""
        while True:
            try:
                if self.status == AgentStatus.IDLE:
                    # Check for work to do
                    await self._check_for_work()
                
                elif self.status == AgentStatus.WORKING:
                    # Continue current work
                    await self._continue_work()
                
                # Brief pause to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in work loop for {self.name}: {e}")
                self.status = AgentStatus.ERROR
                await asyncio.sleep(1.0)
    
    async def send_message(self, message: AgentMessage):
        """Send a message to another agent"""
        await self.outbox.put(message)
    
    async def _check_for_work(self):
        """Check if there's work to be done (implemented by subclasses)"""
        await asyncio.sleep(1.0)  # Default implementation just waits
    
    async def _continue_work(self):
        """Continue current work (implemented by subclasses)"""
        await asyncio.sleep(1.0)  # Default implementation just waits
    
    # Message handlers (to be implemented by subclasses)
    async def _handle_research_request(self, message: AgentMessage):
        """Handle research request"""
        pass
    
    async def _handle_coordination(self, message: AgentMessage):
        """Handle coordination message"""
        pass
    
    async def _handle_fact_check(self, message: AgentMessage):
        """Handle fact-checking request"""
        pass
    
    async def _handle_question(self, message: AgentMessage):
        """Handle question from another agent"""
        pass
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle status update"""
        pass
    
    @abstractmethod
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process a research task (must be implemented by subclasses)"""
        pass
    
    async def _llm_request(self, messages: List[Dict], model: str = "gpt-3.5-turbo", max_tokens: int = 500) -> str:
        """Make a request to the LLM"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM request error in {self.name}: {e}")
            return ""


class ResearchCoordinator(BaseAgent):
    """Coordinates research activities across specialized agents"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("coordinator", "Research Coordinator", openai_api_key)
        self.specialization = "coordination"
        self.capabilities.update(["task_management", "agent_coordination", "research_planning"])
        
        # Research session management
        self.current_session_id: Optional[str] = None
        self.research_goal: Optional[str] = None
        self.available_agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.completed_tasks: Dict[str, ResearchTask] = {}
        
        # Research strategy
        self.research_plan: Optional[Dict] = None
        self.priority_queue: List[ResearchTask] = []
        
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent with the coordinator"""
        self.available_agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.specialization})")
    
    async def start_research_session(self, goal: str) -> str:
        """Start a new research session"""
        self.current_session_id = str(uuid.uuid4())[:8]
        self.research_goal = goal
        
        logger.info(f"Starting research session {self.current_session_id} for goal: {goal}")
        
        # Create research plan
        await self._create_research_plan()
        
        # Generate initial tasks
        await self._generate_initial_tasks()
        
        # Start task assignment loop
        asyncio.create_task(self._task_assignment_loop())
        
        return self.current_session_id
    
    async def _create_research_plan(self):
        """Create a comprehensive research plan using LLM"""
        planning_prompt = f"""
        Create a comprehensive research plan for the goal: "{self.research_goal}"
        
        Available specialized agents:
        {', '.join([f"{agent.name} ({agent.specialization})" for agent in self.available_agents.values()])}
        
        Generate a JSON research plan with:
        1. sub_goals: 3-5 specific objectives
        2. research_domains: domains to investigate (academic, technical, business, social, etc.)
        3. research_questions: 5-7 key questions to answer
        4. agent_assignments: which types of agents should handle which domains
        5. success_criteria: how to measure completion
        6. estimated_timeline: rough time estimates
        
        Return only valid JSON.
        """
        
        messages = [
            {"role": "system", "content": "You are a research planning expert. Return only valid JSON."},
            {"role": "user", "content": planning_prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=800)
            self.research_plan = json.loads(response)
            logger.info("Research plan created successfully")
        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            # Fallback plan
            self.research_plan = {
                "sub_goals": [f"Understand {self.research_goal}", f"Analyze applications of {self.research_goal}"],
                "research_domains": ["academic", "technical", "business"],
                "research_questions": [f"What is {self.research_goal}?", f"How is {self.research_goal} used?"],
                "agent_assignments": {"academic": ["academic_specialist"], "technical": ["technical_specialist"]},
                "success_criteria": ["Basic understanding achieved"],
                "estimated_timeline": "30 minutes"
            }
    
    async def _generate_initial_tasks(self):
        """Generate initial research tasks based on the plan"""
        if not self.research_plan:
            return
        
        task_id = 0
        for domain in self.research_plan.get("research_domains", []):
            for question in self.research_plan.get("research_questions", []):
                if any(keyword in question.lower() for keyword in domain.split()):
                    task = ResearchTask(
                        id=f"{self.current_session_id}_task_{task_id}",
                        description=f"Research {domain} aspects of: {question}",
                        domain=domain,
                        priority=1,
                        metadata={"session_id": self.current_session_id, "domain": domain}
                    )
                    self.priority_queue.append(task)
                    task_id += 1
        
        # Sort by priority
        self.priority_queue.sort(key=lambda x: x.priority)
        logger.info(f"Generated {len(self.priority_queue)} initial research tasks")
    
    async def _task_assignment_loop(self):
        """Continuously assign tasks to available agents"""
        while True:
            try:
                # Find available agents
                available_agents = [
                    agent for agent in self.available_agents.values()
                    if agent.status == AgentStatus.IDLE
                ]
                
                # Assign tasks to available agents
                for agent in available_agents:
                    if self.priority_queue:
                        # Find suitable task for this agent
                        suitable_task = self._find_suitable_task(agent)
                        if suitable_task:
                            await self._assign_task(agent, suitable_task)
                
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in task assignment loop: {e}")
                await asyncio.sleep(5.0)
    
    def _find_suitable_task(self, agent: BaseAgent) -> Optional[ResearchTask]:
        """Find a suitable task for the given agent"""
        for task in self.priority_queue:
            # Check if agent specialization matches task domain
            if (agent.specialization == task.domain or 
                agent.specialization == "general" or
                task.domain in agent.capabilities):
                self.priority_queue.remove(task)
                return task
        return None
    
    async def _assign_task(self, agent: BaseAgent, task: ResearchTask):
        """Assign a task to an agent"""
        task.assigned_agent = agent.agent_id
        task.status = "assigned"
        self.active_tasks[task.id] = task
        
        # Send research request to agent
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=agent.agent_id,
            message_type=MessageType.RESEARCH_REQUEST,
            content={"task": task.__dict__},
            timestamp=datetime.now(),
            priority=task.priority
        )
        
        await agent.inbox.put(message)
        logger.info(f"Assigned task {task.id} to {agent.name}")
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Coordinator doesn't process tasks directly, it delegates them"""
        return {"status": "delegated", "message": "Task delegated to specialized agent"}
    
    async def _handle_research_result(self, message: AgentMessage):
        """Handle research results from agents"""
        result = message.content
        task_id = result.get("task_id")
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.results.append(result)
            task.status = "completed"
            task.completed_at = datetime.now()
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            logger.info(f"Received results for task {task_id}")
            
            # Check if we need follow-up tasks
            await self._check_for_followup_tasks(task, result)
    
    async def _check_for_followup_tasks(self, completed_task: ResearchTask, result: Dict[str, Any]):
        """Check if completed task results suggest new research directions"""
        # Use LLM to identify potential follow-up research
        analysis_prompt = f"""
        Analyze this research result and suggest follow-up research tasks:
        
        Original task: {completed_task.description}
        Domain: {completed_task.domain}
        Results summary: {result.get('summary', 'No summary available')[:500]}
        
        Based on these results, suggest 0-2 specific follow-up research tasks that would:
        1. Fill gaps identified in the results
        2. Explore interesting leads mentioned
        3. Validate or challenge findings
        
        Return as JSON with format:
        {
            "followup_tasks": [
                {"description": "task description", "domain": "domain", "priority": 1-5},
                ...
            ]
        }
        
        Return empty array if no follow-up is needed.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a research analyst identifying follow-up research opportunities."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self._llm_request(messages, max_tokens=400)
            followup_data = json.loads(response)
            
            # Create follow-up tasks
            for task_data in followup_data.get("followup_tasks", []):
                followup_task = ResearchTask(
                    id=f"{self.current_session_id}_followup_{len(self.priority_queue)}",
                    description=task_data["description"],
                    domain=task_data["domain"],
                    priority=task_data.get("priority", 3),
                    metadata={
                        "session_id": self.current_session_id,
                        "parent_task": completed_task.id,
                        "followup": True
                    }
                )
                self.priority_queue.append(followup_task)
            
            if followup_data.get("followup_tasks"):
                logger.info(f"Generated {len(followup_data['followup_tasks'])} follow-up tasks")
                
        except Exception as e:
            logger.error(f"Error generating follow-up tasks: {e}")


class MessageBroker:
    """Handles message routing between agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_log: List[AgentMessage] = []
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the message broker"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.name} with message broker")
    
    async def start_routing(self):
        """Start the message routing loop"""
        logger.info("Starting message broker")
        
        while True:
            try:
                # Check outboxes of all agents
                for agent in self.agents.values():
                    try:
                        message = agent.outbox.get_nowait()
                        await self._route_message(message)
                    except asyncio.QueueEmpty:
                        continue
                
                await asyncio.sleep(0.05)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in message routing: {e}")
                await asyncio.sleep(1.0)
    
    async def _route_message(self, message: AgentMessage):
        """Route a message to the appropriate recipient"""
        receiver = self.agents.get(message.receiver)
        if receiver:
            await receiver.inbox.put(message)
            self.message_log.append(message)
            logger.debug(f"Routed message from {message.sender} to {message.receiver}")
        else:
            logger.warning(f"Unknown recipient: {message.receiver}")


class MultiAgentResearchSystem:
    """Main system that orchestrates the multi-agent research"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        
        # Core components
        self.message_broker = MessageBroker()
        self.coordinator = ResearchCoordinator(openai_api_key)
        self.agents: Dict[str, BaseAgent] = {}
        
        # System status
        self.is_running = False
        self.current_session: Optional[str] = None
        
        # Initialize system
        self._setup_system()
    
    def _setup_system(self):
        """Set up the multi-agent system"""
        # Register coordinator
        self.message_broker.register_agent(self.coordinator)
        self.agents[self.coordinator.agent_id] = self.coordinator
        
        # Register coordinator with itself (for agent management)
        self.coordinator.register_agent(self.coordinator)
    
    def add_agent(self, agent: BaseAgent):
        """Add a specialized agent to the system"""
        self.message_broker.register_agent(agent)
        self.coordinator.register_agent(agent)
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent: {agent.name}")
    
    async def start_system(self):
        """Start the multi-agent system"""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        self.is_running = True
        logger.info("Starting multi-agent research system")
        
        # Start message broker
        broker_task = asyncio.create_task(self.message_broker.start_routing())
        
        # Start all agents
        agent_tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start())
            agent_tasks.append(task)
        
        # Wait for all components
        try:
            await asyncio.gather(broker_task, *agent_tasks)
        except KeyboardInterrupt:
            logger.info("System shutdown requested")
        finally:
            self.is_running = False
    
    async def conduct_research(self, goal: str, max_duration: int = 300) -> Dict[str, Any]:
        """Conduct research on the given goal"""
        logger.info(f"Starting research: {goal}")
        
        # Start research session
        session_id = await self.coordinator.start_research_session(goal)
        self.current_session = session_id
        
        # Wait for research to complete or timeout
        start_time = datetime.now()
        
        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Check timeout
            if elapsed > max_duration:
                logger.info("Research session timed out")
                break
            
            # Check if research is complete
            if self._is_research_complete():
                logger.info("Research session completed")
                break
            
            await asyncio.sleep(5.0)
        
        # Collect results
        results = await self._collect_research_results()
        return results
    
    def _is_research_complete(self) -> bool:
        """Check if the current research session is complete"""
        # Simple completion check - no active tasks and some results
        active_tasks = len(self.coordinator.active_tasks)
        pending_tasks = len(self.coordinator.priority_queue)
        completed_tasks = len(self.coordinator.completed_tasks)
        
        return active_tasks == 0 and pending_tasks == 0 and completed_tasks > 0
    
    async def _collect_research_results(self) -> Dict[str, Any]:
        """Collect and synthesize research results"""
        completed_tasks = self.coordinator.completed_tasks
        
        results = {
            "session_id": self.current_session,
            "goal": self.coordinator.research_goal,
            "total_tasks": len(completed_tasks),
            "tasks_by_domain": {},
            "key_findings": [],
            "synthesis": None,
            "metadata": {
                "agents_used": list(self.agents.keys()),
                "research_plan": self.coordinator.research_plan,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Organize results by domain
        for task in completed_tasks.values():
            domain = task.domain
            if domain not in results["tasks_by_domain"]:
                results["tasks_by_domain"][domain] = []
            
            results["tasks_by_domain"][domain].append({
                "task": task.description,
                "results": task.results,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            })
        
        # Extract key findings
        for task in completed_tasks.values():
            for result in task.results:
                if "findings" in result:
                    results["key_findings"].extend(result["findings"])
        
        return results


# Export main classes
__all__ = [
    'MultiAgentResearchSystem',
    'BaseAgent', 
    'ResearchCoordinator',
    'MessageBroker',
    'ResearchTask',
    'AgentMessage',
    'MessageType',
    'AgentStatus'
]