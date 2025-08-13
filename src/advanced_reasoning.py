"""
Advanced Reasoning System
Implements chain-of-thought prompting, meta-cognitive loops, and sophisticated
reasoning capabilities for the multi-agent research system.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import re

from .multi_agent_research_system import BaseAgent

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning processes"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative" 
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    METACOGNITIVE = "metacognitive"
    SYNTHESIS = "synthesis"


@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain"""
    step_id: str
    step_type: ReasoningType
    question: str
    reasoning: str
    conclusion: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ReasoningChain:
    """Complete chain of reasoning for a problem"""
    chain_id: str
    problem_statement: str
    reasoning_steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    validation_results: Optional[Dict] = None


class ChainOfThoughtEngine:
    """Advanced chain-of-thought reasoning engine"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        
        # Reasoning templates for different types
        self.reasoning_templates = {
            ReasoningType.ANALYTICAL: self._get_analytical_template(),
            ReasoningType.CREATIVE: self._get_creative_template(),
            ReasoningType.CRITICAL: self._get_critical_template(),
            ReasoningType.STRATEGIC: self._get_strategic_template(),
            ReasoningType.METACOGNITIVE: self._get_metacognitive_template(),
            ReasoningType.SYNTHESIS: self._get_synthesis_template()
        }
        
        # Reasoning chains history
        self.reasoning_history: List[ReasoningChain] = []
        self.active_chains: Dict[str, ReasoningChain] = {}
    
    async def reason_through_problem(self, 
                                   problem: str, 
                                   context: Dict[str, Any] = None, 
                                   reasoning_type: ReasoningType = ReasoningType.ANALYTICAL) -> ReasoningChain:
        """Perform comprehensive reasoning through a problem"""
        
        chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.reasoning_history)}"
        context = context or {}
        
        logger.info(f"Starting reasoning chain {chain_id} for problem: {problem[:100]}...")
        
        try:
            # Step 1: Problem decomposition
            decomposition_step = await self._decompose_problem(problem, context)
            
            # Step 2: Evidence gathering assessment
            evidence_step = await self._assess_evidence_needs(problem, context, decomposition_step)
            
            # Step 3: Apply primary reasoning type
            primary_reasoning_step = await self._apply_primary_reasoning(
                problem, context, reasoning_type, [decomposition_step, evidence_step]
            )
            
            # Step 4: Critical evaluation
            critical_step = await self._apply_critical_reasoning(
                problem, context, [decomposition_step, evidence_step, primary_reasoning_step]
            )
            
            # Step 5: Metacognitive reflection
            meta_step = await self._apply_metacognitive_reasoning(
                problem, context, [decomposition_step, evidence_step, primary_reasoning_step, critical_step]
            )
            
            # Step 6: Final synthesis
            synthesis_step = await self._synthesize_conclusion(
                problem, context, [decomposition_step, evidence_step, primary_reasoning_step, critical_step, meta_step]
            )
            
            # Create reasoning chain
            reasoning_steps = [decomposition_step, evidence_step, primary_reasoning_step, critical_step, meta_step, synthesis_step]
            
            # Calculate overall confidence
            overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
            
            # Extract final conclusion
            final_conclusion = synthesis_step.conclusion
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                problem_statement=problem,
                reasoning_steps=reasoning_steps,
                final_conclusion=final_conclusion,
                overall_confidence=overall_confidence
            )
            
            # Validate the reasoning chain
            validation_results = await self._validate_reasoning_chain(reasoning_chain)
            reasoning_chain.validation_results = validation_results
            
            # Store the chain
            self.reasoning_history.append(reasoning_chain)
            self.active_chains[chain_id] = reasoning_chain
            
            logger.info(f"Completed reasoning chain {chain_id} with confidence {overall_confidence:.3f}")
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error in reasoning chain {chain_id}: {e}")
            # Return minimal chain with error information
            error_step = ReasoningStep(
                step_id="error_step",
                step_type=ReasoningType.ANALYTICAL,
                question="Error occurred during reasoning",
                reasoning=f"An error occurred: {str(e)}",
                conclusion="Unable to complete reasoning due to error",
                confidence=0.0
            )
            
            return ReasoningChain(
                chain_id=chain_id,
                problem_statement=problem,
                reasoning_steps=[error_step],
                final_conclusion="Reasoning failed due to error",
                overall_confidence=0.0
            )
    
    async def _decompose_problem(self, problem: str, context: Dict[str, Any]) -> ReasoningStep:
        """Decompose the problem into manageable components"""
        
        template = self.reasoning_templates[ReasoningType.ANALYTICAL]
        
        prompt = f"""
        PROBLEM DECOMPOSITION STEP:
        
        Problem: {problem}
        Context: {json.dumps(context, default=str)[:500]}
        
        {template}
        
        Decompose this problem by identifying:
        1. Core components and sub-problems
        2. Key variables and constraints  
        3. Dependencies and relationships
        4. Required information or data
        5. Potential approaches or methodologies
        
        Provide your reasoning as JSON:
        {{
            "question": "What are the key components of this problem?",
            "reasoning": "Step-by-step decomposition thinking...",
            "conclusion": "Clear summary of problem components",
            "confidence": 0.0-1.0,
            "evidence": ["evidence1", "evidence2"],
            "assumptions": ["assumption1", "assumption2"]
        }}
        """
        
        response = await self._llm_request(prompt, max_tokens=700)
        
        try:
            result = json.loads(response)
            return ReasoningStep(
                step_id="decomposition",
                step_type=ReasoningType.ANALYTICAL,
                question=result.get("question", "Problem decomposition"),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                confidence=result.get("confidence", 0.7),
                evidence=result.get("evidence", []),
                assumptions=result.get("assumptions", [])
            )
        except:
            return ReasoningStep(
                step_id="decomposition",
                step_type=ReasoningType.ANALYTICAL,
                question="Problem decomposition",
                reasoning="Failed to parse decomposition",
                conclusion="Problem requires multi-faceted analysis",
                confidence=0.5
            )
    
    async def _assess_evidence_needs(self, problem: str, context: Dict[str, Any], prev_step: ReasoningStep) -> ReasoningStep:
        """Assess what evidence is needed to solve the problem"""
        
        prompt = f"""
        EVIDENCE ASSESSMENT STEP:
        
        Problem: {problem}
        Previous reasoning: {prev_step.reasoning[:300]}
        Previous conclusion: {prev_step.conclusion}
        
        Based on the problem decomposition, assess what evidence is needed:
        1. What data or information is required?
        2. What sources would be most reliable?
        3. What assumptions need validation?
        4. What are the evidence quality requirements?
        5. What are the gaps in current information?
        
        Provide your assessment as JSON:
        {{
            "question": "What evidence do we need to solve this problem?",
            "reasoning": "Step-by-step evidence needs analysis...",
            "conclusion": "Summary of evidence requirements",
            "confidence": 0.0-1.0,
            "evidence": ["current_evidence1", "current_evidence2"],
            "assumptions": ["evidence_assumption1", "evidence_assumption2"]
        }}
        """
        
        response = await self._llm_request(prompt, max_tokens=600)
        
        try:
            result = json.loads(response)
            return ReasoningStep(
                step_id="evidence_assessment",
                step_type=ReasoningType.ANALYTICAL,
                question=result.get("question", "Evidence needs assessment"),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                confidence=result.get("confidence", 0.7),
                evidence=result.get("evidence", []),
                assumptions=result.get("assumptions", [])
            )
        except:
            return ReasoningStep(
                step_id="evidence_assessment",
                step_type=ReasoningType.ANALYTICAL,
                question="Evidence needs assessment",
                reasoning="Failed to assess evidence needs",
                conclusion="Evidence requirements need clarification",
                confidence=0.5
            )
    
    async def _apply_primary_reasoning(self, problem: str, context: Dict[str, Any], 
                                     reasoning_type: ReasoningType, prev_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply the primary reasoning approach to the problem"""
        
        template = self.reasoning_templates[reasoning_type]
        prev_context = "\n".join([f"Step {i+1}: {step.conclusion}" for i, step in enumerate(prev_steps)])
        
        prompt = f"""
        {reasoning_type.value.upper()} REASONING STEP:
        
        Problem: {problem}
        Previous conclusions: {prev_context}
        
        {template}
        
        Apply {reasoning_type.value} reasoning to address this problem.
        Consider the previous analysis and provide deep reasoning.
        
        Provide your reasoning as JSON:
        {{
            "question": "Key question for {reasoning_type.value} analysis",
            "reasoning": "Detailed step-by-step {reasoning_type.value} thinking...",
            "conclusion": "Clear conclusion from {reasoning_type.value} analysis",
            "confidence": 0.0-1.0,
            "evidence": ["supporting_evidence1", "supporting_evidence2"],
            "assumptions": ["key_assumption1", "key_assumption2"]
        }}
        """
        
        response = await self._llm_request(prompt, max_tokens=800)
        
        try:
            result = json.loads(response)
            return ReasoningStep(
                step_id=f"primary_{reasoning_type.value}",
                step_type=reasoning_type,
                question=result.get("question", f"{reasoning_type.value} analysis"),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                confidence=result.get("confidence", 0.7),
                evidence=result.get("evidence", []),
                assumptions=result.get("assumptions", [])
            )
        except:
            return ReasoningStep(
                step_id=f"primary_{reasoning_type.value}",
                step_type=reasoning_type,
                question=f"{reasoning_type.value} analysis",
                reasoning=f"Failed to apply {reasoning_type.value} reasoning",
                conclusion=f"{reasoning_type.value} approach suggests further analysis needed",
                confidence=0.5
            )
    
    async def _apply_critical_reasoning(self, problem: str, context: Dict[str, Any], prev_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply critical reasoning to evaluate the previous conclusions"""
        
        template = self.reasoning_templates[ReasoningType.CRITICAL]
        prev_context = "\n".join([f"Step {i+1} ({step.step_type.value}): {step.conclusion}" for i, step in enumerate(prev_steps)])
        
        prompt = f"""
        CRITICAL EVALUATION STEP:
        
        Problem: {problem}
        Previous reasoning chain: {prev_context}
        
        {template}
        
        Critically evaluate the previous reasoning:
        1. What are the strengths and weaknesses of each conclusion?
        2. What assumptions might be questionable?
        3. What alternative explanations exist?
        4. What are the potential flaws or biases?
        5. What evidence contradicts the conclusions?
        
        Provide your critical analysis as JSON:
        {{
            "question": "How valid and reliable is the reasoning so far?",
            "reasoning": "Critical evaluation of each step and overall chain...",
            "conclusion": "Assessment of reasoning validity and areas of concern",
            "confidence": 0.0-1.0,
            "evidence": ["contradictory_evidence1", "supporting_evidence2"],
            "assumptions": ["questionable_assumption1", "questionable_assumption2"]
        }}
        """
        
        response = await self._llm_request(prompt, max_tokens=700)
        
        try:
            result = json.loads(response)
            return ReasoningStep(
                step_id="critical_evaluation",
                step_type=ReasoningType.CRITICAL,
                question=result.get("question", "Critical evaluation of reasoning"),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                confidence=result.get("confidence", 0.8),
                evidence=result.get("evidence", []),
                assumptions=result.get("assumptions", [])
            )
        except:
            return ReasoningStep(
                step_id="critical_evaluation", 
                step_type=ReasoningType.CRITICAL,
                question="Critical evaluation of reasoning",
                reasoning="Failed to perform critical evaluation",
                conclusion="Previous reasoning needs critical examination",
                confidence=0.6
            )
    
    async def _apply_metacognitive_reasoning(self, problem: str, context: Dict[str, Any], prev_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply metacognitive reasoning to reflect on the thinking process"""
        
        template = self.reasoning_templates[ReasoningType.METACOGNITIVE]
        confidence_scores = [step.confidence for step in prev_steps]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        prompt = f"""
        METACOGNITIVE REFLECTION STEP:
        
        Problem: {problem}
        Number of reasoning steps: {len(prev_steps)}
        Average confidence: {avg_confidence:.3f}
        Confidence range: {min(confidence_scores):.3f} - {max(confidence_scores):.3f}
        
        {template}
        
        Reflect on the reasoning process itself:
        1. How effective was the overall reasoning approach?
        2. What cognitive biases might have influenced the thinking?
        3. What alternative reasoning strategies could be used?
        4. How confident should we be in the overall conclusions?
        5. What would improve the reasoning quality?
        
        Provide your metacognitive reflection as JSON:
        {{
            "question": "How well did we reason through this problem?",
            "reasoning": "Reflection on the reasoning process, biases, and effectiveness...",
            "conclusion": "Assessment of reasoning quality and suggested improvements",
            "confidence": 0.0-1.0,
            "evidence": ["process_strength1", "process_weakness1"],
            "assumptions": ["meta_assumption1", "meta_assumption2"]
        }}
        """
        
        response = await self._llm_request(prompt, max_tokens=600)
        
        try:
            result = json.loads(response)
            return ReasoningStep(
                step_id="metacognitive_reflection",
                step_type=ReasoningType.METACOGNITIVE,
                question=result.get("question", "Metacognitive reflection"),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                confidence=result.get("confidence", 0.7),
                evidence=result.get("evidence", []),
                assumptions=result.get("assumptions", [])
            )
        except:
            return ReasoningStep(
                step_id="metacognitive_reflection",
                step_type=ReasoningType.METACOGNITIVE,
                question="Metacognitive reflection",
                reasoning="Failed to perform metacognitive reflection",
                conclusion="The reasoning process itself needs evaluation",
                confidence=0.6
            )
    
    async def _synthesize_conclusion(self, problem: str, context: Dict[str, Any], prev_steps: List[ReasoningStep]) -> ReasoningStep:
        """Synthesize all previous reasoning into a final conclusion"""
        
        template = self.reasoning_templates[ReasoningType.SYNTHESIS]
        
        # Prepare summary of all steps
        steps_summary = []
        for i, step in enumerate(prev_steps):
            steps_summary.append(f"""
            Step {i+1} ({step.step_type.value}):
            Question: {step.question}
            Conclusion: {step.conclusion}
            Confidence: {step.confidence:.3f}
            """)
        
        steps_text = "\n".join(steps_summary)
        
        prompt = f"""
        SYNTHESIS AND FINAL CONCLUSION STEP:
        
        Problem: {problem}
        
        All reasoning steps:
        {steps_text}
        
        {template}
        
        Synthesize all the reasoning into a final, comprehensive conclusion:
        1. What is the most likely answer to the original problem?
        2. What are the key insights from all reasoning approaches?
        3. What caveats and limitations should be noted?
        4. What is the overall confidence level?
        5. What follow-up questions or actions are suggested?
        
        Provide your synthesis as JSON:
        {{
            "question": "What is the final answer to the original problem?",
            "reasoning": "Integration of all previous reasoning steps...",
            "conclusion": "Comprehensive final answer with caveats and confidence",
            "confidence": 0.0-1.0,
            "evidence": ["strongest_evidence1", "strongest_evidence2"],
            "assumptions": ["final_assumption1", "final_assumption2"]
        }}
        """
        
        response = await self._llm_request(prompt, max_tokens=800)
        
        try:
            result = json.loads(response)
            return ReasoningStep(
                step_id="final_synthesis",
                step_type=ReasoningType.SYNTHESIS,
                question=result.get("question", "Final synthesis"),
                reasoning=result.get("reasoning", ""),
                conclusion=result.get("conclusion", ""),
                confidence=result.get("confidence", 0.7),
                evidence=result.get("evidence", []),
                assumptions=result.get("assumptions", [])
            )
        except:
            return ReasoningStep(
                step_id="final_synthesis",
                step_type=ReasoningType.SYNTHESIS,
                question="Final synthesis",
                reasoning="Failed to synthesize final conclusion",
                conclusion="Multiple perspectives suggest a complex answer requiring further analysis",
                confidence=0.5
            )
    
    async def _validate_reasoning_chain(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Validate the reasoning chain for logical consistency and quality"""
        
        validation_prompt = f"""
        REASONING VALIDATION:
        
        Problem: {chain.problem_statement}
        
        Reasoning Chain Summary:
        {chr(10).join([f"{i+1}. {step.step_type.value}: {step.conclusion}" for i, step in enumerate(chain.reasoning_steps)])}
        
        Final Conclusion: {chain.final_conclusion}
        Overall Confidence: {chain.overall_confidence:.3f}
        
        Validate this reasoning chain:
        1. Logical consistency between steps
        2. Evidence quality and support
        3. Assumption validity
        4. Conclusion alignment with reasoning
        5. Overall reasoning quality
        
        Provide validation as JSON:
        {{
            "logical_consistency": 0.0-1.0,
            "evidence_quality": 0.0-1.0,
            "assumption_validity": 0.0-1.0,
            "conclusion_alignment": 0.0-1.0,
            "overall_quality": 0.0-1.0,
            "major_issues": ["issue1", "issue2"],
            "strengths": ["strength1", "strength2"],
            "improvement_suggestions": ["suggestion1", "suggestion2"]
        }}
        """
        
        try:
            response = await self._llm_request(validation_prompt, max_tokens=600)
            validation = json.loads(response)
            
            # Add metadata
            validation["validated_at"] = datetime.now().isoformat()
            validation["validation_method"] = "llm_validation"
            
            return validation
            
        except:
            return {
                "logical_consistency": 0.6,
                "evidence_quality": 0.6, 
                "assumption_validity": 0.6,
                "conclusion_alignment": 0.6,
                "overall_quality": 0.6,
                "major_issues": ["validation_failed"],
                "strengths": ["completed_reasoning_chain"],
                "improvement_suggestions": ["manual_validation_needed"],
                "validated_at": datetime.now().isoformat(),
                "validation_method": "fallback"
            }
    
    async def _llm_request(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500) -> str:
        """Make a request to the LLM with reasoning-optimized settings"""
        import openai
        openai.api_key = self.openai_api_key
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert reasoning assistant. Think step by step and provide detailed, logical reasoning. Always return valid JSON when requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent reasoning
                    max_tokens=max_tokens
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM request error in reasoning: {e}")
            return '{"error": "LLM request failed"}'
    
    # Reasoning templates
    def _get_analytical_template(self) -> str:
        return """
        Apply analytical reasoning by:
        - Breaking down complex information into components
        - Identifying patterns and relationships
        - Using logical deduction and evidence
        - Considering multiple perspectives systematically
        - Drawing conclusions based on data and facts
        """
    
    def _get_creative_template(self) -> str:
        return """
        Apply creative reasoning by:
        - Generating novel approaches and solutions
        - Making unexpected connections between ideas
        - Challenging conventional assumptions
        - Exploring alternative possibilities
        - Using analogies and metaphorical thinking
        """
    
    def _get_critical_template(self) -> str:
        return """
        Apply critical reasoning by:
        - Questioning assumptions and premises
        - Evaluating evidence quality and reliability
        - Identifying logical fallacies and biases
        - Considering alternative explanations
        - Assessing argument strength and validity
        """
    
    def _get_strategic_template(self) -> str:
        return """
        Apply strategic reasoning by:
        - Considering long-term implications and consequences
        - Evaluating costs, benefits, and trade-offs
        - Identifying stakeholders and their interests
        - Planning sequences of actions
        - Anticipating potential challenges and opportunities
        """
    
    def _get_metacognitive_template(self) -> str:
        return """
        Apply metacognitive reasoning by:
        - Reflecting on your own thinking process
        - Evaluating the effectiveness of reasoning strategies
        - Identifying cognitive biases and limitations
        - Considering alternative approaches to thinking
        - Assessing confidence and uncertainty levels
        """
    
    def _get_synthesis_template(self) -> str:
        return """
        Apply synthesis reasoning by:
        - Integrating information from multiple sources
        - Combining different perspectives and approaches
        - Creating coherent conclusions from diverse evidence
        - Balancing conflicting information
        - Generating comprehensive understanding
        """
    
    def get_reasoning_summary(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a reasoning chain"""
        if chain_id not in self.active_chains:
            return None
        
        chain = self.active_chains[chain_id]
        
        return {
            "chain_id": chain_id,
            "problem": chain.problem_statement,
            "steps_count": len(chain.reasoning_steps),
            "reasoning_types": [step.step_type.value for step in chain.reasoning_steps],
            "confidence_by_step": [step.confidence for step in chain.reasoning_steps],
            "overall_confidence": chain.overall_confidence,
            "final_conclusion": chain.final_conclusion,
            "validation": chain.validation_results,
            "created_at": chain.created_at.isoformat()
        }
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get summary of all reasoning chains"""
        return [self.get_reasoning_summary(chain.chain_id) for chain in self.reasoning_history]


class ReasoningEnhancedAgent(BaseAgent):
    """Base agent enhanced with advanced reasoning capabilities"""
    
    def __init__(self, agent_id: str, name: str, openai_api_key: str):
        super().__init__(agent_id, name, openai_api_key)
        
        # Initialize reasoning engine
        self.reasoning_engine = ChainOfThoughtEngine(openai_api_key)
        
        # Add reasoning capability
        self.capabilities.add("advanced_reasoning")
    
    async def reason_through_task(self, task_description: str, context: Dict[str, Any] = None, 
                                reasoning_type: ReasoningType = ReasoningType.ANALYTICAL) -> ReasoningChain:
        """Use advanced reasoning to work through a task"""
        return await self.reasoning_engine.reason_through_problem(task_description, context, reasoning_type)
    
    async def validate_reasoning(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """Validate a reasoning chain"""
        return await self.reasoning_engine._validate_reasoning_chain(reasoning_chain)
    
    def get_reasoning_capabilities(self) -> List[str]:
        """Get list of available reasoning types"""
        return [rt.value for rt in ReasoningType]


# Export the reasoning system
__all__ = [
    'ChainOfThoughtEngine',
    'ReasoningEnhancedAgent', 
    'ReasoningType',
    'ReasoningStep',
    'ReasoningChain'
]