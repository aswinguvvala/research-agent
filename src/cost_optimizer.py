"""
Cost Optimizer for Enhanced Research System
Configures all components to use the cheapest models and minimize token usage
while maintaining research quality.
"""

import openai
from typing import Dict, Any, List
import asyncio
import logging

logger = logging.getLogger(__name__)

class CostOptimizer:
    """Optimize system for minimum cost while maintaining quality"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Cost-optimized settings
        self.model_config = {
            "model": "gpt-4o-mini",  # Cheapest available model
            "temperature": 0.3,      # Lower for consistency and shorter responses
            "max_tokens": 200,       # Reduced default token limit
            "frequency_penalty": 0.1, # Reduce repetition
            "presence_penalty": 0.1   # Encourage conciseness
        }
        
        # Token usage tracking
        self.total_tokens_used = 0
        self.request_count = 0
        self.cost_estimate = 0.0
        
        # GPT-4o mini pricing (as of 2024)
        self.cost_per_1k_tokens = {
            "input": 0.000150,   # $0.15 per 1K input tokens
            "output": 0.000600   # $0.60 per 1K output tokens
        }
    
    async def llm_request(self, 
                         messages: List[Dict], 
                         max_tokens: int = None, 
                         temperature: float = None,
                         context: str = "general") -> str:
        """Cost-optimized LLM request with usage tracking"""
        
        # Use cost-optimized defaults
        actual_max_tokens = max_tokens or self.model_config["max_tokens"]
        actual_temperature = temperature or self.model_config["temperature"]
        
        # Adjust max_tokens based on context
        context_limits = {
            "quick": 100,
            "summary": 150, 
            "analysis": 250,
            "synthesis": 400,
            "detailed": 500
        }
        
        if context in context_limits:
            actual_max_tokens = min(actual_max_tokens, context_limits[context])
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=self.model_config["model"],
                    messages=messages,
                    max_tokens=actual_max_tokens,
                    temperature=actual_temperature,
                    frequency_penalty=self.model_config["frequency_penalty"],
                    presence_penalty=self.model_config["presence_penalty"]
                )
            )
            
            # Track usage
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            self.total_tokens_used += total_tokens
            self.request_count += 1
            
            # Calculate cost
            input_cost = (prompt_tokens / 1000) * self.cost_per_1k_tokens["input"]
            output_cost = (completion_tokens / 1000) * self.cost_per_1k_tokens["output"]
            request_cost = input_cost + output_cost
            self.cost_estimate += request_cost
            
            logger.debug(f"LLM Request - Tokens: {total_tokens}, Cost: ${request_cost:.4f}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Cost-optimized LLM request failed: {e}")
            return ""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        avg_tokens_per_request = self.total_tokens_used / max(self.request_count, 1)
        
        return {
            "total_tokens": self.total_tokens_used,
            "total_requests": self.request_count,
            "estimated_cost": self.cost_estimate,
            "avg_tokens_per_request": avg_tokens_per_request,
            "model": self.model_config["model"],
            "cost_per_request": self.cost_estimate / max(self.request_count, 1)
        }
    
    def create_cost_optimized_prompt(self, original_prompt: str, max_words: int = 100) -> str:
        """Create a cost-optimized version of a prompt"""
        
        cost_prefix = f"""Be concise and direct. Limit response to {max_words} words maximum. 
Focus on key points only. Avoid lengthy explanations.

"""
        
        return cost_prefix + original_prompt
    
    def optimize_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get optimized configuration for different agent types"""
        
        base_config = {
            "model": self.model_config["model"],
            "temperature": self.model_config["temperature"],
            "frequency_penalty": self.model_config["frequency_penalty"],
            "presence_penalty": self.model_config["presence_penalty"]
        }
        
        # Agent-specific optimizations
        agent_configs = {
            "academic": {
                **base_config,
                "max_tokens": 250,  # Academic needs slightly more detail
                "context": "analysis"
            },
            "technical": {
                **base_config,
                "max_tokens": 200,  # Technical can be more concise
                "context": "summary"
            },
            "business": {
                **base_config,
                "max_tokens": 150,  # Business wants quick summaries
                "context": "summary"
            },
            "social": {
                **base_config,
                "max_tokens": 150,  # Social impact can be concise
                "context": "summary"
            },
            "fact_checker": {
                **base_config,
                "max_tokens": 100,  # Fact checking is binary/short
                "context": "quick"
            },
            "synthesis": {
                **base_config,
                "max_tokens": 300,  # Synthesis needs more tokens
                "context": "synthesis"
            }
        }
        
        return agent_configs.get(agent_type, base_config)


# Global cost optimizer instance
_cost_optimizer = None

def get_cost_optimizer(openai_api_key: str = None) -> CostOptimizer:
    """Get or create the global cost optimizer"""
    global _cost_optimizer
    
    if _cost_optimizer is None and openai_api_key:
        _cost_optimizer = CostOptimizer(openai_api_key)
    
    return _cost_optimizer

def cost_optimized_llm_request(messages: List[Dict], 
                              max_tokens: int = None,
                              temperature: float = None,
                              context: str = "general",
                              openai_api_key: str = None) -> str:
    """Convenience function for cost-optimized LLM requests"""
    
    optimizer = get_cost_optimizer(openai_api_key)
    if optimizer:
        return asyncio.create_task(
            optimizer.llm_request(messages, max_tokens, temperature, context)
        )
    else:
        raise ValueError("Cost optimizer not initialized. Provide openai_api_key.")

# Export the main components
__all__ = [
    'CostOptimizer',
    'get_cost_optimizer', 
    'cost_optimized_llm_request'
]