"""
Specialized Research Agents
Domain-specific agents that excel in particular areas of research,
providing deeper expertise than a single generalist agent.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import logging
from abc import abstractmethod

import openai
import arxiv
import requests
from bs4 import BeautifulSoup
import feedparser

from .multi_agent_research_system import BaseAgent, ResearchTask, AgentMessage, MessageType

logger = logging.getLogger(__name__)


class AcademicSpecialist(BaseAgent):
    """Specialized agent for academic research, papers, and scholarly content"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("academic_specialist", "Academic Research Specialist", openai_api_key)
        self.specialization = "academic"
        self.capabilities.update([
            "arxiv_search", "paper_analysis", "citation_tracking",
            "academic_synthesis", "methodology_extraction", "literature_review"
        ])
        
        # Academic-specific configurations
        self.preferred_sources = ["arxiv", "semantic_scholar", "pubmed", "acm_digital_library"]
        self.paper_cache = {}
        self.citation_network = {}
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process academic research tasks with paper analysis and synthesis"""
        logger.info(f"Academic specialist processing task: {task.description}")
        
        try:
            # Extract research topics and keywords
            keywords = await self._extract_keywords(task.description)
            
            # Search academic sources
            papers = await self._search_academic_sources(keywords)
            
            # Analyze papers for relevance and quality
            analyzed_papers = await self._analyze_papers(papers, task.description)
            
            # Synthesize academic findings
            synthesis = await self._synthesize_academic_findings(analyzed_papers, task.description)
            
            # Identify research gaps and future directions
            gaps = await self._identify_research_gaps(analyzed_papers, synthesis)
            
            return {
                "task_id": task.id,
                "agent_type": "academic",
                "papers_found": len(papers),
                "papers_analyzed": len(analyzed_papers),
                "findings": [paper["key_findings"] for paper in analyzed_papers],
                "synthesis": synthesis,
                "research_gaps": gaps,
                "methodologies": [paper.get("methodology", "") for paper in analyzed_papers if paper.get("methodology")],
                "confidence": 0.9,
                "sources": [paper["source"] for paper in analyzed_papers],
                "metadata": {
                    "search_terms": keywords,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in academic research task: {e}")
            return {
                "task_id": task.id,
                "agent_type": "academic",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _extract_keywords(self, description: str) -> List[str]:
        """Extract academic keywords from task description"""
        prompt = f"""
        Extract 5-7 academic keywords and search terms from this research description:
        "{description}"
        
        Focus on:
        - Technical terms and concepts
        - Research methodologies
        - Application domains
        - Related fields of study
        
        Return as JSON array: ["keyword1", "keyword2", ...]
        """
        
        messages = [
            {"role": "system", "content": "You are an academic research expert extracting search terms."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=200)
            return json.loads(response)
        except:
            # Fallback: simple word extraction
            words = description.lower().replace(",", "").replace(".", "").split()
            return [word for word in words if len(word) > 4][:7]
    
    async def _search_academic_sources(self, keywords: List[str]) -> List[Dict]:
        """Search multiple academic sources for papers"""
        all_papers = []
        
        # Search ArXiv
        for keyword in keywords[:3]:  # Limit to prevent too many requests
            try:
                arxiv_papers = await self._search_arxiv(keyword)
                all_papers.extend(arxiv_papers)
            except Exception as e:
                logger.warning(f"ArXiv search failed for {keyword}: {e}")
        
        # Search other academic sources (placeholder for now)
        # In production, would integrate Semantic Scholar, PubMed, etc.
        
        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)
        return unique_papers[:10]  # Limit for processing efficiency
    
    async def _search_arxiv(self, query: str) -> List[Dict]:
        """Search ArXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for paper in search.results():
                papers.append({
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published,
                    "url": paper.entry_id,
                    "categories": paper.categories,
                    "source": "arxiv"
                })
            
            return papers
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title_words = set(paper["title"].lower().split())
            
            # Check if this paper is too similar to existing ones
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
                if overlap > 0.8:  # 80% word overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(paper["title"].lower())
        
        return unique_papers
    
    async def _analyze_papers(self, papers: List[Dict], task_description: str) -> List[Dict]:
        """Analyze papers for relevance and extract key information"""
        analyzed_papers = []
        
        for paper in papers[:5]:  # Analyze top 5 papers
            try:
                analysis = await self._analyze_single_paper(paper, task_description)
                if analysis["relevance_score"] > 0.3:  # Only include relevant papers
                    analyzed_papers.append({**paper, **analysis})
            except Exception as e:
                logger.warning(f"Paper analysis failed: {e}")
        
        return analyzed_papers
    
    async def _analyze_single_paper(self, paper: Dict, task_description: str) -> Dict:
        """Analyze a single paper for relevance and key findings"""
        prompt = f"""
        Analyze this academic paper for relevance to the research task:
        
        Task: {task_description}
        
        Paper: {paper["title"]}
        Abstract: {paper["abstract"][:800]}
        Authors: {', '.join(paper["authors"][:3])}
        
        Provide analysis as JSON:
        {{
            "relevance_score": 0.0-1.0,
            "key_findings": ["finding1", "finding2", "finding3"],
            "methodology": "brief description of methods used",
            "limitations": "key limitations mentioned",
            "significance": "why this paper matters",
            "novel_contributions": ["contribution1", "contribution2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an academic paper analyst. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=600)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Paper analysis error: {e}")
            return {
                "relevance_score": 0.5,
                "key_findings": ["Analysis failed"],
                "methodology": "Unknown",
                "limitations": "Unknown",
                "significance": "Potentially relevant",
                "novel_contributions": []
            }
    
    async def _synthesize_academic_findings(self, papers: List[Dict], task_description: str) -> str:
        """Synthesize findings from multiple papers"""
        if not papers:
            return "No relevant papers found for synthesis."
        
        findings_summary = []
        methodologies = []
        
        for paper in papers:
            if paper.get("key_findings"):
                findings_summary.extend(paper["key_findings"])
            if paper.get("methodology"):
                methodologies.append(paper["methodology"])
        
        synthesis_prompt = f"""
        Synthesize academic research findings for the task: "{task_description}"
        
        Key findings from {len(papers)} relevant papers:
        {chr(10).join([f"- {finding}" for finding in findings_summary[:15]])}
        
        Methodologies used:
        {chr(10).join([f"- {method}" for method in methodologies[:5]])}
        
        Provide a comprehensive synthesis that:
        1. Summarizes the current state of research
        2. Identifies consensus and disagreements
        3. Highlights most significant findings
        4. Notes methodological approaches
        5. Suggests implications for the research task
        
        Keep it academic but accessible, around 200 words.
        """
        
        messages = [
            {"role": "system", "content": "You are an academic research synthesizer."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        try:
            return await self._llm_request(messages, max_tokens=400)
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return "Synthesis failed due to processing error."
    
    async def _identify_research_gaps(self, papers: List[Dict], synthesis: str) -> List[str]:
        """Identify research gaps and future directions"""
        prompt = f"""
        Based on this academic research synthesis and paper analysis, identify 3-5 specific research gaps:
        
        Synthesis: {synthesis[:500]}
        
        Number of papers analyzed: {len(papers)}
        
        Identify gaps as specific questions or areas that need more research.
        Return as JSON array: ["gap1", "gap2", "gap3"]
        """
        
        messages = [
            {"role": "system", "content": "You are a research gap analyst. Return JSON array only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=300)
            return json.loads(response)
        except:
            return ["Research methodology validation needed", "Longitudinal studies required", "Cross-domain applications unexplored"]


class TechnicalSpecialist(BaseAgent):
    """Specialized agent for technical implementation, engineering, and practical applications"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("technical_specialist", "Technical Implementation Specialist", openai_api_key)
        self.specialization = "technical"
        self.capabilities.update([
            "technical_analysis", "implementation_research", "tool_evaluation",
            "performance_analysis", "best_practices", "troubleshooting"
        ])
        
        # Technical-specific configurations
        self.preferred_sources = ["github", "stackoverflow", "technical_blogs", "documentation"]
        self.tool_database = {}
        self.implementation_patterns = {}
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process technical research tasks focusing on implementation and practical aspects"""
        logger.info(f"Technical specialist processing task: {task.description}")
        
        try:
            # Extract technical concepts and tools
            tech_concepts = await self._extract_technical_concepts(task.description)
            
            # Research implementation approaches
            implementations = await self._research_implementations(tech_concepts)
            
            # Analyze tools and technologies
            tools_analysis = await self._analyze_tools_and_technologies(tech_concepts)
            
            # Identify best practices and patterns
            best_practices = await self._identify_best_practices(implementations)
            
            # Assess technical challenges and solutions
            challenges = await self._assess_technical_challenges(task.description, implementations)
            
            return {
                "task_id": task.id,
                "agent_type": "technical",
                "technical_concepts": tech_concepts,
                "implementations_found": len(implementations),
                "findings": implementations,
                "tools_analysis": tools_analysis,
                "best_practices": best_practices,
                "challenges": challenges,
                "confidence": 0.85,
                "sources": ["technical_documentation", "implementation_examples"],
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "focus_areas": tech_concepts
                }
            }
            
        except Exception as e:
            logger.error(f"Error in technical research task: {e}")
            return {
                "task_id": task.id,
                "agent_type": "technical",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _extract_technical_concepts(self, description: str) -> List[str]:
        """Extract technical concepts and technologies from task description"""
        prompt = f"""
        Extract technical concepts, technologies, and implementation-related terms from:
        "{description}"
        
        Focus on:
        - Programming languages and frameworks
        - Technical methodologies and patterns
        - Tools and platforms
        - Implementation challenges
        - Performance considerations
        
        Return as JSON array: ["concept1", "concept2", ...]
        Limit to 7 most relevant terms.
        """
        
        messages = [
            {"role": "system", "content": "You are a technical concept extractor."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=200)
            return json.loads(response)
        except:
            return ["implementation", "architecture", "performance", "scalability"]
    
    async def _research_implementations(self, concepts: List[str]) -> List[Dict]:
        """Research implementation approaches for the given concepts"""
        implementations = []
        
        for concept in concepts[:3]:
            impl_data = await self._research_single_implementation(concept)
            if impl_data:
                implementations.append(impl_data)
        
        return implementations
    
    async def _research_single_implementation(self, concept: str) -> Dict:
        """Research implementation details for a single concept"""
        prompt = f"""
        Research technical implementation details for: "{concept}"
        
        Provide comprehensive technical information as JSON:
        {{
            "concept": "{concept}",
            "implementation_approaches": ["approach1", "approach2", "approach3"],
            "popular_tools": ["tool1", "tool2", "tool3"],
            "code_examples": "brief code snippet or pseudocode",
            "performance_considerations": ["consideration1", "consideration2"],
            "common_patterns": ["pattern1", "pattern2"],
            "pitfalls": ["pitfall1", "pitfall2"],
            "resources": ["resource1", "resource2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a technical implementation expert. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=700)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Implementation research failed for {concept}: {e}")
            return None
    
    async def _analyze_tools_and_technologies(self, concepts: List[str]) -> Dict:
        """Analyze relevant tools and technologies"""
        prompt = f"""
        Analyze tools and technologies relevant to these concepts: {', '.join(concepts)}
        
        Provide analysis as JSON:
        {{
            "recommended_tools": [
                {{"name": "tool1", "purpose": "what it does", "pros": ["pro1"], "cons": ["con1"]}},
                {{"name": "tool2", "purpose": "what it does", "pros": ["pro1"], "cons": ["con1"]}}
            ],
            "technology_stack": ["tech1", "tech2", "tech3"],
            "learning_curve": "easy|medium|hard",
            "maturity_assessment": "emerging|established|mature",
            "ecosystem_health": "description of community and support"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a technology analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=600)
            return json.loads(response)
        except:
            return {
                "recommended_tools": [],
                "technology_stack": concepts,
                "learning_curve": "medium",
                "maturity_assessment": "established",
                "ecosystem_health": "Analysis unavailable"
            }
    
    async def _identify_best_practices(self, implementations: List[Dict]) -> List[str]:
        """Identify best practices from implementation research"""
        if not implementations:
            return ["Follow established patterns", "Prioritize maintainability", "Consider scalability"]
        
        all_patterns = []
        for impl in implementations:
            all_patterns.extend(impl.get("common_patterns", []))
        
        # Use LLM to synthesize best practices
        prompt = f"""
        Based on these implementation patterns, identify 5 key best practices:
        {', '.join(all_patterns[:20])}
        
        Return as JSON array of specific, actionable best practices.
        """
        
        messages = [
            {"role": "system", "content": "You are a best practices expert. Return JSON array."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=300)
            return json.loads(response)
        except:
            return ["Use established patterns", "Test thoroughly", "Document implementations", "Consider performance", "Plan for maintenance"]
    
    async def _assess_technical_challenges(self, description: str, implementations: List[Dict]) -> List[Dict]:
        """Assess technical challenges and provide solutions"""
        challenges_text = []
        for impl in implementations:
            challenges_text.extend(impl.get("pitfalls", []))
        
        prompt = f"""
        Assess technical challenges for: "{description}"
        
        Known pitfalls: {', '.join(challenges_text[:10])}
        
        Return as JSON array of challenge objects:
        [
            {{
                "challenge": "description of challenge",
                "severity": "low|medium|high",
                "solutions": ["solution1", "solution2"],
                "prevention": "how to prevent this issue"
            }}
        ]
        
        Limit to 4 most important challenges.
        """
        
        messages = [
            {"role": "system", "content": "You are a technical challenge analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=500)
            return json.loads(response)
        except:
            return [{
                "challenge": "Implementation complexity",
                "severity": "medium",
                "solutions": ["Break down into smaller components", "Use established frameworks"],
                "prevention": "Plan architecture carefully"
            }]


class BusinessSpecialist(BaseAgent):
    """Specialized agent for business applications, market research, and commercial aspects"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("business_specialist", "Business Applications Specialist", openai_api_key)
        self.specialization = "business"
        self.capabilities.update([
            "market_analysis", "business_case_development", "roi_analysis",
            "competitive_analysis", "business_model_research", "adoption_barriers"
        ])
        
        # Business-specific configurations
        self.preferred_sources = ["business_news", "market_reports", "company_data", "financial_data"]
        self.industry_data = {}
        self.market_trends = {}
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process business research tasks focusing on commercial viability and applications"""
        logger.info(f"Business specialist processing task: {task.description}")
        
        try:
            # Extract business concepts and markets
            business_concepts = await self._extract_business_concepts(task.description)
            
            # Research market applications
            market_analysis = await self._research_market_applications(business_concepts)
            
            # Analyze business value and ROI
            value_analysis = await self._analyze_business_value(task.description, business_concepts)
            
            # Identify market opportunities and challenges
            opportunities = await self._identify_market_opportunities(market_analysis)
            
            # Assess competitive landscape
            competitive_analysis = await self._assess_competitive_landscape(business_concepts)
            
            return {
                "task_id": task.id,
                "agent_type": "business",
                "business_concepts": business_concepts,
                "market_analysis": market_analysis,
                "value_analysis": value_analysis,
                "opportunities": opportunities,
                "competitive_analysis": competitive_analysis,
                "findings": [market_analysis, value_analysis],
                "confidence": 0.8,
                "sources": ["market_research", "business_analysis"],
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "focus_markets": business_concepts
                }
            }
            
        except Exception as e:
            logger.error(f"Error in business research task: {e}")
            return {
                "task_id": task.id,
                "agent_type": "business",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _extract_business_concepts(self, description: str) -> List[str]:
        """Extract business-relevant concepts from task description"""
        prompt = f"""
        Extract business and commercial concepts from: "{description}"
        
        Focus on:
        - Market segments and industries
        - Business applications and use cases
        - Revenue models and monetization
        - Competitive advantages
        - Market size and opportunities
        
        Return as JSON array: ["concept1", "concept2", ...]
        Limit to 6 most relevant business terms.
        """
        
        messages = [
            {"role": "system", "content": "You are a business concept analyst."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=200)
            return json.loads(response)
        except:
            return ["market_opportunity", "business_application", "commercial_viability"]
    
    async def _research_market_applications(self, concepts: List[str]) -> Dict:
        """Research market applications and use cases"""
        prompt = f"""
        Research market applications for these business concepts: {', '.join(concepts)}
        
        Provide comprehensive market analysis as JSON:
        {{
            "primary_markets": ["market1", "market2", "market3"],
            "use_cases": ["use_case1", "use_case2", "use_case3"],
            "market_size_estimate": "small|medium|large|very_large",
            "growth_rate": "declining|stable|growing|rapidly_growing",
            "key_players": ["company1", "company2", "company3"],
            "adoption_stage": "emerging|early|mainstream|mature",
            "barriers_to_entry": ["barrier1", "barrier2"],
            "success_factors": ["factor1", "factor2", "factor3"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a market research analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=600)
            return json.loads(response)
        except:
            return {
                "primary_markets": ["enterprise", "consumer"],
                "use_cases": ["efficiency_improvement", "cost_reduction"],
                "market_size_estimate": "medium",
                "growth_rate": "growing",
                "key_players": ["established_companies"],
                "adoption_stage": "early",
                "barriers_to_entry": ["high_complexity", "regulatory_requirements"],
                "success_factors": ["quality", "user_experience", "cost_effectiveness"]
            }
    
    async def _analyze_business_value(self, description: str, concepts: List[str]) -> Dict:
        """Analyze business value proposition and ROI potential"""
        prompt = f"""
        Analyze business value for: "{description}"
        Concepts: {', '.join(concepts)}
        
        Provide value analysis as JSON:
        {{
            "value_propositions": ["value1", "value2", "value3"],
            "revenue_models": ["model1", "model2"],
            "cost_considerations": ["cost1", "cost2", "cost3"],
            "roi_timeline": "immediate|short_term|medium_term|long_term",
            "business_risks": ["risk1", "risk2"],
            "success_metrics": ["metric1", "metric2", "metric3"],
            "investment_requirements": "low|medium|high|very_high",
            "scalability_potential": "limited|moderate|high|very_high"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a business value analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=500)
            return json.loads(response)
        except:
            return {
                "value_propositions": ["efficiency_gains", "cost_savings"],
                "revenue_models": ["subscription", "licensing"],
                "cost_considerations": ["development_costs", "operational_costs"],
                "roi_timeline": "medium_term",
                "business_risks": ["market_uncertainty", "competitive_pressure"],
                "success_metrics": ["user_adoption", "revenue_growth"],
                "investment_requirements": "medium",
                "scalability_potential": "high"
            }
    
    async def _identify_market_opportunities(self, market_analysis: Dict) -> List[Dict]:
        """Identify specific market opportunities"""
        markets = market_analysis.get("primary_markets", [])
        use_cases = market_analysis.get("use_cases", [])
        
        prompt = f"""
        Based on this market analysis, identify specific opportunities:
        Markets: {', '.join(markets)}
        Use cases: {', '.join(use_cases)}
        Growth rate: {market_analysis.get("growth_rate", "unknown")}
        
        Return opportunities as JSON array:
        [
            {{
                "opportunity": "description",
                "market_segment": "specific segment",
                "potential_impact": "low|medium|high",
                "timeframe": "short|medium|long",
                "requirements": ["req1", "req2"]
            }}
        ]
        
        Limit to 3 most promising opportunities.
        """
        
        messages = [
            {"role": "system", "content": "You are a market opportunity analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=400)
            return json.loads(response)
        except:
            return [{
                "opportunity": "Market expansion potential",
                "market_segment": "emerging_markets",
                "potential_impact": "medium",
                "timeframe": "medium",
                "requirements": ["market_research", "strategic_partnerships"]
            }]
    
    async def _assess_competitive_landscape(self, concepts: List[str]) -> Dict:
        """Assess the competitive landscape"""
        prompt = f"""
        Assess competitive landscape for: {', '.join(concepts)}
        
        Provide competitive analysis as JSON:
        {{
            "competitive_intensity": "low|medium|high|very_high",
            "market_leaders": ["leader1", "leader2"],
            "competitive_advantages": ["advantage1", "advantage2"],
            "differentiation_opportunities": ["opportunity1", "opportunity2"],
            "threat_level": "low|medium|high",
            "market_positioning": "niche|mainstream|premium",
            "competitive_gaps": ["gap1", "gap2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a competitive analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=400)
            return json.loads(response)
        except:
            return {
                "competitive_intensity": "medium",
                "market_leaders": ["established_player"],
                "competitive_advantages": ["innovation", "quality"],
                "differentiation_opportunities": ["user_experience", "pricing"],
                "threat_level": "medium",
                "market_positioning": "mainstream",
                "competitive_gaps": ["customer_service", "technical_support"]
            }


class SocialSpecialist(BaseAgent):
    """Specialized agent for social impact, trends, and human factors research"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("social_specialist", "Social Impact Specialist", openai_api_key)
        self.specialization = "social"
        self.capabilities.update([
            "social_impact_analysis", "trend_research", "human_factors",
            "ethical_considerations", "adoption_patterns", "cultural_analysis"
        ])
        
        # Social-specific configurations
        self.preferred_sources = ["social_media", "surveys", "demographic_data", "cultural_reports"]
        self.trend_data = {}
        self.social_indicators = {}
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process social research tasks focusing on human impact and societal factors"""
        logger.info(f"Social specialist processing task: {task.description}")
        
        try:
            # Extract social concepts and factors
            social_concepts = await self._extract_social_concepts(task.description)
            
            # Analyze social impact and implications
            impact_analysis = await self._analyze_social_impact(task.description, social_concepts)
            
            # Research adoption patterns and barriers
            adoption_analysis = await self._research_adoption_patterns(social_concepts)
            
            # Identify ethical considerations
            ethical_analysis = await self._identify_ethical_considerations(task.description)
            
            # Analyze cultural and demographic factors
            cultural_analysis = await self._analyze_cultural_factors(social_concepts)
            
            return {
                "task_id": task.id,
                "agent_type": "social",
                "social_concepts": social_concepts,
                "impact_analysis": impact_analysis,
                "adoption_analysis": adoption_analysis,
                "ethical_analysis": ethical_analysis,
                "cultural_analysis": cultural_analysis,
                "findings": [impact_analysis, adoption_analysis, ethical_analysis],
                "confidence": 0.75,
                "sources": ["social_research", "demographic_analysis"],
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "focus_areas": social_concepts
                }
            }
            
        except Exception as e:
            logger.error(f"Error in social research task: {e}")
            return {
                "task_id": task.id,
                "agent_type": "social",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _extract_social_concepts(self, description: str) -> List[str]:
        """Extract social and human-related concepts from task description"""
        prompt = f"""
        Extract social and human factors from: "{description}"
        
        Focus on:
        - Human behavior and psychology
        - Social trends and movements
        - Cultural implications
        - Ethical considerations
        - Demographic factors
        - Community impact
        
        Return as JSON array: ["concept1", "concept2", ...]
        Limit to 6 most relevant social concepts.
        """
        
        messages = [
            {"role": "system", "content": "You are a social concept analyst."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=200)
            return json.loads(response)
        except:
            return ["social_impact", "user_adoption", "ethical_implications", "cultural_factors"]
    
    async def _analyze_social_impact(self, description: str, concepts: List[str]) -> Dict:
        """Analyze social impact and implications"""
        prompt = f"""
        Analyze social impact for: "{description}"
        Social concepts: {', '.join(concepts)}
        
        Provide impact analysis as JSON:
        {{
            "positive_impacts": ["impact1", "impact2", "impact3"],
            "negative_impacts": ["impact1", "impact2"],
            "affected_groups": ["group1", "group2", "group3"],
            "impact_magnitude": "minimal|moderate|significant|transformative",
            "timeframe": "immediate|short_term|long_term",
            "societal_benefits": ["benefit1", "benefit2"],
            "potential_risks": ["risk1", "risk2"],
            "mitigation_strategies": ["strategy1", "strategy2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a social impact analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=500)
            return json.loads(response)
        except:
            return {
                "positive_impacts": ["improved_efficiency", "enhanced_accessibility"],
                "negative_impacts": ["job_displacement", "privacy_concerns"],
                "affected_groups": ["general_public", "professionals"],
                "impact_magnitude": "moderate",
                "timeframe": "medium_term",
                "societal_benefits": ["increased_productivity", "better_outcomes"],
                "potential_risks": ["inequality", "misuse"],
                "mitigation_strategies": ["regulation", "education"]
            }
    
    async def _research_adoption_patterns(self, concepts: List[str]) -> Dict:
        """Research adoption patterns and user behavior"""
        prompt = f"""
        Research adoption patterns for: {', '.join(concepts)}
        
        Analyze user adoption as JSON:
        {{
            "adoption_curve_stage": "innovators|early_adopters|early_majority|late_majority|laggards",
            "primary_adopters": ["group1", "group2"],
            "adoption_drivers": ["driver1", "driver2", "driver3"],
            "adoption_barriers": ["barrier1", "barrier2", "barrier3"],
            "usage_patterns": ["pattern1", "pattern2"],
            "user_satisfaction": "low|moderate|high",
            "retention_factors": ["factor1", "factor2"],
            "churn_reasons": ["reason1", "reason2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an adoption pattern analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=400)
            return json.loads(response)
        except:
            return {
                "adoption_curve_stage": "early_adopters",
                "primary_adopters": ["tech_enthusiasts", "professionals"],
                "adoption_drivers": ["utility", "efficiency", "novelty"],
                "adoption_barriers": ["complexity", "cost", "trust_issues"],
                "usage_patterns": ["regular_use", "specific_tasks"],
                "user_satisfaction": "moderate",
                "retention_factors": ["value_delivered", "ease_of_use"],
                "churn_reasons": ["limited_value", "better_alternatives"]
            }
    
    async def _identify_ethical_considerations(self, description: str) -> Dict:
        """Identify ethical considerations and concerns"""
        prompt = f"""
        Identify ethical considerations for: "{description}"
        
        Provide ethical analysis as JSON:
        {{
            "primary_ethical_issues": ["issue1", "issue2", "issue3"],
            "stakeholder_concerns": ["concern1", "concern2"],
            "privacy_implications": ["implication1", "implication2"],
            "fairness_issues": ["issue1", "issue2"],
            "transparency_needs": ["need1", "need2"],
            "regulatory_considerations": ["regulation1", "regulation2"],
            "best_practices": ["practice1", "practice2", "practice3"],
            "ethical_frameworks": ["framework1", "framework2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an ethics analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=400)
            return json.loads(response)
        except:
            return {
                "primary_ethical_issues": ["privacy", "fairness", "transparency"],
                "stakeholder_concerns": ["data_protection", "algorithmic_bias"],
                "privacy_implications": ["data_collection", "user_tracking"],
                "fairness_issues": ["equal_access", "bias_prevention"],
                "transparency_needs": ["explainable_decisions", "clear_policies"],
                "regulatory_considerations": ["compliance_requirements", "industry_standards"],
                "best_practices": ["informed_consent", "data_minimization", "regular_audits"],
                "ethical_frameworks": ["utilitarianism", "deontological_ethics"]
            }
    
    async def _analyze_cultural_factors(self, concepts: List[str]) -> Dict:
        """Analyze cultural and demographic factors"""
        prompt = f"""
        Analyze cultural factors for: {', '.join(concepts)}
        
        Provide cultural analysis as JSON:
        {{
            "cultural_variations": ["variation1", "variation2"],
            "demographic_preferences": ["preference1", "preference2"],
            "regional_differences": ["difference1", "difference2"],
            "generational_gaps": ["gap1", "gap2"],
            "accessibility_considerations": ["consideration1", "consideration2"],
            "localization_needs": ["need1", "need2"],
            "cultural_sensitivities": ["sensitivity1", "sensitivity2"],
            "inclusion_opportunities": ["opportunity1", "opportunity2"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a cultural analyst. Return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=400)
            return json.loads(response)
        except:
            return {
                "cultural_variations": ["urban_vs_rural", "developed_vs_developing"],
                "demographic_preferences": ["age_related", "education_level"],
                "regional_differences": ["geographic_preferences", "regulatory_differences"],
                "generational_gaps": ["digital_natives_vs_immigrants", "risk_tolerance"],
                "accessibility_considerations": ["disability_support", "language_barriers"],
                "localization_needs": ["language_translation", "cultural_adaptation"],
                "cultural_sensitivities": ["religious_considerations", "social_norms"],
                "inclusion_opportunities": ["underserved_markets", "accessibility_improvements"]
            }


# Export specialized agents
__all__ = [
    'AcademicSpecialist',
    'TechnicalSpecialist', 
    'BusinessSpecialist',
    'SocialSpecialist'
]