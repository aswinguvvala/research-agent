"""
Self-Initiated Research Agent
An autonomous AI system that proactively conducts research, identifies knowledge gaps,
and asks clarifying questions to build comprehensive understanding.
"""

import os
import json
import time
import asyncio
import hashlib
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt

import openai
import arxiv
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scholarly import scholarly
import wikipedia
import feedparser
from newspaper import Article
import yfinance as yf
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchState(Enum):
    """Defines the current state of the research process"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXPLORING = "exploring"
    QUESTIONING = "questioning"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"


@dataclass
class ResearchNode:
    """
    Represents a node in the research knowledge graph.
    Each node is a piece of information or a question.
    """
    id: str
    content: str
    node_type: str  # 'fact', 'question', 'hypothesis', 'gap'
    source: str
    confidence: float
    timestamp: datetime
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ResearchPlan:
    """
    Represents the agent's research strategy.
    This is what makes the agent "self-initiated" - it creates its own plan.
    """
    goal: str
    sub_goals: List[str]
    research_questions: List[str]
    priority_topics: List[str]
    search_strategies: List[str]
    success_criteria: List[str]
    time_budget: int  # in seconds
    created_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            'goal': self.goal,
            'sub_goals': self.sub_goals,
            'research_questions': self.research_questions,
            'priority_topics': self.priority_topics,
            'search_strategies': self.search_strategies,
            'success_criteria': self.success_criteria,
            'time_budget': self.time_budget,
            'created_at': self.created_at.isoformat()
        }


class KnowledgeGraph:
    """
    Maintains a graph structure of discovered knowledge.
    This allows the agent to understand relationships between concepts
    and identify gaps in understanding.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        
    def add_node(self, node: ResearchNode):
        """Adds a research node to the knowledge graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)
        
        # Add edges for parent-child relationships
        for parent_id in node.parents:
            if parent_id in self.graph:
                self.graph.add_edge(parent_id, node.id)
        
        for child_id in node.children:
            if child_id in self.graph:
                self.graph.add_edge(node.id, child_id)
    
    def find_knowledge_gaps(self) -> List[ResearchNode]:
        """
        Identifies gaps in the knowledge graph where information is missing.
        These become new research questions.
        """
        gaps = []
        
        for node_id, node in self.nodes.items():
            # Low confidence nodes represent gaps
            if node.confidence < 0.5:
                gaps.append(node)
            
            # Nodes with questions but no answers
            if node.node_type == 'question' and not node.children:
                gaps.append(node)
            
            # Hypotheses without supporting evidence
            if node.node_type == 'hypothesis':
                supporting_evidence = [
                    self.nodes[child_id] for child_id in node.children 
                    if child_id in self.nodes and self.nodes[child_id].node_type == 'fact'
                ]
                if len(supporting_evidence) < 2:
                    gaps.append(node)
        
        return gaps
    
    def get_related_nodes(self, node_id: str, depth: int = 2) -> List[ResearchNode]:
        """Gets nodes related to a given node within specified depth."""
        if node_id not in self.graph:
            return []
        
        # Use BFS to find related nodes
        related_ids = set()
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for nid in current_level:
                # Add predecessors and successors
                next_level.update(self.graph.predecessors(nid))
                next_level.update(self.graph.successors(nid))
            related_ids.update(next_level)
            current_level = next_level
        
        return [self.nodes[nid] for nid in related_ids if nid in self.nodes]
    
    def visualize(self, filename: str = "knowledge_graph.png"):
        """Creates a visual representation of the knowledge graph."""
        plt.figure(figsize=(12, 8))
        
        # Color nodes by type
        color_map = {
            'fact': 'lightblue',
            'question': 'yellow',
            'hypothesis': 'lightgreen',
            'gap': 'red'
        }
        
        node_colors = [
            color_map.get(self.nodes[node].node_type, 'gray') 
            for node in self.graph.nodes()
        ]
        
        # Draw the graph
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True,
                node_size=1000, font_size=8, arrows=True)
        
        plt.title("Research Knowledge Graph")
        plt.savefig(filename)
        plt.close()


class ResearchTools:
    """
    Collection of tools the agent uses to gather information.
    This demonstrates integration with multiple data sources.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.cache = {}
        
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """Searches arXiv for academic papers."""
        cache_key = f"arxiv_{query}_{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                results.append({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'summary': paper.summary,
                    'published': paper.published,
                    'url': paper.entry_id,
                    'categories': paper.categories
                })
            
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    def search_wikipedia(self, query: str) -> Dict:
        """Searches Wikipedia for general knowledge."""
        cache_key = f"wiki_{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return {}
            
            # Get the most relevant page
            page = wikipedia.page(search_results[0])
            
            result = {
                'title': page.title,
                'summary': page.summary[:1000],  # First 1000 chars
                'url': page.url,
                'categories': page.categories[:10],
                'links': page.links[:20]
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return {}
    
    def scrape_web_page(self, url: str) -> Dict:
        """Scrapes content from a web page."""
        cache_key = f"scrape_{url}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            result = {
                'title': article.title,
                'authors': article.authors,
                'text': article.text,
                'summary': article.summary,
                'keywords': article.keywords,
                'publish_date': article.publish_date
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Web scraping error for {url}: {e}")
            return {}
    
    def search_google_scholar(self, query: str, num_results: int = 5) -> List[Dict]:
        """Searches Google Scholar for academic content."""
        cache_key = f"scholar_{query}_{num_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            search_query = scholarly.search_pubs(query)
            results = []
            
            for i, paper in enumerate(search_query):
                if i >= num_results:
                    break
                    
                results.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', []),
                    'abstract': paper.get('abstract', ''),
                    'year': paper.get('year', ''),
                    'citations': paper.get('num_citations', 0),
                    'url': paper.get('pub_url', '')
                })
            
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
            return []
    
    def analyze_trends(self, topic: str) -> Dict:
        """
        Analyzes trends related to a topic using multiple sources.
        This shows the agent can synthesize information from different perspectives.
        """
        trends = {
            'search_interest': self._get_search_trends(topic),
            'news_mentions': self._get_news_trends(topic),
            'academic_interest': self._get_academic_trends(topic),
            'social_sentiment': self._get_social_sentiment(topic)
        }
        
        return trends
    
    def _get_search_trends(self, topic: str) -> Dict:
        """Gets search trend data (would use Google Trends API in production)."""
        # Simplified mock implementation
        return {
            'trend': 'increasing',
            'peak_interest': '2024-Q3',
            'related_queries': [f"{topic} tutorial", f"{topic} research", f"{topic} applications"]
        }
    
    def _get_news_trends(self, topic: str) -> Dict:
        """Analyzes news coverage trends."""
        # In production, would use news API
        return {
            'coverage_volume': 'moderate',
            'sentiment': 'positive',
            'key_themes': ['innovation', 'challenges', 'future potential']
        }
    
    def _get_academic_trends(self, topic: str) -> Dict:
        """Analyzes academic research trends."""
        papers = self.search_arxiv(topic, max_results=10)
        
        if papers:
            recent_papers = [p for p in papers if p['published'].year >= 2023]
            return {
                'publication_rate': 'increasing' if len(recent_papers) > 5 else 'stable',
                'hot_subtopics': self._extract_hot_topics(papers),
                'leading_researchers': self._extract_top_authors(papers)
            }
        
        return {'publication_rate': 'unknown', 'hot_subtopics': [], 'leading_researchers': []}
    
    def _get_social_sentiment(self, topic: str) -> Dict:
        """Analyzes social media sentiment (mock implementation)."""
        return {
            'overall_sentiment': 'positive',
            'engagement_level': 'high',
            'discussion_themes': ['practical applications', 'ethical concerns', 'future impact']
        }
    
    def _extract_hot_topics(self, papers: List[Dict]) -> List[str]:
        """Extracts trending subtopics from paper summaries."""
        # Simple keyword extraction (in production, use NLP)
        keywords = []
        for paper in papers:
            if 'summary' in paper:
                # Extract important words (simplified)
                words = paper['summary'].lower().split()
                keywords.extend([w for w in words if len(w) > 7])
        
        # Return most common keywords
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(5)]
    
    def _extract_top_authors(self, papers: List[Dict]) -> List[str]:
        """Identifies leading researchers in the field."""
        authors = []
        for paper in papers:
            authors.extend(paper.get('authors', []))
        
        from collections import Counter
        return [author for author, _ in Counter(authors).most_common(3)]


class SelfInitiatedResearchAgent:
    """
    The main agent that autonomously conducts research.
    This is what makes it "self-initiated" - it doesn't just respond to queries,
    it actively explores, questions, and builds understanding.
    """
    
    def __init__(self, openai_api_key: str, other_api_keys: Dict[str, str] = None):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        self.knowledge_graph = KnowledgeGraph()
        self.research_tools = ResearchTools(other_api_keys)
        
        self.state = ResearchState.INITIALIZING
        self.research_plan = None
        self.research_history = []
        self.questions_for_user = []
        
        # Conversation memory
        self.conversation_history = []
        
        # Research session tracking
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.session_start = datetime.now()
        
    def initiate_research(self, goal: str) -> ResearchPlan:
        """
        Initiates autonomous research based on a high-level goal.
        This is the key differentiator - the agent creates its own research plan.
        """
        logger.info(f"Initiating research for goal: {goal}")
        self.state = ResearchState.PLANNING
        
        # Generate research plan using LLM
        plan_prompt = f"""
        You are a research strategist. Create a comprehensive research plan for the following goal:
        
        Goal: {goal}
        
        Generate a research plan with:
        1. 3-5 specific sub-goals that break down the main goal
        2. 5-7 research questions that need to be answered
        3. 3-5 priority topics to investigate
        4. 3-5 search strategies to employ
        5. 3-5 success criteria to measure completion
        
        Return the plan as a JSON object with keys: sub_goals, research_questions, 
        priority_topics, search_strategies, success_criteria
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research strategist. Return only valid JSON."},
                    {"role": "user", "content": plan_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            self.research_plan = ResearchPlan(
                goal=goal,
                sub_goals=plan_data.get('sub_goals', []),
                research_questions=plan_data.get('research_questions', []),
                priority_topics=plan_data.get('priority_topics', []),
                search_strategies=plan_data.get('search_strategies', []),
                success_criteria=plan_data.get('success_criteria', []),
                time_budget=300,  # 5 minutes default
                created_at=datetime.now()
            )
            
            # Add initial node to knowledge graph
            root_node = ResearchNode(
                id=f"root_{self.session_id}",
                content=goal,
                node_type='question',
                source='user',
                confidence=1.0,
                timestamp=datetime.now()
            )
            self.knowledge_graph.add_node(root_node)
            
            logger.info(f"Research plan created with {len(self.research_plan.sub_goals)} sub-goals")
            return self.research_plan
            
        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            # Fallback plan
            self.research_plan = ResearchPlan(
                goal=goal,
                sub_goals=[f"Understand basics of {goal}", f"Explore applications of {goal}"],
                research_questions=[f"What is {goal}?", f"Why is {goal} important?"],
                priority_topics=[goal],
                search_strategies=["academic search", "web search"],
                success_criteria=["Basic understanding achieved"],
                time_budget=300,
                created_at=datetime.now()
            )
            return self.research_plan
    
    async def execute_research(self, max_iterations: int = 10) -> Dict:
        """
        Executes the research plan autonomously.
        This is where the agent shows its autonomous behavior.
        """
        self.state = ResearchState.EXPLORING
        logger.info(f"Executing research plan with max {max_iterations} iterations")
        
        results = {
            'discoveries': [],
            'questions_raised': [],
            'gaps_identified': [],
            'synthesis': None
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{max_iterations}")
            
            # 1. Identify what to research next
            next_topic = self._select_next_research_topic()
            
            if not next_topic:
                logger.info("No more topics to research")
                break
            
            # 2. Conduct research on the topic
            findings = await self._research_topic(next_topic)
            
            # 3. Add findings to knowledge graph
            for finding in findings:
                node = ResearchNode(
                    id=f"finding_{self.session_id}_{iteration}_{findings.index(finding)}",
                    content=finding['content'],
                    node_type='fact',
                    source=finding['source'],
                    confidence=finding.get('confidence', 0.7),
                    timestamp=datetime.now(),
                    metadata=finding.get('metadata', {})
                )
                self.knowledge_graph.add_node(node)
                results['discoveries'].append(finding)
            
            # 4. Identify gaps and generate questions
            gaps = self.knowledge_graph.find_knowledge_gaps()
            new_questions = self._generate_questions_from_gaps(gaps)
            
            for question in new_questions:
                question_node = ResearchNode(
                    id=f"question_{self.session_id}_{iteration}_{new_questions.index(question)}",
                    content=question,
                    node_type='question',
                    source='agent',
                    confidence=0.5,
                    timestamp=datetime.now()
                )
                self.knowledge_graph.add_node(question_node)
                results['questions_raised'].append(question)
            
            # 5. Decide whether to ask user for clarification
            if iteration % 3 == 2:  # Every 3 iterations
                self.state = ResearchState.QUESTIONING
                user_questions = self._generate_user_questions()
                if user_questions:
                    self.questions_for_user.extend(user_questions)
                    logger.info(f"Generated {len(user_questions)} questions for user")
            
            # 6. Check if success criteria are met
            if self._check_completion():
                logger.info("Research objectives completed")
                break
        
        # 7. Synthesize findings
        self.state = ResearchState.SYNTHESIZING
        results['synthesis'] = self._synthesize_research()
        results['gaps_identified'] = [node.content for node in self.knowledge_graph.find_knowledge_gaps()]
        
        self.state = ResearchState.COMPLETE
        return results
    
    def _select_next_research_topic(self) -> Optional[str]:
        """
        Intelligently selects the next topic to research based on current knowledge.
        This shows the agent's ability to prioritize and strategize.
        """
        # Find unanswered questions
        unanswered = [
            node for node in self.knowledge_graph.nodes.values()
            if node.node_type == 'question' and not node.children
        ]
        
        if unanswered:
            # Prioritize by confidence (lower confidence = higher priority)
            unanswered.sort(key=lambda x: x.confidence)
            return unanswered[0].content
        
        # If no unanswered questions, explore priority topics
        if self.research_plan and self.research_plan.priority_topics:
            for topic in self.research_plan.priority_topics:
                # Check if we've already researched this topic
                existing = [
                    node for node in self.knowledge_graph.nodes.values()
                    if topic.lower() in node.content.lower()
                ]
                if len(existing) < 3:  # Arbitrary threshold
                    return topic
        
        return None
    
    async def _research_topic(self, topic: str) -> List[Dict]:
        """
        Conducts multi-source research on a topic.
        Uses async to parallelize searches for efficiency.
        """
        logger.info(f"Researching topic: {topic}")
        findings = []
        
        # Parallel research from multiple sources
        tasks = [
            self._async_arxiv_search(topic),
            self._async_wikipedia_search(topic),
            self._async_scholar_search(topic),
            self._async_web_search(topic)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Research task failed: {result}")
            elif result:
                findings.extend(result)
        
        # Analyze trends for deeper insights
        trends = self.research_tools.analyze_trends(topic)
        if trends:
            findings.append({
                'content': f"Trend analysis for {topic}: {json.dumps(trends)}",
                'source': 'trend_analysis',
                'confidence': 0.6,
                'metadata': trends
            })
        
        return findings
    
    async def _async_arxiv_search(self, topic: str) -> List[Dict]:
        """Async wrapper for arXiv search."""
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(None, self.research_tools.search_arxiv, topic, 3)
        
        findings = []
        for paper in papers:
            findings.append({
                'content': f"Research paper: {paper['title']}. Summary: {paper['summary'][:500]}",
                'source': f"arxiv:{paper['url']}",
                'confidence': 0.9,
                'metadata': paper
            })
        return findings
    
    async def _async_wikipedia_search(self, topic: str) -> List[Dict]:
        """Async wrapper for Wikipedia search."""
        loop = asyncio.get_event_loop()
        wiki_result = await loop.run_in_executor(None, self.research_tools.search_wikipedia, topic)
        
        if wiki_result:
            return [{
                'content': f"Wikipedia: {wiki_result['title']}. {wiki_result['summary']}",
                'source': f"wikipedia:{wiki_result['url']}",
                'confidence': 0.7,
                'metadata': wiki_result
            }]
        return []
    
    async def _async_scholar_search(self, topic: str) -> List[Dict]:
        """Async wrapper for Google Scholar search."""
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(None, self.research_tools.search_google_scholar, topic, 3)
        
        findings = []
        for paper in papers:
            findings.append({
                'content': f"Scholar: {paper['title']} ({paper.get('year', 'unknown')}). "
                          f"Citations: {paper.get('citations', 0)}. {paper.get('abstract', '')[:300]}",
                'source': f"scholar:{paper.get('url', '')}",
                'confidence': 0.8,
                'metadata': paper
            })
        return findings
    
    async def _async_web_search(self, topic: str) -> List[Dict]:
        """Async web search (mock implementation)."""
        # In production, would use a real web search API
        await asyncio.sleep(0.1)  # Simulate API call
        
        return [{
            'content': f"Web search result for {topic}: Latest developments and practical applications.",
            'source': 'web_search',
            'confidence': 0.6,
            'metadata': {'query': topic}
        }]
    
    def _generate_questions_from_gaps(self, gaps: List[ResearchNode]) -> List[str]:
        """
        Generates new research questions based on identified knowledge gaps.
        This is a key part of the autonomous behavior.
        """
        questions = []
        
        for gap in gaps[:3]:  # Limit to avoid too many questions
            # Use LLM to generate insightful questions
            prompt = f"""
            Based on this knowledge gap in our research:
            "{gap.content}"
            
            Generate 2 specific, research-worthy questions that would help fill this gap.
            Make the questions focused and answerable through research.
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a research assistant generating focused questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                question_text = response.choices[0].message.content
                # Parse questions (assuming they're on separate lines)
                new_questions = [q.strip() for q in question_text.split('\n') if q.strip() and '?' in q]
                questions.extend(new_questions[:2])
                
            except Exception as e:
                logger.error(f"Error generating questions: {e}")
                # Fallback question
                questions.append(f"What additional information would help understand {gap.content}?")
        
        return questions
    
    def _generate_user_questions(self) -> List[str]:
        """
        Generates questions to ask the user for clarification or direction.
        This is what makes the agent truly interactive and intelligent.
        """
        # Analyze current knowledge state
        total_nodes = len(self.knowledge_graph.nodes)
        question_nodes = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'question']
        fact_nodes = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'fact']
        
        questions = []
        
        # Ask for clarification on ambiguous areas
        if len(question_nodes) > len(fact_nodes):
            questions.append(
                "I've identified several open questions in this research area. "
                "Which aspect would you like me to prioritize: theoretical foundations, "
                "practical applications, or recent developments?"
            )
        
        # Ask for specific interests
        if self.research_plan and len(fact_nodes) > 5:
            topics = self.research_plan.priority_topics[:3]
            questions.append(
                f"Based on my research into {', '.join(topics)}, "
                f"are there specific subtopics or applications you're most interested in?"
            )
        
        # Ask for depth vs breadth preference
        if total_nodes > 10:
            questions.append(
                "Would you prefer me to go deeper into the current findings, "
                "or explore more broadly across related topics?"
            )
        
        # Ask about controversial or uncertain areas
        low_confidence = [n for n in self.knowledge_graph.nodes.values() if n.confidence < 0.5]
        if low_confidence:
            questions.append(
                "I've found some conflicting or uncertain information. "
                "Would you like me to investigate these discrepancies further?"
            )
        
        return questions
    
    def _check_completion(self) -> bool:
        """
        Checks if research objectives have been met.
        This allows the agent to know when to stop researching.
        """
        if not self.research_plan:
            return False
        
        # Check time budget
        elapsed = (datetime.now() - self.session_start).total_seconds()
        if elapsed > self.research_plan.time_budget:
            logger.info("Time budget exceeded")
            return True
        
        # Check if success criteria are met
        # This is simplified - in production, would use more sophisticated checking
        fact_nodes = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'fact']
        if len(fact_nodes) > 20:  # Arbitrary threshold
            logger.info("Sufficient facts gathered")
            return True
        
        # Check if all research questions have been addressed
        unanswered = [
            q for q in self.research_plan.research_questions
            if not any(q.lower() in n.content.lower() for n in fact_nodes)
        ]
        
        if not unanswered:
            logger.info("All research questions addressed")
            return True
        
        return False
    
    def _synthesize_research(self) -> str:
        """
        Synthesizes all research findings into a coherent summary.
        This demonstrates the agent's ability to not just gather but understand.
        """
        logger.info("Synthesizing research findings")
        
        # Gather all facts and key insights
        facts = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'fact']
        questions = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'question']
        
        # Build synthesis prompt
        fact_summary = "\n".join([f"- {fact.content[:200]}" for fact in facts[:10]])
        question_summary = "\n".join([f"- {q.content}" for q in questions[:5]])
        
        synthesis_prompt = f"""
        Synthesize the following research findings into a comprehensive summary:
        
        Research Goal: {self.research_plan.goal if self.research_plan else 'Unknown'}
        
        Key Findings:
        {fact_summary}
        
        Open Questions:
        {question_summary}
        
        Create a synthesis that:
        1. Summarizes the main insights discovered
        2. Identifies patterns and connections
        3. Highlights important gaps that remain
        4. Suggests next steps for further research
        5. Provides actionable conclusions
        
        Make the synthesis comprehensive but concise (around 300 words).
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research synthesizer creating comprehensive summaries."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error synthesizing research: {e}")
            # Fallback synthesis
            return f"""
            Research Summary for: {self.research_plan.goal if self.research_plan else 'Research Topic'}
            
            Findings: Discovered {len(facts)} key facts across multiple sources.
            Questions: Identified {len(questions)} areas requiring further investigation.
            
            The research reveals important insights but also highlights areas needing deeper exploration.
            Further investigation recommended for comprehensive understanding.
            """
    
    def interactive_dialogue(self, user_input: str) -> str:
        """
        Engages in dialogue with the user about the research.
        This is where the agent shows it can discuss and refine its understanding.
        """
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Build context from knowledge graph
        relevant_nodes = []
        for keyword in user_input.lower().split():
            for node in self.knowledge_graph.nodes.values():
                if keyword in node.content.lower():
                    relevant_nodes.append(node)
                    break
        
        context = "\n".join([f"- {node.content[:200]}" for node in relevant_nodes[:5]])
        
        # Generate response
        dialogue_prompt = f"""
        You are an intelligent research agent discussing your findings with the user.
        
        Current research context:
        {context}
        
        User input: {user_input}
        
        Respond in a way that:
        1. Directly addresses the user's question or comment
        2. References specific findings from your research when relevant
        3. Admits uncertainty when you don't have enough information
        4. Asks clarifying questions if the user's intent is unclear
        5. Suggests areas for further investigation if appropriate
        
        Keep the response conversational but informative.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": dialogue_prompt},
                    *self.conversation_history[-5:]
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            agent_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in dialogue: {e}")
            return "I apologize, but I'm having trouble formulating a response. Could you rephrase your question?"
    
    def generate_research_report(self) -> str:
        """
        Generates a detailed research report with all findings.
        This shows the agent can produce professional, actionable output.
        """
        facts = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'fact']
        questions = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'question']
        gaps = self.knowledge_graph.find_knowledge_gaps()
        
        report = f"""
# Research Report
**Session ID:** {self.session_id}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Research Goal:** {self.research_plan.goal if self.research_plan else 'Not specified'}

## Executive Summary
{self._synthesize_research()}

## Research Methodology
- **Sources Consulted:** Academic papers (arXiv, Google Scholar), Wikipedia, Web sources
- **Total Findings:** {len(facts)} facts discovered
- **Open Questions:** {len(questions)} questions identified
- **Knowledge Gaps:** {len(gaps)} gaps requiring further research

## Key Findings

"""
        
        # Group findings by source type
        arxiv_findings = [f for f in facts if 'arxiv' in f.source]
        scholar_findings = [f for f in facts if 'scholar' in f.source]
        web_findings = [f for f in facts if 'web' in f.source or 'wiki' in f.source]
        
        if arxiv_findings:
            report += "### Academic Research (arXiv)\n"
            for finding in arxiv_findings[:5]:
                report += f"- {finding.content[:200]}...\n"
            report += "\n"
        
        if scholar_findings:
            report += "### Scholarly Articles\n"
            for finding in scholar_findings[:5]:
                report += f"- {finding.content[:200]}...\n"
            report += "\n"
        
        if web_findings:
            report += "### General Knowledge\n"
            for finding in web_findings[:5]:
                report += f"- {finding.content[:200]}...\n"
            report += "\n"
        
        # Add research questions
        report += "## Research Questions Identified\n"
        for question in questions[:10]:
            report += f"- {question.content}\n"
        report += "\n"
        
        # Add gaps and next steps
        report += "## Knowledge Gaps and Next Steps\n"
        for gap in gaps[:5]:
            report += f"- **Gap:** {gap.content[:150]}\n"
        report += "\n"
        
        # Add recommendations
        report += """## Recommendations

Based on this research, I recommend:
1. Further investigation into the identified knowledge gaps
2. Consultation with domain experts on open questions
3. Practical experimentation to validate theoretical findings
4. Continuous monitoring of new developments in this area

## Appendix

### Knowledge Graph Statistics
"""
        
        # Add graph statistics
        report += f"- Total Nodes: {len(self.knowledge_graph.nodes)}\n"
        report += f"- Total Edges: {self.knowledge_graph.graph.number_of_edges()}\n"
        report += f"- Connected Components: {nx.number_weakly_connected_components(self.knowledge_graph.graph)}\n"
        
        return report
    
    def save_session(self, filepath: str):
        """Saves the research session for later analysis."""
        session_data = {
            'session_id': self.session_id,
            'research_plan': self.research_plan.to_dict() if self.research_plan else None,
            'knowledge_graph': {
                'nodes': {k: v.__dict__ for k, v in self.knowledge_graph.nodes.items()},
                'edges': list(self.knowledge_graph.graph.edges())
            },
            'conversation_history': self.conversation_history,
            'questions_for_user': self.questions_for_user,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session saved to {filepath}")


# Demo and example usage
async def demo_research_agent():
    """
    Demonstrates the Self-Initiated Research Agent capabilities.
    """
    print("\n" + "="*60)
    print("🤖 SELF-INITIATED RESEARCH AGENT DEMO")
    print("="*60)
    
    # Initialize agent (you'll need to provide your OpenAI API key)
    agent = SelfInitiatedResearchAgent(
        openai_api_key="your-openai-api-key-here"
    )
    
    # Example research goals
    research_goals = [
        "Understanding the latest advances in self-driving car technology",
        "Investigating the impact of artificial intelligence on healthcare",
        "Exploring quantum computing applications in cryptography"
    ]
    
    print("\nExample Research Goals:")
    for i, goal in enumerate(research_goals, 1):
        print(f"{i}. {goal}")
    
    choice = input("\nSelect a goal (1-3) or enter your own: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= 3:
        goal = research_goals[int(choice) - 1]
    else:
        goal = choice
    
    print(f"\n🎯 Research Goal: {goal}")
    print("\n📋 Creating research plan...")
    
    # Create research plan
    plan = agent.initiate_research(goal)
    
    print("\n📊 Research Plan Created:")
    print(f"  • Sub-goals: {len(plan.sub_goals)}")
    print(f"  • Research questions: {len(plan.research_questions)}")
    print(f"  • Priority topics: {', '.join(plan.priority_topics[:3])}")
    
    print("\n🔬 Beginning autonomous research...")
    print("(This will take a few moments as the agent explores multiple sources)\n")
    
    # Execute research
    results = await agent.execute_research(max_iterations=5)
    
    print("\n✅ Research Complete!")
    print(f"  • Discoveries made: {len(results['discoveries'])}")
    print(f"  • Questions raised: {len(results['questions_raised'])}")
    print(f"  • Gaps identified: {len(results['gaps_identified'])}")
    
    print("\n📝 Research Synthesis:")
    print("-" * 40)
    print(results['synthesis'])
    print("-" * 40)
    
    # Show questions for user
    if agent.questions_for_user:
        print("\n❓ The agent has questions for you:")
        for i, question in enumerate(agent.questions_for_user[:3], 1):
            print(f"{i}. {question}")
        
        print("\n💬 You can now interact with the agent about its findings.")
        print("Type 'quit' to exit, 'report' for full report, or ask questions.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'report':
                report = agent.generate_research_report()
                print("\n" + report)
            else:
                response = agent.interactive_dialogue(user_input)
                print(f"Agent: {response}\n")
    
    # Save session
    save_choice = input("\nWould you like to save this research session? (y/n): ")
    if save_choice.lower() == 'y':
        filename = f"research_session_{agent.session_id}.json"
        agent.save_session(filename)
        print(f"Session saved to {filename}")
        
        # Also save knowledge graph visualization
        agent.knowledge_graph.visualize(f"knowledge_graph_{agent.session_id}.png")
        print(f"Knowledge graph saved to knowledge_graph_{agent.session_id}.png")


if __name__ == "__main__":
    print("\n🤖 SELF-INITIATED RESEARCH AGENT")
    print("="*60)
    print("\nThis agent autonomously conducts research, identifies knowledge gaps,")
    print("and asks clarifying questions to build comprehensive understanding.\n")
    
    print("Key Features:")
    print("  ✓ Autonomous research planning and execution")
    print("  ✓ Multi-source information gathering (arXiv, Wikipedia, Scholar)")
    print("  ✓ Knowledge graph construction and gap analysis")
    print("  ✓ Self-initiated questioning and hypothesis generation")
    print("  ✓ Interactive dialogue about findings")
    print("  ✓ Professional research report generation\n")
    
    print("Requirements:")
    print("  • OpenAI API key for LLM capabilities")
    print("  • Python packages: openai, arxiv, wikipedia-api, scholarly, newspaper3k")
    print("  • Additional: networkx, matplotlib, beautifulsoup4, feedparser\n")
    
    print("To install dependencies:")
    print("pip install openai arxiv wikipedia-api scholarly newspaper3k")
    print("pip install networkx matplotlib beautifulsoup4 feedparser\n")
    
    try:
        # Run the demo
        asyncio.run(demo_research_agent())
    except KeyboardInterrupt:
        print("\n\n👋 Research agent terminated by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure all dependencies are installed and API keys are configured.")


