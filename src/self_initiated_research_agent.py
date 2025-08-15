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
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retry_api_call(max_retries=3, delay=1.0, backoff=2.0):
    """
    Decorator for retrying API calls with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API call failed after {max_retries} attempts: {e}")
            
            # If all retries failed, raise the last exception
            raise last_exception
        return wrapper
    return decorator


class ResearchState(Enum):
    """Defines the current state of the research process"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXPLORING = "exploring"
    QUESTIONING = "questioning"
    PAUSED_FOR_USER = "paused_for_user"
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
        """Adds a research node to the knowledge graph with intelligent relationship detection."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)
        
        # Add edges for parent-child relationships
        for parent_id in node.parents:
            if parent_id in self.graph:
                self.graph.add_edge(parent_id, node.id)
        
        for child_id in node.children:
            if child_id in self.graph:
                self.graph.add_edge(node.id, child_id)
        
        # Auto-detect semantic relationships with existing nodes
        self._create_semantic_connections(node)
    
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
    
    def _create_semantic_connections(self, new_node: ResearchNode):
        """Create intelligent connections between semantically related nodes."""
        # Skip if this is the only node
        if len(self.nodes) <= 1:
            return
            
        new_content = new_node.content.lower()
        new_words = set(new_content.split())
        
        # Find semantically related existing nodes
        for existing_id, existing_node in self.nodes.items():
            if existing_id == new_node.id:
                continue
                
            existing_content = existing_node.content.lower()
            existing_words = set(existing_content.split())
            
            # Calculate semantic similarity (simple word overlap for now)
            common_words = new_words.intersection(existing_words)
            similarity = len(common_words) / max(len(new_words), len(existing_words))
            
            # Create connections based on node types and similarity
            should_connect = False
            
            # High similarity threshold (shared concepts)
            if similarity > 0.3:
                should_connect = True
            
            # Special rules for different node type combinations
            if new_node.node_type == 'fact' and existing_node.node_type == 'question':
                # Facts can answer questions
                if similarity > 0.2 or any(word in existing_content for word in ['what', 'how', 'why'] 
                                         if word in new_content):
                    should_connect = True
                    # Question -> Fact (question leads to answer)
                    if existing_id not in new_node.parents:
                        new_node.parents.append(existing_id)
                    if new_node.id not in existing_node.children:
                        existing_node.children.append(new_node.id)
                        
            elif new_node.node_type == 'question' and existing_node.node_type == 'fact':
                # Questions can be prompted by facts
                if similarity > 0.2:
                    should_connect = True
                    # Fact -> Question (fact prompts question)
                    if existing_id not in new_node.parents:
                        new_node.parents.append(existing_id)
                    if new_node.id not in existing_node.children:
                        existing_node.children.append(new_node.id)
            
            # Create bidirectional connections for related same-type nodes
            elif new_node.node_type == existing_node.node_type and similarity > 0.25:
                should_connect = True
            
            # Add edge to NetworkX graph if connection should be made
            if should_connect and not self.graph.has_edge(existing_id, new_node.id):
                self.graph.add_edge(existing_id, new_node.id)
                
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


@dataclass
class DomainContext:
    """
    Represents the detected domain and context for research.
    """
    main_entity: str  # Primary subject (e.g., "Tesla", "Machine Learning", "Climate Change")
    domain_type: str  # Type of research (e.g., "company", "technical", "scientific")
    focus_areas: List[str]  # Specific aspects being researched
    context_keywords: List[str]  # Key terms that helped with detection
    confidence: float  # Confidence in the detection (0.0-1.0)


class DomainDetector:
    """
    Intelligent domain detection engine that identifies entities, research types,
    and focus areas from research goals without hardcoding specific companies or topics.
    """
    
    def __init__(self):
        # Generic patterns for different domain types
        self.company_patterns = [
            # Company indicators
            'company', 'corporation', 'inc', 'ltd', 'llc', 'business', 'enterprise',
            'startup', 'firm', 'organization', 'brand', 'manufacturer', 'maker',
            # Business activities
            'sales', 'revenue', 'market share', 'competitors', 'strategy', 'products',
            'services', 'customers', 'business model', 'operations', 'leadership'
        ]
        
        self.technical_patterns = [
            'algorithm', 'technology', 'software', 'hardware', 'system', 'platform',
            'framework', 'architecture', 'implementation', 'development', 'programming',
            'engineering', 'technical', 'methodology', 'protocol', 'standard',
            'ai', 'machine learning', 'deep learning', 'neural network', 'model'
        ]
        
        self.scientific_patterns = [
            'research', 'study', 'analysis', 'theory', 'hypothesis', 'experiment',
            'discovery', 'findings', 'evidence', 'data', 'methodology', 'peer review',
            'publication', 'journal', 'scientific', 'academic', 'scholar'
        ]
        
        self.product_patterns = [
            'product', 'device', 'tool', 'solution', 'application', 'feature',
            'version', 'model', 'release', 'update', 'innovation', 'design'
        ]
    
    def detect_domain(self, research_goal: str) -> DomainContext:
        """
        Analyze research goal and detect domain, entity, and context.
        """
        goal_lower = research_goal.lower()
        words = goal_lower.split()
        
        # Extract potential entities (proper nouns, capitalized words)
        entities = self._extract_entities(research_goal)
        
        # Classify domain type
        domain_scores = {
            'company': self._calculate_pattern_score(goal_lower, self.company_patterns),
            'technical': self._calculate_pattern_score(goal_lower, self.technical_patterns),
            'scientific': self._calculate_pattern_score(goal_lower, self.scientific_patterns),
            'product': self._calculate_pattern_score(goal_lower, self.product_patterns)
        }
        
        # Determine primary domain
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])
        domain_type = primary_domain[0] if primary_domain[1] > 0.3 else 'general'
        confidence = primary_domain[1]
        
        # Identify focus areas based on context
        focus_areas = self._extract_focus_areas(goal_lower, domain_type)
        
        # Determine main entity
        main_entity = self._determine_main_entity(entities, research_goal, domain_type)
        
        # Extract context keywords
        context_keywords = self._extract_context_keywords(goal_lower, domain_type)
        
        return DomainContext(
            main_entity=main_entity,
            domain_type=domain_type,
            focus_areas=focus_areas,
            context_keywords=context_keywords,
            confidence=confidence
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entity names from text."""
        import re
        # Simple entity extraction - look for capitalized words/phrases
        entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
        # Filter out common words
        common_words = {'The', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
        return [entity for entity in entities if entity not in common_words]
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate how well text matches given patterns."""
        matches = sum(1 for pattern in patterns if pattern in text)
        return matches / len(patterns) if patterns else 0.0
    
    def _extract_focus_areas(self, text: str, domain_type: str) -> List[str]:
        """Extract specific focus areas based on domain type."""
        focus_areas = []
        
        if domain_type == 'company':
            business_areas = ['products', 'technology', 'market', 'strategy', 'operations', 
                            'innovation', 'competition', 'financial', 'leadership']
            focus_areas = [area for area in business_areas if area in text]
        
        elif domain_type == 'technical':
            tech_areas = ['architecture', 'implementation', 'performance', 'security',
                         'scalability', 'methodology', 'framework', 'tools']
            focus_areas = [area for area in tech_areas if area in text]
        
        elif domain_type == 'scientific':
            science_areas = ['methodology', 'results', 'applications', 'theory',
                           'implications', 'evidence', 'analysis']
            focus_areas = [area for area in science_areas if area in text]
        
        # If no specific areas found, use general analysis
        if not focus_areas:
            focus_areas = ['overview', 'analysis']
        
        return focus_areas
    
    def _determine_main_entity(self, entities: List[str], text: str, domain_type: str) -> str:
        """Determine the main entity/subject of research."""
        if entities:
            # Return the longest entity (likely most specific)
            return max(entities, key=len)
        
        # Extract key terms if no proper nouns found
        important_words = []
        words = text.split()
        for i, word in enumerate(words):
            if (word.lower() in self.technical_patterns or 
                word.lower() in self.scientific_patterns or
                len(word) > 6):  # Longer words likely more important
                important_words.append(word)
        
        if important_words:
            return ' '.join(important_words[:2])  # Take first 2 important words
        
        # Fallback to first few words
        return ' '.join(text.split()[:3])
    
    def _extract_context_keywords(self, text: str, domain_type: str) -> List[str]:
        """Extract key contextual terms."""
        words = text.split()
        keywords = []
        
        # Domain-specific keyword extraction
        if domain_type == 'company':
            relevant_patterns = self.company_patterns
        elif domain_type == 'technical':
            relevant_patterns = self.technical_patterns
        elif domain_type == 'scientific':
            relevant_patterns = self.scientific_patterns
        else:
            relevant_patterns = self.company_patterns + self.technical_patterns
        
        # Find matching patterns
        for word in words:
            if word.lower() in relevant_patterns or len(word) > 7:
                keywords.append(word.lower())
        
        return list(set(keywords))[:5]  # Return unique keywords, max 5


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
        # Initialize OpenAI client with new v1.0+ syntax
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        self.knowledge_graph = KnowledgeGraph()
        self.research_tools = ResearchTools(other_api_keys)
        self.domain_detector = DomainDetector()
        
        self.state = ResearchState.INITIALIZING
        self.research_plan = None
        self.research_history = []
        self.questions_for_user = []
        
        # Conversation memory
        self.conversation_history = []
        
        # Research session tracking
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.session_start = datetime.now()
        
        # Pause/resume functionality
        self.research_paused = False
        self.current_iteration = 0
        self.research_context = {}
        self.pending_user_questions = []
    
    def _safe_to_string(self, content, max_length=None):
        """Safely convert any content to string, handling dicts and other types"""
        try:
            if isinstance(content, dict):
                # Convert dict to readable string
                if 'content' in content:
                    content = str(content['content'])
                elif 'title' in content:
                    content = str(content['title'])
                elif 'summary' in content:
                    content = str(content['summary'])
                else:
                    content = str(content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Ensure content is string and truncate if needed
            content = str(content)
            if max_length:
                content = content[:max_length]
            return content
        except Exception as e:
            logger.warning(f"Error converting content to string: {e}")
            return f"[Content conversion error: {type(content).__name__}]"
    
    @retry_api_call(max_retries=3, delay=1.0, backoff=2.0)
    def _create_research_plan_api_call(self, plan_prompt: str):
        """Create research plan with retry logic."""
        return self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research strategist. Return only valid JSON."},
                {"role": "user", "content": plan_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
    
    @retry_api_call(max_retries=3, delay=1.0, backoff=2.0)
    def _generate_questions_api_call(self, prompt: str):
        """Generate research questions with retry logic."""
        return self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant generating focused questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
    
    @retry_api_call(max_retries=3, delay=1.0, backoff=2.0)
    def _interactive_dialogue_api_call(self, dialogue_prompt: str, conversation_history: list):
        """Interactive dialogue with retry logic."""
        return self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": dialogue_prompt},
                *conversation_history[-5:]
            ],
            temperature=0.7,
            max_tokens=350
        )
    
    @retry_api_call(max_retries=3, delay=1.0, backoff=2.0)
    def _analyze_feedback_api_call(self, feedback_prompt: str):
        """Analyze user feedback with retry logic."""
        return self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract research directions from user feedback. Return only valid JSON."},
                {"role": "user", "content": feedback_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
    
    @retry_api_call(max_retries=3, delay=1.0, backoff=2.0)
    def _synthesize_research_api_call(self, synthesis_prompt: str, is_technical: bool = False):
        """Synthesize research findings with retry logic."""
        system_message = ("You are an AI/ML research expert creating comprehensive technical explanations." 
                         if is_technical 
                         else "You are a research synthesizer creating comprehensive summaries.")
        
        return self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.7,
            max_tokens=600 if is_technical else 500
        )
        
    def initiate_research(self, goal: str) -> ResearchPlan:
        """
        Initiates autonomous research based on a high-level goal.
        This is the key differentiator - the agent creates its own research plan.
        """
        logger.info(f"Initiating research for goal: {goal}")
        self.state = ResearchState.PLANNING
        
        # Use dynamic domain detection instead of hardcoded keywords
        domain_context = self.domain_detector.detect_domain(goal)
        logger.info(f"Detected domain: {domain_context.domain_type}, entity: {domain_context.main_entity}, confidence: {domain_context.confidence:.2f}")
        
        # Store domain context for later use
        self.current_domain_context = domain_context
        
        # Generate dynamic research plan prompt based on detected domain
        plan_prompt = self._generate_domain_aware_prompt(goal, domain_context)
        
        try:
            response = self._create_research_plan_api_call(plan_prompt)
            plan_data = json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            plan_data = None
            
        if plan_data:
            # Validate and convert data types for plan components
            def ensure_string_list(items):
                """Convert any dictionaries or complex objects to strings"""
                if not isinstance(items, list):
                    return []
                string_items = []
                for item in items:
                    if isinstance(item, dict):
                        # Convert dict to readable string representation
                        if 'model_name' in item:
                            string_items.append(item['model_name'])
                        elif 'name' in item:
                            string_items.append(item['name'])
                        elif 'title' in item:
                            string_items.append(item['title'])
                        else:
                            # Fallback: use first string value or stringify the dict
                            str_values = [str(v) for v in item.values() if v and not isinstance(v, dict)]
                            string_items.append(str_values[0] if str_values else str(item))
                    elif isinstance(item, str):
                        string_items.append(item)
                    else:
                        string_items.append(str(item))
                return string_items
            
            self.research_plan = ResearchPlan(
                goal=goal,
                sub_goals=ensure_string_list(plan_data.get('sub_goals', [])),
                research_questions=ensure_string_list(plan_data.get('research_questions', [])),
                priority_topics=ensure_string_list(plan_data.get('priority_topics', [])),
                search_strategies=ensure_string_list(plan_data.get('search_strategies', [])),
                success_criteria=ensure_string_list(plan_data.get('success_criteria', [])),
                time_budget=300,  # 5 minutes default
                created_at=datetime.now()
            )
        else:
            # Generic domain-aware fallback plan
            self.research_plan = self._generate_domain_aware_fallback(goal, domain_context)
        
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
    
    def _generate_domain_aware_prompt(self, goal: str, domain_context: DomainContext) -> str:
        """
        Generate research plan prompt dynamically based on detected domain context.
        This replaces hardcoded company/topic-specific templates with intelligent adaptation.
        """
        entity = domain_context.main_entity
        domain_type = domain_context.domain_type
        focus_areas = domain_context.focus_areas
        
        if domain_type == 'company':
            return f"""
            You are a business research strategist specializing in company analysis. Create a comprehensive research plan for:
            
            Goal: {goal}
            
            This appears to be a company-focused query about {entity}. Structure your research around these business domains:
            
            **Key Business Areas to Investigate:**
            1. **Products & Services**: Core offerings, product lines, and service capabilities
            2. **Technology & Innovation**: Technical capabilities, R&D efforts, and technological advantages
            3. **Market Position**: Competitive landscape, market share, and strategic positioning
            4. **Business Strategy**: Corporate strategy, growth plans, and market expansion
            5. **Operations & Performance**: Business model, financial performance, and operational efficiency
            6. **Leadership & Culture**: Management team, company culture, and organizational structure
            
            Focus areas identified: {', '.join(focus_areas)}
            
            Generate a comprehensive research plan with:
            1. 5-7 specific sub-goals covering relevant business areas for {entity}
            2. 8-12 research questions about {entity}'s strategies, market position, and competitive advantages
            3. 6-8 priority topics including specific {entity} products, initiatives, and market segments
            4. 5-7 search strategies targeting business analysis and industry sources
            5. 5-7 success criteria measuring understanding of {entity}'s business and competitive position
            
            Return the plan as a JSON object with keys: sub_goals, research_questions, 
            priority_topics, search_strategies, success_criteria
            """
        
        elif domain_type == 'technical':
            return f"""
            You are a technical research strategist specializing in technology analysis. Create a comprehensive research plan for:
            
            Goal: {goal}
            
            This appears to be a technical query about {entity}. Structure your research around these technical domains:
            
            **Key Technical Areas to Investigate:**
            1. **Architecture & Design**: System architecture, design principles, and core components
            2. **Implementation & Development**: Technical implementation, development methodologies, and tools
            3. **Performance & Scalability**: Performance characteristics, scalability considerations, and benchmarks
            4. **Applications & Use Cases**: Practical applications, real-world implementations, and case studies
            5. **Comparison & Alternatives**: Competitive analysis, alternative approaches, and trade-offs
            6. **Evolution & Future**: Technical evolution, emerging trends, and future developments
            
            Focus areas identified: {', '.join(focus_areas)}
            
            Generate a comprehensive research plan with:
            1. 5-8 specific sub-goals covering technical aspects of {entity}
            2. 8-12 technical research questions about architecture, implementation, and applications
            3. 6-8 priority topics including specific technologies, methodologies, and implementations
            4. 5-7 search strategies using technical sources and research databases
            5. 5-7 success criteria measuring technical understanding and practical knowledge
            
            Return the plan as a JSON object with keys: sub_goals, research_questions, 
            priority_topics, search_strategies, success_criteria
            """
        
        elif domain_type == 'scientific':
            return f"""
            You are a scientific research strategist specializing in academic and research analysis. Create a comprehensive research plan for:
            
            Goal: {goal}
            
            This appears to be a scientific query about {entity}. Structure your research around these scientific domains:
            
            **Key Scientific Areas to Investigate:**
            1. **Current Understanding**: Established knowledge, scientific consensus, and foundational concepts
            2. **Recent Developments**: Latest research, breakthrough discoveries, and emerging findings
            3. **Methodology & Techniques**: Research methodologies, experimental approaches, and analytical techniques
            4. **Applications & Implications**: Practical applications, societal implications, and real-world impact
            5. **Controversies & Debates**: Scientific debates, conflicting evidence, and unresolved questions
            6. **Future Directions**: Research frontiers, emerging questions, and future investigation areas
            
            Focus areas identified: {', '.join(focus_areas)}
            
            Generate a comprehensive research plan with:
            1. 5-7 specific sub-goals covering scientific aspects of {entity}
            2. 8-12 research questions about current knowledge, recent developments, and implications
            3. 6-8 priority topics including key concepts, researchers, and research areas
            4. 5-7 search strategies targeting academic sources and scientific databases
            5. 5-7 success criteria measuring scientific understanding and research comprehension
            
            Return the plan as a JSON object with keys: sub_goals, research_questions, 
            priority_topics, search_strategies, success_criteria
            """
        
        else:  # general or low-confidence detection
            return f"""
            You are a research strategist. Create a comprehensive research plan for:
            
            Goal: {goal}
            
            Main subject identified: {entity}
            Context areas: {', '.join(focus_areas) if focus_areas else 'general analysis'}
            
            Generate a comprehensive research plan with:
            1. 4-6 specific sub-goals that break down the main research objective
            2. 6-8 research questions covering key aspects of {entity}
            3. 4-6 priority topics for investigation
            4. 4-6 search strategies appropriate for this topic
            5. 4-6 success criteria to measure research completion
            
            Return the plan as a JSON object with keys: sub_goals, research_questions, 
            priority_topics, search_strategies, success_criteria
            """
    
    def _generate_domain_aware_fallback(self, goal: str, domain_context: DomainContext) -> ResearchPlan:
        """
        Generate fallback research plan when API-generated plan fails, using domain-aware templates.
        """
        entity = domain_context.main_entity
        domain_type = domain_context.domain_type
        focus_areas = domain_context.focus_areas
        
        if domain_type == 'company':
            return ResearchPlan(
                goal=goal,
                sub_goals=[
                    f"Research {entity}'s products and services relevant to {goal}",
                    f"Analyze {entity}'s market position and competitive strategy for {goal}",
                    f"Investigate {entity}'s technology and innovation in {goal}",
                    f"Examine {entity}'s business strategy and operations around {goal}",
                    f"Study {entity}'s future plans and market outlook for {goal}"
                ],
                research_questions=[
                    f"How does {entity} approach {goal} differently from competitors?",
                    f"What specific {entity} products or services relate to {goal}?",
                    f"What is {entity}'s current market position regarding {goal}?",
                    f"How does {goal} fit into {entity}'s broader business strategy?",
                    f"What are {entity}'s future plans for {goal}?",
                    f"What challenges does {entity} face in {goal}?"
                ],
                priority_topics=[
                    f"{entity} {goal}",
                    f"{entity} strategy {goal}",
                    f"{entity} market {goal}",
                    f"{entity} technology {goal}",
                    f"{entity} competition {goal}"
                ],
                search_strategies=[
                    f"{entity} official sources and announcements",
                    f"Industry analysis and {entity} coverage",
                    f"Business news and {entity} market research",
                    f"{entity} investor relations and financial reports"
                ],
                success_criteria=[
                    f"Comprehensive understanding of {entity}'s approach to {goal}",
                    f"Clear analysis of {entity}'s competitive position in {goal}",
                    f"Detailed knowledge of {entity}'s strategy for {goal}"
                ],
                time_budget=300,
                created_at=datetime.now()
            )
        
        elif domain_type == 'technical':
            return ResearchPlan(
                goal=goal,
                sub_goals=[
                    f"Understand the technical architecture and design of {entity}",
                    f"Research implementation methods and development approaches for {entity}",
                    f"Find practical applications and use cases of {entity}",
                    f"Compare {entity} with alternative technical solutions",
                    f"Analyze performance characteristics and benchmarks of {entity}"
                ],
                research_questions=[
                    f"What is the core architecture of {entity}?",
                    f"How is {entity} implemented and developed?",
                    f"What are the main applications of {entity}?",
                    f"How does {entity} compare to similar technologies?",
                    f"What are the performance characteristics of {entity}?",
                    f"What are the limitations and challenges of {entity}?"
                ],
                priority_topics=[
                    f"{entity} architecture",
                    f"{entity} implementation",
                    f"{entity} applications",
                    f"{entity} performance",
                    f"{entity} comparison"
                ],
                search_strategies=[
                    f"Technical documentation for {entity}",
                    f"Academic papers and research on {entity}",
                    f"Implementation guides and tutorials for {entity}",
                    f"Performance benchmarks and comparisons"
                ],
                success_criteria=[
                    f"Technical understanding of {entity} architecture",
                    f"Knowledge of {entity} implementation approaches",
                    f"Awareness of {entity} practical applications"
                ],
                time_budget=300,
                created_at=datetime.now()
            )
        
        elif domain_type == 'scientific':
            return ResearchPlan(
                goal=goal,
                sub_goals=[
                    f"Understand current scientific knowledge about {entity}",
                    f"Research recent developments and findings in {entity}",
                    f"Analyze practical applications and implications of {entity}",
                    f"Examine scientific debates and controversies around {entity}",
                    f"Investigate future research directions in {entity}"
                ],
                research_questions=[
                    f"What is the current scientific consensus on {entity}?",
                    f"What are the latest research findings about {entity}?",
                    f"What are the practical applications of knowledge about {entity}?",
                    f"What scientific debates exist regarding {entity}?",
                    f"What are the future research directions for {entity}?",
                    f"What are the societal implications of {entity}?"
                ],
                priority_topics=[
                    f"{entity} research",
                    f"{entity} applications",
                    f"{entity} implications",
                    f"{entity} methodology",
                    f"{entity} future"
                ],
                search_strategies=[
                    f"Academic databases for {entity} research",
                    f"Scientific journals and publications on {entity}",
                    f"Research institutions studying {entity}",
                    f"Recent conferences and symposiums on {entity}"
                ],
                success_criteria=[
                    f"Scientific understanding of {entity}",
                    f"Knowledge of current research on {entity}",
                    f"Awareness of {entity} applications and implications"
                ],
                time_budget=300,
                created_at=datetime.now()
            )
        
        else:  # general fallback
            return ResearchPlan(
                goal=goal,
                sub_goals=[
                    f"Understand the basics and fundamentals of {entity}",
                    f"Research practical applications and use cases of {entity}",
                    f"Analyze the current state and recent developments in {entity}",
                    f"Investigate the broader context and implications of {entity}"
                ],
                research_questions=[
                    f"What is {entity} and why is it important?",
                    f"How is {entity} currently being used or applied?",
                    f"What are recent developments in {entity}?",
                    f"What are the broader implications of {entity}?"
                ],
                priority_topics=[entity, f"{entity} applications", f"{entity} development"],
                search_strategies=[
                    f"General search for {entity}",
                    f"Recent news and developments about {entity}",
                    f"Academic and professional sources on {entity}"
                ],
                success_criteria=[
                    f"Basic understanding of {entity}",
                    f"Knowledge of {entity} applications",
                    f"Awareness of recent developments in {entity}"
                ],
                time_budget=300,
                created_at=datetime.now()
            )
    
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
            
            # Find the source node that prompted this research
            source_node_id = None
            for node_id, node in self.knowledge_graph.nodes.items():
                if next_topic.lower() in node.content.lower():
                    source_node_id = node_id
                    break
            
            # 3. Add findings to knowledge graph with proper relationships
            for finding in findings:
                node = ResearchNode(
                    id=f"finding_{self.session_id}_{iteration}_{findings.index(finding)}",
                    content=self._safe_to_string(finding['content']),
                    node_type='fact',
                    source=self._safe_to_string(finding['source']),
                    confidence=finding.get('confidence', 0.7),
                    timestamp=datetime.now(),
                    metadata=finding.get('metadata', {}),
                    parents=[source_node_id] if source_node_id else []
                )
                self.knowledge_graph.add_node(node)
                results['discoveries'].append(finding)
            
            # 4. Identify gaps and generate questions
            gaps = self.knowledge_graph.find_knowledge_gaps()
            new_questions = self._generate_questions_from_gaps(gaps)
            
            for question in new_questions:
                # Link questions to the gaps that generated them
                source_gap_id = None
                if gaps:
                    gap_index = new_questions.index(question) % len(gaps)
                    source_gap_id = gaps[gap_index].id
                
                question_node = ResearchNode(
                    id=f"question_{self.session_id}_{iteration}_{new_questions.index(question)}",
                    content=self._safe_to_string(question),
                    node_type='question',
                    source='agent',
                    confidence=0.5,
                    timestamp=datetime.now(),
                    parents=[source_gap_id] if source_gap_id else []
                )
                self.knowledge_graph.add_node(question_node)
                results['questions_raised'].append(self._safe_to_string(question))
            
            # 5. Decide whether to ask user for clarification
            if iteration % 3 == 2:  # Every 3 iterations
                self.state = ResearchState.QUESTIONING
                user_questions = self._generate_user_questions()
                if user_questions:
                    self.questions_for_user.extend(user_questions)
                    self.pending_user_questions.extend(user_questions)
                    logger.info(f"Generated {len(user_questions)} questions for user")
                    
                    # Pause research for user input
                    self.state = ResearchState.PAUSED_FOR_USER
                    self.research_paused = True
                    self.current_iteration = iteration + 1
                    self.research_context = {
                        'results': results,
                        'max_iterations': max_iterations,
                        'iteration': iteration
                    }
                    logger.info("Research paused for user input")
                    return results  # Return current results and pause
            
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
    
    async def resume_research(self, additional_iterations: int = 3) -> Dict:
        """
        Resume research after user has provided input.
        This allows for true interactive research continuation.
        """
        if not self.research_paused or not self.research_context:
            logger.warning("Cannot resume research: not currently paused or no context available")
            return {"error": "Research not paused or no context available"}
        
        logger.info("Resuming research based on user input")
        self.state = ResearchState.EXPLORING
        self.research_paused = False
        
        # Get context from pause
        results = self.research_context.get('results', {
            'discoveries': [],
            'questions_raised': [],
            'gaps_identified': [],
            'synthesis': None
        })
        
        start_iteration = self.current_iteration
        max_iterations = start_iteration + additional_iterations
        
        logger.info(f"Resuming from iteration {start_iteration} for {additional_iterations} more iterations")
        
        for iteration in range(start_iteration, max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{max_iterations}")
            
            # 1. Identify what to research next (considering user input)
            next_topic = self._select_next_research_topic()
            
            if not next_topic:
                logger.info("No more topics to research")
                break
            
            # 2. Conduct research on the topic
            findings = await self._research_topic(next_topic)
            
            # Find the source node that prompted this research
            source_node_id = None
            for node_id, node in self.knowledge_graph.nodes.items():
                if next_topic.lower() in node.content.lower():
                    source_node_id = node_id
                    break
            
            # 3. Add findings to knowledge graph with proper relationships
            for finding in findings:
                node = ResearchNode(
                    id=f"finding_{self.session_id}_{iteration}_{findings.index(finding)}",
                    content=self._safe_to_string(finding['content']),
                    node_type='fact',
                    source=self._safe_to_string(finding['source']),
                    confidence=finding.get('confidence', 0.7),
                    timestamp=datetime.now(),
                    metadata=finding.get('metadata', {}),
                    parents=[source_node_id] if source_node_id else []
                )
                self.knowledge_graph.add_node(node)
                results['discoveries'].append(finding)
            
            # 4. Identify gaps and generate questions
            gaps = self.knowledge_graph.find_knowledge_gaps()
            new_questions = self._generate_questions_from_gaps(gaps)
            
            for question in new_questions:
                # Link questions to the gaps that generated them
                source_gap_id = None
                if gaps:
                    gap_index = new_questions.index(question) % len(gaps)
                    source_gap_id = gaps[gap_index].id
                
                question_node = ResearchNode(
                    id=f"question_{self.session_id}_{iteration}_{new_questions.index(question)}",
                    content=self._safe_to_string(question),
                    node_type='question',
                    source='agent',
                    confidence=0.5,
                    timestamp=datetime.now(),
                    parents=[source_gap_id] if source_gap_id else []
                )
                self.knowledge_graph.add_node(question_node)
                results['questions_raised'].append(self._safe_to_string(question))
            
            # 5. Check if should pause again for more user input
            if iteration % 3 == 2 and iteration < max_iterations - 1:  # Don't pause on last iteration
                user_questions = self._generate_user_questions()
                if user_questions:
                    self.questions_for_user.extend(user_questions)
                    self.pending_user_questions.extend(user_questions)
                    logger.info(f"Generated {len(user_questions)} questions for user")
                    
                    # Pause again for more user input
                    self.state = ResearchState.PAUSED_FOR_USER
                    self.research_paused = True
                    self.current_iteration = iteration + 1
                    self.research_context = {
                        'results': results,
                        'max_iterations': max_iterations,
                        'iteration': iteration
                    }
                    logger.info("Research paused again for additional user input")
                    return results
            
            # 6. Check if success criteria are met
            if self._check_completion():
                logger.info("Research objectives completed")
                break
        
        # 7. Synthesize findings
        self.state = ResearchState.SYNTHESIZING
        results['synthesis'] = self._synthesize_research()
        results['gaps_identified'] = [node.content for node in self.knowledge_graph.find_knowledge_gaps()]
        
        # Clear research context
        self.research_context = {}
        self.pending_user_questions = []
        
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
        Enhanced for technical AI/ML topics.
        """
        logger.info(f"Researching topic: {topic}")
        findings = []
        
        # Optimize search terms for technical topics
        search_terms = self._optimize_search_terms(topic)
        
        # Parallel research from multiple sources with optimized terms
        tasks = []
        for term in search_terms:
            tasks.extend([
                self._async_arxiv_search(term),
                self._async_wikipedia_search(term),
                self._async_scholar_search(term),
                self._async_web_search(term)
            ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Research task failed: {result}")
            elif result:
                findings.extend(result)
        
        # Remove duplicates based on content similarity
        findings = self._deduplicate_findings(findings)
        
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
    
    def _optimize_search_terms(self, topic: str) -> List[str]:
        """
        Optimizes search terms for technical AI/ML topics.
        """
        original_topic = topic.lower()
        search_terms = [topic]  # Always include original
        
        # DeepFace-specific optimizations
        if 'deepface' in original_topic:
            search_terms.extend([
                "DeepFace Facebook facial recognition",
                "DeepFace neural network architecture",
                "DeepFace face verification model",
                "Facebook DeepFace paper Taigman",
                "DeepFace deep learning face recognition"
            ])
        
        # DistilBERT-specific optimizations
        if 'distilbert' in original_topic or 'bert' in original_topic:
            search_terms.extend([
                "DistilBERT knowledge distillation",
                "DistilBERT BERT compression",
                "DistilBERT transformer model",
                "DistilBERT Hugging Face implementation",
                "knowledge distillation BERT neural networks"
            ])
        
        # General AI/ML term enhancements
        if any(keyword in original_topic for keyword in ['neural network', 'deep learning', 'machine learning']):
            search_terms.extend([
                f"{topic} implementation",
                f"{topic} architecture",
                f"{topic} training",
                f"{topic} paper"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        optimized_terms = []
        for term in search_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                optimized_terms.append(term)
        
        return optimized_terms[:3]  # Limit to top 3 to avoid too many requests
    
    def _deduplicate_findings(self, findings: List[Dict]) -> List[Dict]:
        """
        Remove duplicate findings based on content similarity.
        """
        if not findings:
            return findings
            
        unique_findings = []
        seen_content = set()
        
        for finding in findings:
            content = finding.get('content', '')
            # Simple deduplication based on first 100 characters
            content_key = content[:100].lower().strip()
            
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                unique_findings.append(finding)
        
        return unique_findings
    
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
                response = self._generate_questions_api_call(prompt)
                
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
        Enhanced for technical AI/ML topics with structured explanations.
        """
        logger.info("Synthesizing research findings")
        
        def safe_content_string(node, max_length=300):
            """Safely convert node content to string, handling dicts and other types"""
            try:
                content = node.content
                if isinstance(content, dict):
                    # Convert dict to readable string
                    if 'content' in content:
                        content = str(content['content'])
                    elif 'title' in content:
                        content = str(content['title'])
                    elif 'summary' in content:
                        content = str(content['summary'])
                    else:
                        content = str(content)
                elif not isinstance(content, str):
                    content = str(content)
                
                # Ensure content is string and truncate if needed
                content = str(content)[:max_length]
                return content
            except Exception as e:
                logger.warning(f"Error converting node content to string: {e}")
                return f"[Content conversion error: {type(node.content).__name__}]"
        
        # Gather all facts and key insights
        facts = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'fact']
        questions = [n for n in self.knowledge_graph.nodes.values() if n.node_type == 'question']
        
        # Use stored domain context for enhanced synthesis
        goal = self.research_plan.goal if self.research_plan else 'Unknown'
        domain_context = getattr(self, 'current_domain_context', None)
        
        # Fallback to domain detection if not available
        if not domain_context:
            domain_context = self.domain_detector.detect_domain(goal)
        
        domain_type = domain_context.domain_type
        entity = domain_context.main_entity
        
        # Create domain-aware fact summary
        fact_summary = "\n".join([f"- {safe_content_string(fact, 250)}" for fact in facts[:10]])
        question_summary = "\n".join([f"- {safe_content_string(q)}" for q in questions[:5]])
        
        # Generate domain-aware synthesis prompt
        synthesis_prompt = self._generate_domain_aware_synthesis_prompt(
            goal, entity, domain_type, fact_summary, question_summary
        )
        
        # Execute domain-aware synthesis
        try:
            is_technical = (domain_type == 'technical')  # For backwards compatibility with API call
            response = self._synthesize_research_api_call(synthesis_prompt, is_technical)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error synthesizing research: {e}")
            return self._generate_domain_aware_fallback_synthesis(goal, entity, domain_type, facts, questions)
    
    def _generate_domain_aware_synthesis_prompt(self, goal: str, entity: str, domain_type: str, 
                                               fact_summary: str, question_summary: str) -> str:
        """Generate synthesis prompt based on detected domain."""
        
        if domain_type == 'company':
            return f"""
            You are a business research analyst. Synthesize the following research findings about {entity}:
            
            Research Goal: {goal}
            
            Business Findings:
            {fact_summary}
            
            Open Questions:
            {question_summary}
            
            Create a business-focused synthesis that:
            1. **Company Overview**: Summarize {entity}'s position and approach to {goal}
            2. **Market Strategy**: Analyze {entity}'s strategy and competitive advantages
            3. **Business Impact**: Discuss implications for {entity}'s business and market
            4. **Key Insights**: Highlight the most important discoveries about {entity}
            5. **Strategic Outlook**: Assess future opportunities and challenges for {entity}
            
            Structure for business stakeholders interested in {entity}'s approach to {goal}.
            Target length: 350-400 words with actionable business insights.
            """
        
        elif domain_type == 'technical':
            return f"""
            You are a technical research expert. Synthesize the following research findings about {entity}:
            
            Research Goal: {goal}
            
            Technical Findings:
            {fact_summary}
            
            Open Questions:
            {question_summary}
            
            Create a technical synthesis that:
            1. **Technical Overview**: Explain the core architecture and design of {entity}
            2. **Implementation**: Describe key implementation details and methodologies
            3. **Applications**: Detail practical applications and use cases
            4. **Performance**: Highlight performance characteristics and benchmarks
            5. **Technical Insights**: Provide actionable technical knowledge
            
            Structure for technical professionals learning about {entity}.
            Target length: 400-450 words with clear technical explanations.
            """
        
        elif domain_type == 'scientific':
            return f"""
            You are a scientific research analyst. Synthesize the following research findings about {entity}:
            
            Research Goal: {goal}
            
            Scientific Findings:
            {fact_summary}
            
            Open Questions:
            {question_summary}
            
            Create a scientific synthesis that:
            1. **Current Understanding**: Summarize established knowledge about {entity}
            2. **Recent Developments**: Highlight latest research and discoveries
            3. **Applications**: Discuss practical applications and implications
            4. **Research Gaps**: Identify areas needing further investigation
            5. **Scientific Significance**: Assess the broader impact of findings
            
            Structure for researchers and academics interested in {entity}.
            Target length: 350-400 words with scientific rigor.
            """
        
        else:  # general domain
            return f"""
            You are a research synthesist. Synthesize the following research findings about {entity}:
            
            Research Goal: {goal}
            
            Key Findings:
            {fact_summary}
            
            Open Questions:
            {question_summary}
            
            Create a comprehensive synthesis that:
            1. **Overview**: Summarize main insights about {entity}
            2. **Key Patterns**: Identify important patterns and connections
            3. **Practical Implications**: Discuss real-world significance
            4. **Knowledge Gaps**: Highlight areas requiring further research
            5. **Conclusions**: Provide actionable insights and next steps
            
            Make the synthesis accessible and comprehensive.
            Target length: 300-350 words.
            """
    
    def _generate_domain_aware_fallback_synthesis(self, goal: str, entity: str, domain_type: str, 
                                                 facts: List, questions: List) -> str:
        """Generate fallback synthesis when API fails."""
        
        if domain_type == 'company':
            return f"""
            ## Business Research Summary: {entity} and {goal}
            
            ### Research Coverage
            - **Total Findings**: {len(facts)} business insights discovered
            - **Open Questions**: {len(questions)} areas for further investigation
            
            ### Key Business Insights
            Our research into {entity}'s approach to {goal} has revealed important strategic and operational insights. 
            The analysis covers {entity}'s market position, competitive strategy, and business implications related to {goal}.
            
            ### Strategic Assessment  
            Based on the research findings, {entity} demonstrates specific approaches and capabilities in {goal}. 
            The business implications and competitive positioning provide valuable insights for understanding 
            {entity}'s strategic direction and market opportunities.
            
            ### Research Gaps
            Several areas warrant additional investigation to complete the business analysis of {entity} and {goal}.
            These gaps represent opportunities for deeper strategic understanding.
            """
        
        else:  # technical, scientific, or general
            return f"""
            ## Research Summary: {entity}
            
            ### Research Coverage
            - **Total Findings**: {len(facts)} insights discovered across multiple sources
            - **Open Questions**: {len(questions)} areas requiring further investigation
            
            ### Key Insights
            Our comprehensive research into {entity} has yielded valuable insights related to {goal}. 
            The findings span multiple aspects including technical details, practical applications, 
            and broader implications.
            
            ### Analysis
            The research reveals important patterns and relationships that enhance our understanding 
            of {entity} in the context of {goal}. These insights provide a solid foundation for 
            further exploration and practical application.
            
            ### Next Steps
            Additional research in the identified gap areas would strengthen the overall understanding 
            and provide more comprehensive coverage of {entity} and its relationship to {goal}.
            """
    
    def get_unanswered_questions(self) -> List[ResearchNode]:
        """
        Returns research questions that haven't been adequately answered.
        This helps the agent know what to research next.
        """
        unanswered = [
            node for node in self.knowledge_graph.nodes.values()
            if node.node_type == 'question' and not node.children
        ]
        
        return unanswered[:5]  # Return up to 5 unanswered questions
    
    def interactive_dialogue(self, user_input: str) -> str:
        """
        Engages in dialogue with the user about the research.
        Enhanced to support research continuation and better integration.
        """
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Clear pending questions since user provided input
        if self.pending_user_questions:
            self.pending_user_questions = []
        
        # Build context from knowledge graph
        relevant_nodes = []
        for keyword in user_input.lower().split():
            for node in self.knowledge_graph.nodes.values():
                if keyword in node.content.lower():
                    relevant_nodes.append(node)
                    break
        
        context = "\n".join([f"- {node.content[:200]}" for node in relevant_nodes[:5]])
        
        # Check if research is paused and this response should trigger continuation
        research_status = ""
        if self.research_paused and self.state == ResearchState.PAUSED_FOR_USER:
            research_status = "\n\nIMPORTANT: Research is currently paused waiting for user input. Based on the user's response, you should acknowledge their input and indicate readiness to continue research with their guidance."
        
        # Enhanced dialogue prompt for research integration
        dialogue_prompt = f"""
        You are an intelligent research agent discussing your findings with the user.
        
        Current research context:
        {context}
        
        Research State: {self.state.value if self.state else 'unknown'}
        {research_status}
        
        User input: {user_input}
        
        Respond in a way that:
        1. Directly addresses the user's question or comment
        2. References specific findings from your research when relevant
        3. Admits uncertainty when you don't have enough information
        4. If research is paused, acknowledge their input and express readiness to continue research
        5. Suggests specific areas for deeper investigation based on their response
        6. Shows how their input will guide the next phase of research
        
        Keep the response conversational but research-focused. Show understanding of how their input improves the research direction.
        """
        
        try:
            response = self._interactive_dialogue_api_call(dialogue_prompt, self.conversation_history)
            
            agent_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            
            # If research was paused, update research plan based on user input
            if self.research_paused and self.research_plan:
                self._incorporate_user_feedback(user_input)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in dialogue: {e}")
            return "I apologize, but I'm having trouble formulating a response. Could you rephrase your question?"
    
    def _incorporate_user_feedback(self, user_input: str):
        """
        Incorporate user feedback into research planning for better continuation.
        """
        try:
            # Extract key topics and directions from user input
            feedback_analysis_prompt = f"""
            Analyze this user feedback about ongoing research and extract key directions:
            
            User feedback: {user_input}
            Current research goal: {self.research_plan.goal if self.research_plan else 'Unknown'}
            
            Extract:
            1. New topics or areas the user wants explored
            2. Specific aspects they want more depth on
            3. Areas they want less focus on
            4. Any new research questions implied by their response
            
            Return as JSON with keys: new_topics, focus_areas, avoid_areas, new_questions
            """
            
            response = self._analyze_feedback_api_call(feedback_analysis_prompt)
            
            feedback_data = json.loads(response.choices[0].message.content)
            
            # Update research plan with user feedback
            if self.research_plan:
                # Add new topics to priority list
                new_topics = feedback_data.get('new_topics', [])
                if new_topics:
                    self.research_plan.priority_topics.extend(new_topics)
                
                # Add new research questions
                new_questions = feedback_data.get('new_questions', [])
                if new_questions:
                    self.research_plan.research_questions.extend(new_questions)
                
                logger.info(f"Incorporated user feedback: {len(new_topics)} new topics, {len(new_questions)} new questions")
                
        except Exception as e:
            logger.error(f"Error incorporating user feedback: {e}")
            # Continue without feedback integration if it fails
    
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


