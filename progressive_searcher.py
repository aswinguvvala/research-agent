"""
Progressive Searcher for Research Agent
Implements iterative query refinement to find better, more relevant sources.
"""

import re
import asyncio
import aiohttp
import arxiv
import requests
import feedparser
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchStage:
    """Represents a stage in progressive search."""
    stage_name: str
    query_terms: List[str]
    search_strategy: str
    expected_sources: int
    priority: float


@dataclass
class SearchResult:
    """Enhanced search result with metadata."""
    source_data: Dict[str, Any]
    search_stage: str
    query_used: str
    relevance_hint: float  # Preliminary relevance estimate


class ProgressiveSearcher:
    """Implements progressive query refinement for better source discovery."""
    
    def __init__(self, max_sources_per_stage: int = 5, max_total_sources: int = 15):
        self.max_sources_per_stage = max_sources_per_stage
        self.max_total_sources = max_total_sources
        
        # Domain-specific search patterns
        self.domain_patterns = {
            'docker': {
                'core_terms': ['docker', 'container', 'containerization'],
                'related_terms': ['kubernetes', 'orchestration', 'microservices', 'devops'],
                'specific_terms': ['dockerfile', 'docker-compose', 'registry', 'image', 'volume'],
                'comparison_terms': ['virtualization', 'vm vs container', 'bare metal'],
                'best_practices': ['production', 'deployment', 'security', 'monitoring']
            },
            'deep_learning': {
                'core_terms': ['deep learning', 'neural networks', 'deep neural networks'],
                'related_terms': ['machine learning', 'artificial intelligence', 'backpropagation'],
                'specific_terms': ['CNN', 'RNN', 'LSTM', 'GRU', 'convolutional', 'recurrent', 'transformer'],
                'comparison_terms': ['CNN vs RNN', 'LSTM vs GRU', 'feedforward vs recurrent'],
                'best_practices': ['gradient descent', 'optimization', 'regularization', 'batch normalization'],
                'problems': ['vanishing gradient', 'exploding gradient', 'overfitting', 'underfitting'],
                'optimizers': ['Adam', 'SGD', 'RMSprop', 'AdaGrad', 'AdaDelta', 'optimizer', 'optimization algorithms'],
                'optimization_concepts': ['learning rate', 'momentum', 'weight decay', 'gradient clipping', 'adaptive learning'],
                'training_algorithms': ['gradient descent', 'stochastic gradient descent', 'mini-batch gradient descent', 'backpropagation', 'training algorithms'],
                'optimization_papers': ['optimization neural networks', 'training deep learning', 'gradient-based optimization', 'neural network optimization']
            },
            'machine_learning': {
                'core_terms': ['machine learning', 'ML', 'artificial intelligence'],
                'related_terms': ['deep learning', 'neural networks', 'data science'],
                'specific_terms': ['algorithm', 'model', 'training', 'prediction', 'classification'],
                'comparison_terms': ['supervised vs unsupervised', 'regression vs classification'],
                'best_practices': ['feature engineering', 'model validation', 'deployment']
            },
            'ai': {
                'core_terms': ['artificial intelligence', 'AI', 'cognitive computing'],
                'related_terms': ['machine learning', 'deep learning', 'natural language processing'],
                'specific_terms': ['reasoning', 'knowledge representation', 'expert systems'],
                'comparison_terms': ['narrow AI vs general AI', 'symbolic vs connectionist'],
                'best_practices': ['ethics', 'explainability', 'bias mitigation']
            },
            'medical': {
                'core_terms': ['medical', 'clinical', 'healthcare'],
                'related_terms': ['treatment', 'diagnosis', 'therapy', 'pharmaceutical'],
                'specific_terms': ['clinical trial', 'efficacy', 'adverse effects', 'dosage'],
                'comparison_terms': ['treatment vs placebo', 'drug vs therapy'],
                'best_practices': ['evidence based medicine', 'patient safety', 'quality care']
            },
            'web_development': {
                'core_terms': ['web development', 'web application', 'frontend', 'backend'],
                'related_terms': ['javascript', 'react', 'node.js', 'api', 'database'],
                'specific_terms': ['responsive design', 'single page application', 'REST', 'GraphQL'],
                'comparison_terms': ['frontend vs backend', 'spa vs mpa', 'sql vs nosql'],
                'best_practices': ['performance optimization', 'security', 'accessibility']
            },
            'software_development': {
                'core_terms': ['software development', 'software engineering', 'programming'],
                'related_terms': ['coding', 'development process', 'software design', 'architecture'],
                'specific_terms': ['agile', 'scrum', 'waterfall', 'devops', 'testing', 'debugging'],
                'comparison_terms': ['agile vs waterfall', 'monolith vs microservices'],
                'best_practices': ['code quality', 'version control', 'documentation', 'testing']
            },
            'sdlc': {
                'core_terms': ['software development life cycle', 'SDLC', 'development lifecycle'],
                'related_terms': ['software process', 'development methodology', 'project management'],
                'specific_terms': ['requirements analysis', 'design', 'implementation', 'testing', 'deployment', 'maintenance'],
                'comparison_terms': ['waterfall vs agile', 'spiral vs iterative', 'v-model vs prototype'],
                'best_practices': ['quality assurance', 'risk management', 'change management', 'documentation'],
                'phases': ['planning', 'analysis', 'design', 'coding', 'testing', 'deployment']
            },
            'agile_methodology': {
                'core_terms': ['agile methodology', 'agile development', 'scrum', 'kanban'],
                'related_terms': ['sprint', 'user story', 'backlog', 'retrospective', 'standup'],
                'specific_terms': ['sprint planning', 'product owner', 'scrum master', 'velocity'],
                'comparison_terms': ['scrum vs kanban', 'agile vs waterfall', 'agile vs lean'],
                'best_practices': ['continuous integration', 'iterative development', 'customer collaboration']
            }
        }
        
        # Technical acronym expansions for better search
        self.acronym_expansions = {
            # Machine Learning / AI
            'cnn': 'convolutional neural network',
            'cnns': 'convolutional neural networks',
            'rnn': 'recurrent neural network',
            'rnns': 'recurrent neural networks',
            'lstm': 'long short term memory',
            'gru': 'gated recurrent unit',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            # Software Development
            'sdlc': 'software development life cycle',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'qa': 'quality assurance',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'cicd': 'continuous integration continuous deployment',
            'agile': 'agile methodology',
            'scrum': 'scrum methodology',
            'devops': 'development operations',
            'tdd': 'test driven development',
            'bdd': 'behavior driven development'
        }
        
        # Search stage templates
        self.search_stages_template = [
            SearchStage("broad_overview", [], "conceptual", 3, 1.0),
            SearchStage("core_concepts", [], "foundational", 4, 1.2),
            SearchStage("specific_techniques", [], "detailed", 3, 1.1),
            SearchStage("comparisons", [], "comparative", 2, 0.9),
            SearchStage("best_practices", [], "practical", 3, 1.0)
        ]
    
    def _expand_query_terms(self, query: str) -> str:
        """Expand technical acronyms and fix common typos in query."""
        expanded_query = query.lower()
        
        # Fix common typos
        expanded_query = re.sub(r'\bawe\b', 'and why do we', expanded_query)
        expanded_query = re.sub(r'\bwhy dowe\b', 'why do we', expanded_query)
        
        # Expand acronyms
        words = expanded_query.split()
        expanded_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.acronym_expansions:
                expanded_words.append(self.acronym_expansions[clean_word])
                expanded_words.append(clean_word)  # Keep original too
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    async def progressive_search(self, original_query: str) -> List[SearchResult]:
        """
        Conduct progressive search with iterative refinement.
        
        Args:
            original_query: Original research query
            
        Returns:
            List of SearchResult objects with enhanced metadata
        """
        logger.info(f"Starting progressive search for: {original_query}")
        
        # Expand query terms for better matching
        expanded_query = self._expand_query_terms(original_query)
        logger.info(f"Expanded query: {expanded_query}")
        
        # Detect domain and generate search stages
        domain = self._detect_domain(expanded_query)
        search_stages = self._generate_search_stages(expanded_query, domain)
        
        all_results = []
        seen_titles = set()  # Avoid duplicates
        
        # Execute search stages
        for stage in search_stages:
            logger.info(f"Executing search stage: {stage.stage_name}")
            
            stage_results = await self._execute_search_stage(stage, expanded_query)
            
            # Filter duplicates and add to results
            for result in stage_results:
                title = result.source_data.get('title', '').lower()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_results.append(result)
            
            # Check if we have enough sources
            if len(all_results) >= self.max_total_sources:
                break
            
            # Brief pause between stages to avoid rate limiting
            await asyncio.sleep(1)
        
        # Sort by relevance hint and priority
        all_results.sort(key=lambda x: x.relevance_hint, reverse=True)
        
        logger.info(f"Progressive search completed: {len(all_results)} sources found")
        return all_results[:self.max_total_sources]
    
    def _detect_domain(self, query: str) -> Optional[str]:
        """Detect the domain/field of the research query."""
        query_lower = query.lower()
        
        # Special case for deep learning queries that mention specific architectures or problems
        deep_learning_indicators = [
            'cnn', 'rnn', 'lstm', 'gru', 'convolutional', 'recurrent',
            'vanishing gradient', 'exploding gradient', 'backpropagation',
            'neural network', 'deep learning', 'transformer', 'cnns', 'rnns'
        ]
        
        # Special case for optimization queries in neural networks
        optimization_indicators = [
            'optimizer', 'optimizers', 'optimization', 'adam', 'sgd', 'rmsprop',
            'gradient descent', 'learning rate', 'momentum'
        ]
        
        has_deep_learning = any(indicator in query_lower for indicator in deep_learning_indicators)
        has_optimization = any(indicator in query_lower for indicator in optimization_indicators)
        
        if has_deep_learning or (has_optimization and ('neural' in query_lower or 'network' in query_lower)):
            logger.info("Detected deep learning domain based on technical indicators")
            return 'deep_learning'
        
        # Score each domain based on keyword presence
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = 0
            
            # Check all term categories
            for term_category, terms in patterns.items():
                for term in terms:
                    if term.lower() in query_lower:
                        # Weight core terms and problems higher
                        if term_category == 'core_terms':
                            weight = 2.0
                        elif term_category == 'problems':
                            weight = 1.8  # Problems are highly specific
                        else:
                            weight = 1.0
                        score += weight
            
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            detected_domain = max(domain_scores, key=domain_scores.get)
            logger.info(f"Detected domain: {detected_domain} (score: {domain_scores[detected_domain]})")
            return detected_domain
        
        logger.info("No specific domain detected, using general search")
        return None
    
    def _generate_search_stages(self, query: str, domain: Optional[str]) -> List[SearchStage]:
        """Generate search stages based on query and detected domain."""
        stages = []
        query_words = self._extract_key_terms(query)
        
        for template in self.search_stages_template:
            stage = SearchStage(
                stage_name=template.stage_name,
                query_terms=[],
                search_strategy=template.search_strategy,
                expected_sources=template.expected_sources,
                priority=template.priority
            )
            
            # Generate query terms for this stage
            if template.stage_name == "broad_overview":
                stage.query_terms = self._generate_broad_terms(query, domain)
            elif template.stage_name == "core_concepts":
                stage.query_terms = self._generate_core_terms(query, domain)
            elif template.stage_name == "specific_techniques":
                stage.query_terms = self._generate_specific_terms(query, domain)
            elif template.stage_name == "comparisons":
                stage.query_terms = self._generate_comparison_terms(query, domain)
            elif template.stage_name == "best_practices":
                stage.query_terms = self._generate_practice_terms(query, domain)
            
            if stage.query_terms:  # Only add stages with valid terms
                stages.append(stage)
        
        return stages
    
    def _generate_broad_terms(self, query: str, domain: Optional[str]) -> List[str]:
        """Generate broad search terms for overview stage."""
        terms = []
        key_terms = self._extract_key_terms(query)
        
        # Use original key terms
        terms.extend(key_terms[:3])  # Top 3 key terms
        
        # Add domain-specific broad terms
        if domain and domain in self.domain_patterns:
            domain_data = self.domain_patterns[domain]
            terms.extend(domain_data.get('core_terms', [])[:2])
        
        # Generate conceptual variations
        conceptual_terms = []
        for term in key_terms[:2]:
            conceptual_terms.extend([
                f"{term} overview",
                f"{term} introduction",
                f"{term} fundamentals"
            ])
        
        terms.extend(conceptual_terms[:3])
        return terms[:5]  # Limit to 5 terms
    
    def _generate_core_terms(self, query: str, domain: Optional[str]) -> List[str]:
        """Generate core concept search terms."""
        terms = []
        key_terms = self._extract_key_terms(query)
        query_lower = query.lower()
        
        # Special handling for optimization queries
        if any(opt_term in query_lower for opt_term in ['optimizer', 'optimizers', 'optimization']):
            if domain == 'deep_learning':
                # Generate specific training algorithm combinations (more precise than "optimizer")
                architectures = ['CNN', 'RNN', 'LSTM', 'GRU', 'neural network', 'convolutional neural network', 'recurrent neural network']
                training_terms = ['Adam', 'SGD', 'stochastic gradient descent', 'RMSprop', 'gradient descent', 'backpropagation']
                
                # Prioritize specific algorithm names over generic "optimizer"
                for arch in architectures[:4]:
                    for training_term in training_terms[:4]:
                        terms.append(f"{arch} {training_term}")
                        terms.append(f"{training_term} {arch}")
                
                # Add academic paper terminology
                for arch in architectures[:3]:
                    terms.append(f"training {arch}")
                    terms.append(f"{arch} training algorithms")
                    terms.append(f"optimization {arch}")
                
                # Add direct combinations with key terms using training terminology
                for key_term in key_terms[:2]:
                    terms.append(f"{key_term} gradient descent")
                    terms.append(f"{key_term} Adam optimizer")
                    terms.append(f"{key_term} SGD")
                    terms.append(f"training {key_term}")
        
        # Combine key terms with domain concepts
        if domain and domain in self.domain_patterns:
            domain_data = self.domain_patterns[domain]
            related_terms = domain_data.get('related_terms', [])
            
            # Create combinations
            for key_term in key_terms[:2]:
                for related_term in related_terms[:3]:
                    terms.append(f"{key_term} {related_term}")
        
        # Add specific domain core terms
        if domain and domain in self.domain_patterns:
            terms.extend(self.domain_patterns[domain].get('core_terms', []))
            
            # Add optimizer-specific terms if this is a deep learning optimization query
            if domain == 'deep_learning' and any(opt_term in query_lower for opt_term in ['optimizer', 'optimizers']):
                terms.extend(self.domain_patterns[domain].get('optimizers', []))
        
        return terms[:8]  # Increased limit for optimization queries
    
    def _generate_specific_terms(self, query: str, domain: Optional[str]) -> List[str]:
        """Generate specific technique/implementation terms."""
        terms = []
        key_terms = self._extract_key_terms(query)
        query_lower = query.lower()
        
        if domain and domain in self.domain_patterns:
            domain_data = self.domain_patterns[domain]
            specific_terms = domain_data.get('specific_terms', [])
            
            # Special handling for optimization queries in deep learning
            if domain == 'deep_learning' and any(opt_term in query_lower for opt_term in ['optimizer', 'optimizers']):
                # Add specific training algorithm and concept combinations
                optimization_terms = domain_data.get('optimization_concepts', [])
                training_algorithms = domain_data.get('training_algorithms', [])
                
                # Combine architectures with optimization concepts
                for opt_concept in optimization_terms[:4]:
                    terms.append(f"CNN {opt_concept}")
                    terms.append(f"RNN {opt_concept}")
                
                # Add specific training algorithm terms
                for training_alg in training_algorithms[:3]:
                    terms.append(f"{training_alg} CNN")
                    terms.append(f"{training_alg} RNN")
                    terms.append(f"{training_alg} neural networks")
                
                # Add specific algorithm implementation terms
                specific_optimizers = ['Adam optimization', 'SGD training', 'RMSprop algorithm', 'gradient descent implementation']
                for optimizer in specific_optimizers:
                    terms.append(f"{optimizer} neural networks")
                    terms.append(f"{optimizer} deep learning")
            
            # Combine with key terms
            for key_term in key_terms[:2]:
                for specific_term in specific_terms[:3]:
                    terms.append(f"{key_term} {specific_term}")
            
            # Add implementation-focused terms
            for key_term in key_terms[:2]:
                terms.extend([
                    f"{key_term} implementation",
                    f"{key_term} tutorial",
                    f"{key_term} guide"
                ])
        
        return terms[:7]  # Increased for optimization queries
    
    def _generate_comparison_terms(self, query: str, domain: Optional[str]) -> List[str]:
        """Generate comparison and alternative terms."""
        terms = []
        key_terms = self._extract_key_terms(query)
        query_lower = query.lower()
        
        # Special handling for optimization comparisons
        if domain == 'deep_learning' and any(opt_term in query_lower for opt_term in ['optimizer', 'optimizers']):
            # Add specific optimizer comparisons
            optimizer_comparisons = [
                'Adam vs SGD',
                'RMSprop vs Adam',
                'SGD vs AdaGrad',
                'optimizer comparison neural networks',
                'best optimizer CNN',
                'best optimizer RNN'
            ]
            terms.extend(optimizer_comparisons)
        
        if domain and domain in self.domain_patterns:
            domain_data = self.domain_patterns[domain]
            comparison_terms = domain_data.get('comparison_terms', [])
            terms.extend(comparison_terms)
        
        # Generic comparison patterns
        for key_term in key_terms[:2]:
            terms.extend([
                f"{key_term} vs alternatives",
                f"{key_term} comparison",
                f"alternatives to {key_term}"
            ])
        
        return terms[:6]  # Increased for optimization queries
    
    def _generate_practice_terms(self, query: str, domain: Optional[str]) -> List[str]:
        """Generate best practice and practical application terms."""
        terms = []
        key_terms = self._extract_key_terms(query)
        
        if domain and domain in self.domain_patterns:
            domain_data = self.domain_patterns[domain]
            practice_terms = domain_data.get('best_practices', [])
            
            # Special handling for optimization best practices in deep learning
            if domain == 'deep_learning' and any(opt_term in query_lower for opt_term in ['optimizer', 'optimizers']):
                # Add specific training best practices
                training_best_practices = [
                    'learning rate scheduling',
                    'adaptive learning rates',
                    'momentum in gradient descent',
                    'weight initialization',
                    'gradient clipping techniques',
                    'optimizer comparison',
                    'training stability'
                ]
                
                for practice in training_best_practices[:4]:
                    terms.append(f"CNN {practice}")
                    terms.append(f"RNN {practice}")
                
                # Add optimizer-specific best practices
                for key_term in key_terms[:2]:
                    terms.extend([
                        f"{key_term} training best practices",
                        f"{key_term} optimization techniques",
                        f"{key_term} learning rate tuning"
                    ])
            
            # Combine with key terms
            for key_term in key_terms[:2]:
                for practice_term in practice_terms[:2]:
                    terms.append(f"{key_term} {practice_term}")
        
        # Generic best practice terms
        for key_term in key_terms[:2]:
            terms.extend([
                f"{key_term} best practices",
                f"{key_term} production",
                f"{key_term} enterprise"
            ])
        
        return terms[:7]  # Increased for optimization queries
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query, removing stop words."""
        # Clean and tokenize
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        words = clean_query.lower().split()
        
        # Remove stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'what',
            'when', 'where', 'why', 'with', 'about', 'help', 'research', 'study'
        }
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return most important terms (longer words first, then by frequency if needed)
        key_terms.sort(key=lambda x: (-len(x), x))
        return key_terms[:5]
    
    async def _execute_search_stage(self, stage: SearchStage, original_query: str) -> List[SearchResult]:
        """Execute a single search stage."""
        results = []
        
        # Search each query term in the stage
        for query_term in stage.query_terms:
            try:
                # Execute searches across different sources
                search_tasks = [
                    self._search_arxiv_progressive(query_term, stage),
                    self._search_web_progressive(query_term, stage)
                ]
                
                # Add PubMed for medical queries
                if 'medical' in query_term.lower() or 'clinical' in query_term.lower():
                    search_tasks.append(self._search_pubmed_progressive(query_term, stage))
                
                # Execute searches
                stage_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                # Collect results
                for result_set in stage_results:
                    if isinstance(result_set, list):
                        results.extend(result_set)
                    elif isinstance(result_set, Exception):
                        logger.warning(f"Search failed for '{query_term}': {result_set}")
                
                # Limit results per query term
                if len(results) >= stage.expected_sources:
                    break
                    
            except Exception as e:
                logger.error(f"Error in search stage {stage.stage_name} for term '{query_term}': {e}")
                continue
        
        # Calculate preliminary relevance scores
        for result in results:
            result.relevance_hint = self._calculate_preliminary_relevance(
                original_query, result.source_data, stage
            )
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance_hint, reverse=True)
        return results[:stage.expected_sources]
    
    async def _search_arxiv_progressive(self, query: str, stage: SearchStage) -> List[SearchResult]:
        """Search ArXiv with progressive query."""
        try:
            # Add rate limiting to respect ArXiv API guidelines
            await asyncio.sleep(1)  # 1 second delay between requests
            
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                # Clean and properly format author names
                authors = []
                for author in paper.authors:
                    author_str = str(author).strip()
                    # Remove any extra whitespace and ensure proper formatting
                    if author_str and len(author_str) > 1:
                        authors.append(author_str)
                
                # Clean the title - remove extra whitespace and line breaks
                title = re.sub(r'\s+', ' ', paper.title.strip())
                
                # Clean the abstract - remove extra whitespace and line breaks
                abstract = re.sub(r'\s+', ' ', paper.summary.strip()) if paper.summary else ""
                
                source_data = {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                    "source_type": "arxiv",
                    "arxiv_id": paper.entry_id.split('/')[-1],
                    "categories": [str(cat) for cat in paper.categories]
                }
                
                results.append(SearchResult(
                    source_data=source_data,
                    search_stage=stage.stage_name,
                    query_used=query,
                    relevance_hint=0.0  # Will be calculated later
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"ArXiv progressive search failed for '{query}': {e}")
            # Check if it's a redirect or connection issue
            if "301" in str(e) or "redirect" in str(e).lower():
                logger.info("ArXiv API returned redirect - this is expected during HTTPS migration")
            return []
    
    async def _search_pubmed_progressive(self, query: str, stage: SearchStage) -> List[SearchResult]:
        """Search PubMed with progressive query."""
        try:
            # Use E-utilities API for PubMed search
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": 3,
                "retmode": "json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    search_data = await response.json()
            
            # Extract PMIDs and get details (simplified)
            results = []
            if "esearchresult" in search_data and search_data["esearchresult"]["idlist"]:
                # For simplicity, create basic source data
                # In production, you'd fetch full details
                for i, pmid in enumerate(search_data["esearchresult"]["idlist"][:3]):
                    source_data = {
                        "title": f"PubMed Article {pmid}",
                        "authors": ["Unknown"],
                        "abstract": f"Medical research article from PubMed (PMID: {pmid})",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "source_type": "pubmed",
                        "pmid": pmid
                    }
                    
                    results.append(SearchResult(
                        source_data=source_data,
                        search_stage=stage.stage_name,
                        query_used=query,
                        relevance_hint=0.0
                    ))
            
            return results
            
        except Exception as e:
            logger.warning(f"PubMed progressive search failed for '{query}': {e}")
            return []
    
    async def _search_web_progressive(self, query: str, stage: SearchStage) -> List[SearchResult]:
        """Search web sources with progressive query."""
        try:
            # Search known research blogs and sources
            research_urls = [
                "https://blog.openai.com",
                "https://ai.googleblog.com",
                "https://research.facebook.com/blog"
            ]
            
            results = []
            
            # Simple approach: check if any recent posts match our query
            for blog_url in research_urls[:1]:  # Limit to avoid rate limiting
                try:
                    feed_urls = [f"{blog_url}/feed", f"{blog_url}/rss"]
                    
                    for feed_url in feed_urls:
                        try:
                            feed = feedparser.parse(feed_url)
                            if feed.entries:
                                # Check recent entries for relevance
                                query_terms = query.lower().split()
                                
                                for entry in feed.entries[:2]:
                                    title = entry.get('title', '').lower()
                                    summary = entry.get('summary', '').lower()
                                    
                                    # Simple relevance check
                                    relevance_score = sum(1 for term in query_terms 
                                                        if term in title or term in summary)
                                    
                                    if relevance_score > 0:
                                        source_data = {
                                            "title": entry.get('title', ''),
                                            "url": entry.get('link', ''),
                                            "summary": entry.get('summary', ''),
                                            "published": entry.get('published', ''),
                                            "source_type": "blog",
                                            "blog_url": blog_url
                                        }
                                        
                                        results.append(SearchResult(
                                            source_data=source_data,
                                            search_stage=stage.stage_name,
                                            query_used=query,
                                            relevance_hint=relevance_score
                                        ))
                                break  # Found working feed
                        except Exception:
                            continue
                            
                except Exception:
                    continue
            
            return results[:2]  # Limit web results
            
        except Exception as e:
            logger.warning(f"Web progressive search failed for '{query}': {e}")
            return []
    
    def _calculate_preliminary_relevance(self, 
                                         original_query: str, 
                                         source_data: Dict[str, Any], 
                                         stage: SearchStage) -> float:
        """Calculate preliminary relevance score for search results."""
        score = 0.0
        
        # Extract text for analysis
        title = source_data.get('title', '').lower()
        abstract = source_data.get('abstract', source_data.get('summary', '')).lower()
        
        query_terms = original_query.lower().split()
        
        # Title relevance (high weight)
        title_matches = sum(1 for term in query_terms if term in title)
        score += (title_matches / len(query_terms)) * 0.6
        
        # Abstract relevance
        abstract_matches = sum(1 for term in query_terms if term in abstract)
        score += (abstract_matches / len(query_terms)) * 0.3
        
        # Stage priority
        score *= stage.priority
        
        # Source type bonus
        source_type = source_data.get('source_type', '')
        if source_type == 'arxiv':
            score *= 1.2  # Academic papers get bonus
        elif source_type == 'pubmed':
            score *= 1.1  # Medical research gets bonus
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_search_summary(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate summary of progressive search results."""
        stage_counts = Counter(result.search_stage for result in results)
        source_types = Counter(result.source_data.get('source_type', 'unknown') for result in results)
        
        avg_relevance = sum(result.relevance_hint for result in results) / len(results) if results else 0.0
        
        return {
            'total_results': len(results),
            'stages_executed': list(stage_counts.keys()),
            'results_per_stage': dict(stage_counts),
            'source_types': dict(source_types),
            'average_relevance_hint': round(avg_relevance, 3),
            'max_relevance_hint': max((result.relevance_hint for result in results), default=0.0)
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_progressive_search():
        searcher = ProgressiveSearcher(max_sources_per_stage=3, max_total_sources=10)
        
        # Test queries
        test_queries = [
            "Docker containerization best practices",
            "Machine learning model interpretability",
            "COVID-19 treatment effectiveness"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing Progressive Search")
            print(f"Query: {query}")
            print("=" * 60)
            
            results = await searcher.progressive_search(query)
            summary = searcher.get_search_summary(results)
            
            print(f"Total Results: {summary['total_results']}")
            print(f"Stages Executed: {summary['stages_executed']}")
            print(f"Average Relevance: {summary['average_relevance_hint']}")
            
            print(f"\nTop Results:")
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. {result.source_data.get('title', 'Unknown Title')}")
                print(f"   Stage: {result.search_stage}")
                print(f"   Relevance: {result.relevance_hint:.3f}")
                print(f"   Type: {result.source_data.get('source_type', 'unknown')}")
    
    # Run test
    asyncio.run(test_progressive_search())