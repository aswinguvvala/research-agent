"""
Vector Knowledge System
Advanced semantic search and knowledge management using Chroma vector database
for the multi-agent research system.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
import hashlib
from dataclasses import dataclass, field

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

import openai
import numpy as np
from sentence_transformers import SentenceTransformer

from .multi_agent_research_system import ResearchTask

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge in the vector database"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    source: str = ""
    domain: str = "general"
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


class VectorKnowledgeSystem:
    """Advanced knowledge management with semantic search capabilities"""
    
    def __init__(self, openai_api_key: str, persist_directory: str = "./chroma_db"):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        self.persist_directory = persist_directory
        self.chroma_client = None
        self.collections = {}
        
        # Embedding model
        self.embedding_model = None
        self.embedding_function = None
        
        # Knowledge management
        self.knowledge_cache = {}
        self.query_cache = {}
        
        # System status
        self.is_initialized = False
        
        # Initialize the system
        asyncio.create_task(self._initialize_system())
    
    async def _initialize_system(self):
        """Initialize the vector database and embedding models"""
        try:
            if not CHROMA_AVAILABLE:
                logger.warning("Chroma not available. Installing fallback vector search...")
                self._setup_fallback_system()
                return
            
            # Initialize Chroma client
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Initialize collections for different domains
            await self._setup_collections()
            
            # Load existing knowledge if available
            await self._load_existing_knowledge()
            
            self.is_initialized = True
            logger.info("Vector knowledge system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector system: {e}")
            self._setup_fallback_system()
    
    def _setup_fallback_system(self):
        """Set up fallback system when Chroma is not available"""
        logger.info("Setting up fallback vector search system")
        
        try:
            # Use sentence-transformers for embedding if available
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.fallback_storage = {
                "documents": [],
                "embeddings": [],
                "metadata": []
            }
            self.is_initialized = True
            logger.info("Fallback system initialized")
        except:
            logger.warning("Full fallback to basic text search")
            self.knowledge_cache = {}
            self.is_initialized = True
    
    async def _setup_collections(self):
        """Set up collections for different research domains"""
        if not self.chroma_client:
            return
        
        domain_collections = [
            "academic_research",
            "technical_implementation", 
            "business_applications",
            "social_impact",
            "cross_domain_synthesis",
            "verified_facts"
        ]
        
        for domain in domain_collections:
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=domain,
                    embedding_function=self.embedding_function,
                    metadata={"description": f"Knowledge collection for {domain}"}
                )
                self.collections[domain] = collection
                logger.info(f"Collection '{domain}' ready")
            except Exception as e:
                logger.error(f"Failed to create collection {domain}: {e}")
    
    async def _load_existing_knowledge(self):
        """Load existing knowledge from collections"""
        if not self.chroma_client:
            return
        
        for domain, collection in self.collections.items():
            try:
                count = collection.count()
                if count > 0:
                    logger.info(f"Loaded {count} items from {domain} collection")
            except Exception as e:
                logger.error(f"Error loading from {domain}: {e}")
    
    async def add_knowledge(self, item: KnowledgeItem) -> bool:
        """Add a knowledge item to the vector database"""
        if not self.is_initialized:
            await asyncio.sleep(1)  # Wait for initialization
            if not self.is_initialized:
                return False
        
        try:
            # Choose appropriate collection
            collection_name = self._get_collection_for_domain(item.domain)
            
            if self.chroma_client and collection_name in self.collections:
                # Add to Chroma
                collection = self.collections[collection_name]
                collection.add(
                    documents=[item.content],
                    metadatas=[{
                        "source": item.source,
                        "domain": item.domain,
                        "confidence": item.confidence,
                        "timestamp": item.timestamp.isoformat(),
                        **item.metadata
                    }],
                    ids=[item.id]
                )
                
            elif hasattr(self, 'fallback_storage'):
                # Add to fallback system
                embedding = self.embedding_model.encode(item.content).tolist()
                self.fallback_storage["documents"].append(item.content)
                self.fallback_storage["embeddings"].append(embedding)
                self.fallback_storage["metadata"].append({
                    "id": item.id,
                    "source": item.source,
                    "domain": item.domain,
                    "confidence": item.confidence,
                    "timestamp": item.timestamp.isoformat(),
                    **item.metadata
                })
            else:
                # Basic cache storage
                self.knowledge_cache[item.id] = item
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge item: {e}")
            return False
    
    def _get_collection_for_domain(self, domain: str) -> str:
        """Get the appropriate collection name for a domain"""
        domain_mapping = {
            "academic": "academic_research",
            "technical": "technical_implementation",
            "business": "business_applications", 
            "social": "social_impact",
            "fact_checking": "verified_facts",
            "synthesis": "cross_domain_synthesis"
        }
        return domain_mapping.get(domain, "cross_domain_synthesis")
    
    async def semantic_search(self, query: str, domain: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Perform semantic search across knowledge base"""
        if not self.is_initialized:
            return []
        
        # Check cache first
        cache_key = f"{query}_{domain}_{limit}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        try:
            results = []
            
            if self.chroma_client:
                results = await self._chroma_search(query, domain, limit)
            elif hasattr(self, 'fallback_storage'):
                results = await self._fallback_search(query, domain, limit)
            else:
                results = await self._basic_search(query, domain, limit)
            
            # Cache results
            self.query_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _chroma_search(self, query: str, domain: Optional[str], limit: int) -> List[Dict]:
        """Search using Chroma vector database"""
        results = []
        
        # Determine which collections to search
        collections_to_search = []
        if domain:
            collection_name = self._get_collection_for_domain(domain)
            if collection_name in self.collections:
                collections_to_search.append((collection_name, self.collections[collection_name]))
        else:
            collections_to_search = list(self.collections.items())
        
        # Search each collection
        for collection_name, collection in collections_to_search:
            try:
                search_results = collection.query(
                    query_texts=[query],
                    n_results=min(limit, 5),  # Limit per collection
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Process results
                for i, document in enumerate(search_results['documents'][0]):
                    metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                    distance = search_results['distances'][0][i] if search_results['distances'] else 1.0
                    
                    results.append({
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": 1.0 - distance,  # Convert distance to similarity
                        "collection": collection_name,
                        "source": metadata.get("source", "unknown")
                    })
                    
            except Exception as e:
                logger.warning(f"Search failed for collection {collection_name}: {e}")
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _fallback_search(self, query: str, domain: Optional[str], limit: int) -> List[Dict]:
        """Search using fallback embedding system"""
        if not self.fallback_storage["documents"]:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.fallback_storage["embeddings"]):
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by domain if specified
            results = []
            for idx, similarity in similarities[:limit * 2]:  # Get more to allow filtering
                metadata = self.fallback_storage["metadata"][idx]
                
                if domain and metadata.get("domain") != domain:
                    continue
                
                results.append({
                    "content": self.fallback_storage["documents"][idx],
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "collection": "fallback",
                    "source": metadata.get("source", "unknown")
                })
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    async def _basic_search(self, query: str, domain: Optional[str], limit: int) -> List[Dict]:
        """Basic text search fallback"""
        results = []
        query_terms = query.lower().split()
        
        for item_id, item in self.knowledge_cache.items():
            if domain and item.domain != domain:
                continue
            
            # Simple keyword matching
            content_lower = item.content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            
            if matches > 0:
                results.append({
                    "content": item.content,
                    "metadata": item.metadata,
                    "similarity_score": matches / len(query_terms),
                    "collection": "cache",
                    "source": item.source
                })
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def find_related_knowledge(self, content: str, domain: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Find knowledge items related to the given content"""
        return await self.semantic_search(content, domain, limit)
    
    async def suggest_research_directions(self, current_findings: List[str]) -> List[str]:
        """Suggest new research directions based on current findings"""
        if not current_findings:
            return []
        
        try:
            # Combine findings for analysis
            findings_text = " ".join(current_findings[:5])
            
            # Search for related knowledge
            related_items = await self.semantic_search(findings_text, limit=10)
            
            # Extract unique research directions using LLM
            related_content = [item["content"] for item in related_items[:5]]
            
            prompt = f"""
            Based on current research findings and related knowledge, suggest new research directions:
            
            Current findings: {findings_text[:800]}
            
            Related knowledge:
            {chr(10).join([f"- {content[:200]}" for content in related_content])}
            
            Suggest 4-6 specific, actionable research directions as JSON array:
            ["direction1", "direction2", "direction3", "direction4"]
            
            Focus on unexplored angles, cross-domain opportunities, and knowledge gaps.
            """
            
            messages = [
                {"role": "system", "content": "You are a research direction strategist. Return JSON array only."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._llm_request(messages, max_tokens=300)
            directions = json.loads(response)
            
            return directions[:6]  # Limit to 6 suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest research directions: {e}")
            return [
                "Explore interdisciplinary connections",
                "Investigate practical applications",
                "Study long-term implications",
                "Research methodological improvements"
            ]
    
    async def identify_knowledge_gaps(self, domain: str, research_goal: str) -> List[Dict]:
        """Identify gaps in knowledge for a specific domain and goal"""
        try:
            # Search existing knowledge in the domain
            existing_knowledge = await self.semantic_search(research_goal, domain, limit=20)
            
            if not existing_knowledge:
                return [{
                    "gap_type": "foundational",
                    "description": f"Limited knowledge available for {research_goal} in {domain}",
                    "priority": "high",
                    "suggestions": ["Conduct foundational research", "Survey domain experts"]
                }]
            
            # Analyze knowledge coverage using LLM
            knowledge_summary = " ".join([item["content"][:100] for item in existing_knowledge[:10]])
            
            prompt = f"""
            Analyze knowledge gaps for research goal: "{research_goal}" in domain: "{domain}"
            
            Existing knowledge summary: {knowledge_summary}
            
            Identify specific knowledge gaps as JSON array:
            [
                {{
                    "gap_type": "methodological|theoretical|empirical|applied",
                    "description": "specific description of the gap",
                    "priority": "high|medium|low", 
                    "suggestions": ["suggestion1", "suggestion2"]
                }}
            ]
            
            Focus on significant gaps that would advance understanding.
            Limit to 4 most important gaps.
            """
            
            messages = [
                {"role": "system", "content": "You are a knowledge gap analyst. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._llm_request(messages, max_tokens=500)
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to identify knowledge gaps: {e}")
            return [{
                "gap_type": "analysis",
                "description": "Unable to analyze knowledge gaps automatically", 
                "priority": "medium",
                "suggestions": ["Manual gap analysis needed", "Expert consultation recommended"]
            }]
    
    async def create_knowledge_map(self, domain: str) -> Dict[str, Any]:
        """Create a conceptual map of knowledge in a domain"""
        try:
            # Get all knowledge items in the domain
            all_items = await self.semantic_search("", domain, limit=100)
            
            if not all_items:
                return {"domain": domain, "concepts": [], "connections": [], "coverage": "empty"}
            
            # Extract key concepts using LLM
            content_samples = [item["content"][:200] for item in all_items[:20]]
            
            prompt = f"""
            Create a knowledge map for domain: "{domain}"
            
            Sample content: {chr(10).join(content_samples)}
            
            Extract knowledge structure as JSON:
            {{
                "core_concepts": ["concept1", "concept2", "concept3"],
                "subtopics": ["subtopic1", "subtopic2", "subtopic3"],
                "methodologies": ["method1", "method2"],
                "key_relationships": [
                    {{"concept1": "concept2", "relationship": "supports|contradicts|extends"}}
                ],
                "coverage_assessment": "comprehensive|partial|sparse",
                "knowledge_clusters": ["cluster1", "cluster2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a knowledge mapping expert. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._llm_request(messages, max_tokens=600)
            knowledge_map = json.loads(response)
            
            # Add metadata
            knowledge_map["domain"] = domain
            knowledge_map["total_items"] = len(all_items)
            knowledge_map["last_updated"] = datetime.now().isoformat()
            
            return knowledge_map
            
        except Exception as e:
            logger.error(f"Failed to create knowledge map: {e}")
            return {
                "domain": domain,
                "core_concepts": [],
                "coverage_assessment": "unknown",
                "error": str(e)
            }
    
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
            logger.error(f"LLM request error: {e}")
            return ""
    
    async def batch_add_research_findings(self, findings: List[Dict], session_id: str) -> int:
        """Add multiple research findings to the knowledge base"""
        added_count = 0
        
        for finding in findings:
            try:
                # Create knowledge item
                item_id = f"{session_id}_{hashlib.md5(str(finding).encode()).hexdigest()[:8]}"
                
                knowledge_item = KnowledgeItem(
                    id=item_id,
                    content=str(finding.get("content", finding.get("finding", ""))),
                    metadata={
                        "session_id": session_id,
                        "agent_type": finding.get("agent_type", "unknown"),
                        "task_id": finding.get("task_id", ""),
                        "original_finding": finding
                    },
                    source=finding.get("source", "research_session"),
                    domain=finding.get("agent_type", "general"),
                    confidence=finding.get("confidence", 0.7),
                    timestamp=datetime.now()
                )
                
                # Add to knowledge base
                success = await self.add_knowledge(knowledge_item)
                if success:
                    added_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to add finding to knowledge base: {e}")
        
        logger.info(f"Added {added_count}/{len(findings)} findings to knowledge base")
        return added_count
    
    async def get_research_context(self, query: str, max_items: int = 10) -> List[Dict]:
        """Get relevant research context for a query"""
        # Search across all domains
        context_items = []
        
        # Search each domain
        domains = ["academic", "technical", "business", "social"]
        items_per_domain = max(1, max_items // len(domains))
        
        for domain in domains:
            domain_items = await self.semantic_search(query, domain, items_per_domain)
            context_items.extend(domain_items)
        
        # Add cross-domain synthesis if available
        synthesis_items = await self.semantic_search(query, "synthesis", items_per_domain)
        context_items.extend(synthesis_items)
        
        # Sort by relevance and remove duplicates
        seen_content = set()
        unique_items = []
        
        for item in sorted(context_items, key=lambda x: x["similarity_score"], reverse=True):
            content_hash = hashlib.md5(item["content"].encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_items.append(item)
        
        return unique_items[:max_items]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge system"""
        stats = {
            "is_initialized": self.is_initialized,
            "system_type": "unknown",
            "total_items": 0,
            "collections": {},
            "cache_size": len(self.query_cache)
        }
        
        try:
            if self.chroma_client:
                stats["system_type"] = "chroma_db"
                for name, collection in self.collections.items():
                    count = collection.count()
                    stats["collections"][name] = count
                    stats["total_items"] += count
                    
            elif hasattr(self, 'fallback_storage'):
                stats["system_type"] = "fallback_embeddings"
                stats["total_items"] = len(self.fallback_storage["documents"])
                
            else:
                stats["system_type"] = "basic_cache"
                stats["total_items"] = len(self.knowledge_cache)
                
        except Exception as e:
            stats["error"] = str(e)
        
        return stats


# Export the vector knowledge system
__all__ = [
    'VectorKnowledgeSystem',
    'KnowledgeItem'
]