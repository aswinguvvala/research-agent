"""
Content Extractor for Research Agent
Extracts and processes content from PDFs, web pages, and research sources.
"""

import requests
from bs4 import BeautifulSoup
import PyPDF2
import re
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any
import io
from datetime import datetime
import json


class ContentExtractor:
    """Extracts content from various research sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Agent 1.0 (Academic Research Tool)'
        })
    
    def extract_pdf_content(self, pdf_url: str) -> Dict[str, Any]:
        """Extract text content and metadata from a PDF URL."""
        try:
            # Download PDF
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Read PDF content
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            # Extract metadata
            metadata = self._extract_pdf_metadata(pdf_reader, text_content)
            
            # Clean and structure the text
            cleaned_text = self._clean_text(text_content)
            sections = self._extract_sections(cleaned_text)
            
            return {
                "content": cleaned_text,
                "raw_content": text_content,
                "sections": sections,
                "metadata": metadata,
                "source_type": "pdf",
                "url": pdf_url,
                "extracted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to extract PDF content: {str(e)}",
                "url": pdf_url,
                "source_type": "pdf"
            }
    
    def extract_web_content(self, url: str) -> Dict[str, Any]:
        """Extract content from a web page."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                element.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_web_metadata(soup, url)
            
            # Clean text
            cleaned_content = self._clean_text(content)
            
            return {
                "content": cleaned_content,
                "raw_content": content,
                "metadata": metadata,
                "source_type": "webpage",
                "url": url,
                "extracted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to extract web content: {str(e)}",
                "url": url,
                "source_type": "webpage"
            }
    
    def extract_arxiv_content(self, arxiv_id: str) -> Dict[str, Any]:
        """Extract content from an ArXiv paper."""
        try:
            # Get ArXiv metadata - use HTTPS to avoid 301 redirects
            arxiv_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = self.session.get(arxiv_url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Parse ArXiv response
            metadata = self._parse_arxiv_metadata(response.text, arxiv_id)
            
            # Try to get PDF content
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_content = self.extract_pdf_content(pdf_url)
            
            if "error" not in pdf_content:
                # Merge ArXiv metadata with PDF content
                pdf_content["metadata"].update(metadata)
                return pdf_content
            else:
                # Return metadata only if PDF extraction failed
                return {
                    "content": metadata.get("abstract", ""),
                    "metadata": metadata,
                    "source_type": "arxiv",
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "extracted_at": datetime.now().isoformat(),
                    "note": "PDF extraction failed, metadata only"
                }
                
        except Exception as e:
            return {
                "error": f"Failed to extract ArXiv content: {str(e)}",
                "arxiv_id": arxiv_id,
                "source_type": "arxiv"
            }
    
    def _extract_pdf_metadata(self, pdf_reader: PyPDF2.PdfReader, text_content: str) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}
        
        # Try to get PDF metadata
        if pdf_reader.metadata:
            metadata.update({
                "title": pdf_reader.metadata.get("/Title", ""),
                "author": pdf_reader.metadata.get("/Author", ""),
                "subject": pdf_reader.metadata.get("/Subject", ""),
                "creator": pdf_reader.metadata.get("/Creator", ""),
                "producer": pdf_reader.metadata.get("/Producer", ""),
                "creation_date": pdf_reader.metadata.get("/CreationDate", "")
            })
        
        # Extract title from text if not in metadata
        if not metadata.get("title") and text_content:
            title = self._extract_title_from_text(text_content)
            if title:
                metadata["title"] = title
        
        # Extract authors from text
        authors = self._extract_authors_from_text(text_content)
        if authors:
            metadata["authors"] = authors
            
        # Extract abstract
        abstract = self._extract_abstract_from_text(text_content)
        if abstract:
            metadata["abstract"] = abstract
            
        # Extract DOI
        doi = self._extract_doi_from_text(text_content)
        if doi:
            metadata["doi"] = doi
            
        return metadata
    
    def _extract_web_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from web page."""
        metadata = {"url": url}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if name in ['author', 'authors']:
                metadata["author"] = content
            elif name in ['description', 'abstract']:
                metadata["description"] = content
            elif name == 'keywords':
                metadata["keywords"] = content
            elif name == 'date':
                metadata["date"] = content
        
        # Open Graph tags
        for meta in soup.find_all('meta', property=True):
            prop = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if prop == 'og:title':
                metadata["og_title"] = content
            elif prop == 'og:description':
                metadata["og_description"] = content
            elif prop == 'article:author':
                metadata["author"] = content
            elif prop == 'article:published_time':
                metadata["published_date"] = content
        
        # Schema.org data
        schema_data = self._extract_schema_data(soup)
        if schema_data:
            metadata["schema"] = schema_data
            
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from web page."""
        content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '#main'
        ]
        
        # Try to find main content area
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Extract text, preserving some structure
            for br in main_content.find_all('br'):
                br.replace_with('\n')
            for p in main_content.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                p.append('\n')
                
            return main_content.get_text()
        
        return soup.get_text()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\-.,;:!?()\'\""]', ' ', text)
        
        # Fix line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common paper sections from text."""
        sections = {}
        
        # Common section patterns
        section_patterns = {
            "abstract": r"abstract\s*:?\s*(.*?)(?=\n\s*(?:introduction|keywords|1\.|i\.|background))",
            "introduction": r"(?:introduction|1\.\s*introduction)\s*:?\s*(.*?)(?=\n\s*(?:related work|background|methodology|2\.))",
            "methodology": r"(?:methodology|methods?|approach)\s*:?\s*(.*?)(?=\n\s*(?:results|experiments?|evaluation|discussion))",
            "results": r"(?:results?|findings?)\s*:?\s*(.*?)(?=\n\s*(?:discussion|conclusion|related work))",
            "conclusion": r"(?:conclusion|conclusions?)\s*:?\s*(.*?)(?=\n\s*(?:references?|bibliography|acknowledgment))"
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()[:1000]  # Limit length
        
        return sections
    
    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from text content."""
        # Look for title at the beginning of the text
        lines = text.strip().split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.startswith(('http', 'www', 'doi:')):
                # Check if it looks like a title (no multiple sentences)
                if line.count('.') <= 1 and line.count('?') <= 1:
                    return line
        return None
    
    def _extract_authors_from_text(self, text: str) -> List[str]:
        """Extract author names from text."""
        authors = []
        
        # Look for author patterns
        author_patterns = [
            r"authors?:\s*([^\n]+)",
            r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*,\s*([A-Z][a-z]+\s+[A-Z][a-z]+))*"
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text[:1000], re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        authors.extend([m for m in match if m])
                    else:
                        authors.extend([name.strip() for name in match.split(',')])
                break
        
        # Clean and validate author names
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if len(author) > 3 and len(author) < 50 and ' ' in author:
                cleaned_authors.append(author)
        
        return cleaned_authors[:6]  # Limit to 6 authors
    
    def _extract_abstract_from_text(self, text: str) -> Optional[str]:
        """Extract abstract from text."""
        abstract_pattern = r"abstract\s*:?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.))"
        match = re.search(abstract_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:  # Must be substantial
                return abstract[:1000]  # Limit length
        return None
    
    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        doi_pattern = r"doi:\s*([^\s\n]+)|https?://doi\.org/([^\s\n]+)"
        match = re.search(doi_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1) or match.group(2)
        return None
    
    def _parse_arxiv_metadata(self, xml_content: str, arxiv_id: str) -> Dict[str, Any]:
        """Parse ArXiv API response for metadata."""
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            entry = soup.find('entry')
            
            if not entry:
                return {"arxiv_id": arxiv_id}
            
            metadata = {"arxiv_id": arxiv_id}
            
            # Title
            title = entry.find('title')
            if title:
                metadata["title"] = title.get_text().strip()
            
            # Authors
            authors = []
            for author in entry.find_all('author'):
                name = author.find('name')
                if name:
                    authors.append(name.get_text().strip())
            metadata["authors"] = authors
            
            # Abstract
            summary = entry.find('summary')
            if summary:
                metadata["abstract"] = summary.get_text().strip()
            
            # Published date
            published = entry.find('published')
            if published:
                metadata["published"] = published.get_text().strip()
            
            # Updated date
            updated = entry.find('updated')
            if updated:
                metadata["updated"] = updated.get_text().strip()
            
            # Categories
            categories = []
            for category in entry.find_all('category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            metadata["categories"] = categories
            
            # DOI
            doi = entry.find('arxiv:doi')
            if doi:
                metadata["doi"] = doi.get_text().strip()
            
            return metadata
            
        except Exception as e:
            return {"arxiv_id": arxiv_id, "parse_error": str(e)}
    
    def _extract_schema_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract Schema.org structured data."""
        schema_data = {}
        
        # JSON-LD data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    schema_data.update(data)
            except json.JSONDecodeError:
                continue
        
        return schema_data
    
    def extract_multiple_sources(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract content from multiple sources."""
        results = []
        
        for url in urls:
            try:
                if "arxiv.org" in url:
                    # Extract ArXiv ID and process
                    arxiv_id_match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url)
                    if arxiv_id_match:
                        result = self.extract_arxiv_content(arxiv_id_match.group(1))
                    else:
                        result = self.extract_web_content(url)
                elif url.endswith('.pdf'):
                    result = self.extract_pdf_content(url)
                else:
                    result = self.extract_web_content(url)
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "error": f"Failed to process {url}: {str(e)}",
                    "url": url
                })
        
        return results
    
    def summarize_content(self, content: str, max_sentences: int = 3) -> str:
        """Create a simple extractive summary of content."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        # Simple heuristic: take first sentence, longest sentences, and last sentence
        summary_sentences = [sentences[0]]  # First sentence
        
        if max_sentences > 2:
            # Add longest sentences
            sorted_sentences = sorted(sentences[1:-1], key=len, reverse=True)
            summary_sentences.extend(sorted_sentences[:max_sentences-2])
        
        if len(sentences) > 1:
            summary_sentences.append(sentences[-1])  # Last sentence
        
        return '. '.join(summary_sentences[:max_sentences]) + '.'


# Example usage and testing
if __name__ == "__main__":
    extractor = ContentExtractor()
    
    # Test ArXiv extraction
    print("Testing ArXiv extraction...")
    arxiv_result = extractor.extract_arxiv_content("2205.11916")
    if "error" not in arxiv_result:
        print(f"Title: {arxiv_result['metadata'].get('title', 'N/A')}")
        print(f"Authors: {arxiv_result['metadata'].get('authors', 'N/A')}")
        print(f"Abstract: {arxiv_result['metadata'].get('abstract', 'N/A')[:200]}...")
    else:
        print(f"Error: {arxiv_result['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test web extraction
    print("Testing web extraction...")
    web_result = extractor.extract_web_content("https://blog.openai.com/chatgpt/")
    if "error" not in web_result:
        print(f"Title: {web_result['metadata'].get('title', 'N/A')}")
        print(f"Content preview: {web_result['content'][:200]}...")
    else:
        print(f"Error: {web_result['error']}")