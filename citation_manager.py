"""
Citation Manager for Research Agent
Handles multiple citation formats (APA, MLA, IEEE) and bibliography generation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import re


class Source:
    """Represents a research source with metadata for citation generation."""
    
    def __init__(self, 
                 title: str,
                 authors: List[str] = None,
                 year: str = None,
                 url: str = None,
                 doi: str = None,
                 journal: str = None,
                 volume: str = None,
                 issue: str = None,
                 pages: str = None,
                 source_type: str = "article",
                 accessed_date: str = None):
        self.title = title
        self.authors = authors or []
        self.year = year or str(datetime.now().year)
        self.url = url
        self.doi = doi
        self.journal = journal
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.source_type = source_type  # article, book, website, arxiv, etc.
        self.accessed_date = accessed_date or datetime.now().strftime("%B %d, %Y")
        
    def __repr__(self):
        return f"Source(title='{self.title[:50]}...', authors={len(self.authors)}, year={self.year})"


class CitationManager:
    """Manages citations and bibliography generation in multiple formats."""
    
    def __init__(self):
        self.sources: List[Source] = []
        self.citation_counter = 1
        
    def add_source(self, source: Source) -> int:
        """Add a source and return its citation number."""
        self.sources.append(source)
        current_number = self.citation_counter
        self.citation_counter += 1
        return current_number
        
    def format_authors(self, authors: List[str], style: str = "apa", max_authors: int = None) -> str:
        """Format author names according to citation style."""
        if not authors:
            return "Unknown Author"
            
        if style.lower() == "apa":
            if len(authors) == 1:
                return self._format_author_apa(authors[0])
            elif len(authors) == 2:
                return f"{self._format_author_apa(authors[0])} & {self._format_author_apa(authors[1])}"
            elif len(authors) <= 20:
                formatted = [self._format_author_apa(author) for author in authors[:-1]]
                return ", ".join(formatted) + f", & {self._format_author_apa(authors[-1])}"
            else:
                return f"{self._format_author_apa(authors[0])} et al."
                
        elif style.lower() == "mla":
            if len(authors) == 1:
                return self._format_author_mla(authors[0])
            elif len(authors) == 2:
                return f"{self._format_author_mla(authors[0])} and {authors[1]}"
            else:
                return f"{self._format_author_mla(authors[0])} et al."
                
        elif style.lower() == "ieee":
            if len(authors) <= 6:
                formatted = [self._format_author_ieee(author) for author in authors]
                return ", ".join(formatted[:-1]) + f", and {formatted[-1]}" if len(formatted) > 1 else formatted[0]
            else:
                return f"{self._format_author_ieee(authors[0])} et al."
                
        return ", ".join(authors)
    
    def _format_author_apa(self, author: str) -> str:
        """Format author name for APA style (Last, F. M.)"""
        parts = author.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_initials = ". ".join([name[0] for name in parts[:-1]]) + "."
            return f"{last_name}, {first_initials}"
        return author
    
    def _format_author_mla(self, author: str) -> str:
        """Format author name for MLA style (Last, First Middle)"""
        parts = author.strip().split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return author
    
    def _format_author_ieee(self, author: str) -> str:
        """Format author name for IEEE style (F. M. Last)"""
        parts = author.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            initials = ". ".join([name[0] for name in parts[:-1]]) + "."
            return f"{initials} {last_name}"
        return author
    
    def format_citation(self, source: Source, style: str = "apa") -> str:
        """Generate a full citation for a source in the specified style."""
        style = style.lower()
        
        if style == "apa":
            return self._format_apa(source)
        elif style == "mla":
            return self._format_mla(source)
        elif style == "ieee":
            return self._format_ieee(source)
        else:
            return self._format_apa(source)  # Default to APA
    
    def _format_apa(self, source: Source) -> str:
        """Format citation in APA style."""
        citation_parts = []
        
        # Authors
        if source.authors:
            citation_parts.append(self.format_authors(source.authors, "apa"))
        else:
            citation_parts.append("Unknown Author")
            
        # Year
        citation_parts.append(f"({source.year})")
        
        # Title
        if source.source_type == "article":
            citation_parts.append(f"{source.title}.")
        else:
            citation_parts.append(f"*{source.title}*.")
            
        # Journal/Publication details
        if source.journal:
            journal_part = f"*{source.journal}*"
            if source.volume:
                journal_part += f", {source.volume}"
                if source.issue:
                    journal_part += f"({source.issue})"
            if source.pages:
                journal_part += f", {source.pages}"
            citation_parts.append(journal_part + ".")
            
        # DOI or URL
        if source.doi:
            citation_parts.append(f"https://doi.org/{source.doi}")
        elif source.url:
            citation_parts.append(source.url)
            
        return " ".join(citation_parts)
    
    def _format_mla(self, source: Source) -> str:
        """Format citation in MLA style."""
        citation_parts = []
        
        # Authors
        if source.authors:
            citation_parts.append(self.format_authors(source.authors, "mla") + ".")
        else:
            citation_parts.append("Unknown Author.")
            
        # Title
        if source.source_type == "article":
            citation_parts.append(f'"{source.title}."')
        else:
            citation_parts.append(f"*{source.title}*.")
            
        # Journal/Publication details
        if source.journal:
            journal_part = f"*{source.journal}*"
            if source.volume:
                journal_part += f", vol. {source.volume}"
                if source.issue:
                    journal_part += f", no. {source.issue}"
            if source.year:
                journal_part += f", {source.year}"
            if source.pages:
                journal_part += f", pp. {source.pages}"
            citation_parts.append(journal_part + ".")
        else:
            citation_parts.append(f"{source.year}.")
            
        # URL and access date
        if source.url:
            citation_parts.append(f"{source.url}. Accessed {source.accessed_date}.")
            
        return " ".join(citation_parts)
    
    def _format_ieee(self, source: Source) -> str:
        """Format citation in IEEE style."""
        citation_parts = []
        
        # Authors
        if source.authors:
            citation_parts.append(self.format_authors(source.authors, "ieee") + ",")
        else:
            citation_parts.append("Unknown Author,")
            
        # Title
        citation_parts.append(f'"{source.title},"')
        
        # Journal/Publication details
        if source.journal:
            journal_part = f"*{source.journal}*"
            if source.volume:
                journal_part += f", vol. {source.volume}"
                if source.issue:
                    journal_part += f", no. {source.issue}"
            if source.pages:
                journal_part += f", pp. {source.pages}"
            if source.year:
                journal_part += f", {source.year}"
            citation_parts.append(journal_part + ".")
        else:
            citation_parts.append(f"{source.year}.")
            
        # DOI or URL
        if source.doi:
            citation_parts.append(f"doi: {source.doi}")
        elif source.url:
            citation_parts.append(f"[Online]. Available: {source.url}")
            
        return " ".join(citation_parts)
    
    def generate_bibliography(self, style: str = "apa") -> str:
        """Generate a complete bibliography in the specified style."""
        if not self.sources:
            return "No sources to cite."
            
        bibliography = []
        bibliography.append(f"References ({style.upper()})")
        bibliography.append("=" * 50)
        bibliography.append("")
        
        # Sort sources alphabetically by first author's last name
        sorted_sources = sorted(self.sources, key=lambda s: s.authors[0].split()[-1] if s.authors else s.title)
        
        for i, source in enumerate(sorted_sources, 1):
            citation = self.format_citation(source, style)
            bibliography.append(f"{i}. {citation}")
            bibliography.append("")
            
        return "\n".join(bibliography)
    
    def get_in_text_citation(self, source_index: int, style: str = "apa", page: str = None) -> str:
        """Generate in-text citation for a source."""
        if source_index < 1 or source_index > len(self.sources):
            return "[Invalid citation]"
            
        source = self.sources[source_index - 1]
        style = style.lower()
        
        if style == "apa":
            if source.authors:
                first_author = source.authors[0].split()[-1]  # Last name
                if len(source.authors) == 1:
                    author_part = first_author
                elif len(source.authors) == 2:
                    second_author = source.authors[1].split()[-1]
                    author_part = f"{first_author} & {second_author}"
                else:
                    author_part = f"{first_author} et al."
            else:
                author_part = "Unknown Author"
                
            citation = f"({author_part}, {source.year}"
            if page:
                citation += f", p. {page}"
            citation += ")"
            return citation
            
        elif style == "mla":
            if source.authors:
                first_author = source.authors[0].split()[-1]
                citation = f"({first_author}"
                if page:
                    citation += f" {page}"
                citation += ")"
            else:
                citation = f'("{source.title[:20]}..."'
                if page:
                    citation += f" {page}"
                citation += ")"
            return citation
            
        elif style == "ieee":
            return f"[{source_index}]"
            
        return f"[{source_index}]"  # Default
    
    def clear_sources(self):
        """Clear all sources and reset counter."""
        self.sources.clear()
        self.citation_counter = 1
    
    def export_sources(self) -> List[Dict[str, Any]]:
        """Export sources as a list of dictionaries."""
        return [
            {
                "title": source.title,
                "authors": source.authors,
                "year": source.year,
                "url": source.url,
                "doi": source.doi,
                "journal": source.journal,
                "volume": source.volume,
                "issue": source.issue,
                "pages": source.pages,
                "source_type": source.source_type,
                "accessed_date": source.accessed_date
            }
            for source in self.sources
        ]


# Example usage and testing
if __name__ == "__main__":
    # Create citation manager
    cm = CitationManager()
    
    # Add sample sources
    source1 = Source(
        title="Large Language Models are Zero-Shot Reasoners",
        authors=["Takeshi Kojima", "Shixiang Shane Gu", "Machel Reid", "Yutaka Matsuo", "Yusuke Iwasawa"],
        year="2022",
        journal="Advances in Neural Information Processing Systems",
        volume="35",
        url="https://arxiv.org/abs/2205.11916",
        source_type="article"
    )
    
    source2 = Source(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year="2017",
        journal="Advances in Neural Information Processing Systems",
        volume="30",
        url="https://arxiv.org/abs/1706.03762",
        source_type="article"
    )
    
    # Add sources and get citation numbers
    cite1 = cm.add_source(source1)
    cite2 = cm.add_source(source2)
    
    # Test different citation styles
    print("APA Style:")
    print(cm.format_citation(source1, "apa"))
    print()
    
    print("MLA Style:")
    print(cm.format_citation(source1, "mla"))
    print()
    
    print("IEEE Style:")
    print(cm.format_citation(source1, "ieee"))
    print()
    
    # Test in-text citations
    print("In-text citations:")
    print(f"APA: {cm.get_in_text_citation(cite1, 'apa')}")
    print(f"MLA: {cm.get_in_text_citation(cite1, 'mla')}")
    print(f"IEEE: {cm.get_in_text_citation(cite1, 'ieee')}")
    print()
    
    # Generate bibliography
    print(cm.generate_bibliography("apa"))