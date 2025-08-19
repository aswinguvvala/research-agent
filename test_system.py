"""
Enhanced Test Script for Autonomous Research Agent
Tests all components including the enhanced research agent with comprehensive diagnostics.
"""

import asyncio
import os
import sys
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test imports
def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    try:
        from citation_manager import CitationManager, Source
        from content_extractor import ContentExtractor
        from research_agent import ResearchAgent
        from enhanced_research_agent import EnhancedResearchAgent
        from cli import ResearchCLI
        print("✅ All core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are present and dependencies are installed.")
        return False

def test_dependencies():
    """Test optional ML dependencies."""
    print("🔍 Testing ML dependencies...")
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ ML dependencies available (enhanced features enabled)")
        return True
    except ImportError:
        print("⚠️ ML dependencies missing (enhanced features will be disabled)")
        print("Install with: pip install sentence-transformers scikit-learn")
        return False


def test_citation_manager():
    """Test the citation manager functionality."""
    print("\n🔬 Testing Citation Manager...")
    
    try:
        from citation_manager import CitationManager, Source
        cm = CitationManager()
        
        # Test adding a source
        source = Source(
            title="Test Paper on AI Research",
            authors=["John Doe", "Jane Smith"],
            year="2024",
            url="https://example.com/paper.pdf",
            journal="AI Research Journal",
            source_type="article"
        )
        
        citation_num = cm.add_source(source)
        assert citation_num == 1, "Citation numbering failed"
        
        # Test citation formatting
        apa_citation = cm.format_citation(source, "apa")
        mla_citation = cm.format_citation(source, "mla")
        ieee_citation = cm.format_citation(source, "ieee")
        
        assert "Doe" in apa_citation, "APA formatting failed"
        assert "Doe" in mla_citation, "MLA formatting failed"
        assert "Doe" in ieee_citation, "IEEE formatting failed"
        
        # Test bibliography generation
        bibliography = cm.generate_bibliography("apa")
        assert "References" in bibliography, "Bibliography generation failed"
        
        print("   ✅ Source addition")
        print("   ✅ APA formatting")
        print("   ✅ MLA formatting") 
        print("   ✅ IEEE formatting")
        print("   ✅ Bibliography generation")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Citation manager test failed: {e}")
        return False


def test_content_extractor():
    """Test the content extractor functionality."""
    print("\n📄 Testing Content Extractor...")
    
    try:
        from content_extractor import ContentExtractor
        extractor = ContentExtractor()
        
        # Test ArXiv extraction (using a real ArXiv ID)
        print("   Testing ArXiv extraction...")
        arxiv_result = extractor.extract_arxiv_content("2205.11916")
        
        if "error" not in arxiv_result:
            assert "title" in arxiv_result.get("metadata", {}), "ArXiv title extraction failed"
            print("   ✅ ArXiv extraction")
        else:
            print(f"   ⚠️  ArXiv extraction failed: {arxiv_result.get('error', 'Unknown error')}")
        
        # Test web content extraction (using a simple webpage)
        print("   Testing web extraction...")
        try:
            web_result = extractor.extract_web_content("https://example.com")
            if "error" not in web_result:
                print("   ✅ Web extraction")
            else:
                print(f"   ⚠️  Web extraction failed: {web_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ⚠️  Web extraction test skipped: {e}")
        
        # Test text cleaning
        dirty_text = "  This    is   a\n\ntest    text with   weird   spacing  "
        clean_text = extractor._clean_text(dirty_text)
        assert "test text" in clean_text, "Text cleaning failed"
        print("   ✅ Text cleaning")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Content extractor test failed: {e}")
        return False


async def test_research_agent():
    """Test the basic research agent functionality."""
    print("\n🤖 Testing Basic Research Agent...")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️  Skipping research agent test - no OPENAI_API_KEY set")
        return False
    
    try:
        from research_agent import ResearchAgent
        agent = ResearchAgent(api_key, max_sources=3)  # Limit sources for testing
        
        # Test basic initialization
        assert agent.openai_api_key == api_key, "API key not set correctly"
        assert agent.max_sources == 3, "Max sources not set correctly"
        print("   ✅ Agent initialization")
        
        # Test ArXiv search
        print("   Testing ArXiv search...")
        arxiv_sources = await agent._search_arxiv("machine learning")
        if arxiv_sources:
            print(f"   ✅ ArXiv search ({len(arxiv_sources)} results)")
        else:
            print("   ⚠️  ArXiv search returned no results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic research agent test failed: {e}")
        logger.error(f"Basic agent error: {traceback.format_exc()}")
        return False


async def test_enhanced_research_agent():
    """Test the enhanced research agent functionality."""
    print("\n🧠 Testing Enhanced Research Agent...")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️  Skipping enhanced research agent test - no OPENAI_API_KEY set")
        print("   Set OPENAI_API_KEY environment variable to test the enhanced agent")
        return False
    
    try:
        from enhanced_research_agent import EnhancedResearchAgent
        
        # Test initialization
        print("   Testing enhanced agent initialization...")
        agent = EnhancedResearchAgent(api_key, debug_mode=True, max_sources=5)
        print("   ✅ Enhanced agent initialized successfully")
        
        # Test simple search functionality
        print("   Testing progressive search...")
        search_results = await agent.progressive_searcher.progressive_search("machine learning")
        if search_results:
            print(f"   ✅ Progressive search returned {len(search_results)} results")
        else:
            print("   ⚠️  Progressive search returned no results (possible connectivity issue)")
            return False
        
        # Test source validation
        print("   Testing source validation...")
        raw_sources = [result.source_data for result in search_results]
        validated_sources = agent.relevance_validator.batch_validate_sources("machine learning", raw_sources)
        print(f"   ✅ Source validation completed: {len(validated_sources)}/{len(raw_sources)} sources validated")
        
        # Test complete research pipeline with simple query
        print("   Testing complete enhanced research pipeline...")
        try:
            result = await agent.conduct_enhanced_research("what is backpropagation", "apa")
            
            if result.synthesis and "Research failed" not in result.synthesis:
                print(f"   ✅ Enhanced research completed successfully!")
                print(f"      - Sources found: {len(result.validated_sources)}")
                print(f"      - Synthesis length: {len(result.synthesis)} chars")
                print(f"      - Confidence: {result.confidence_score:.3f}")
                print(f"      - Research time: {result.research_time:.1f}s")
                
                if result.quality_assessment:
                    print(f"      - Quality level: {result.quality_assessment.overall_quality.value}")
                
                return True
            else:
                print(f"   ❌ Enhanced research failed: {result.synthesis}")
                return False
                
        except Exception as e:
            print(f"   ❌ Enhanced research pipeline failed: {e}")
            logger.error(f"Enhanced pipeline error: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        print(f"   ❌ Enhanced research agent test failed: {e}")
        logger.error(f"Enhanced agent error: {traceback.format_exc()}")
        return False


def test_cli_components():
    """Test CLI components without running interactive mode."""
    print("\n💻 Testing CLI Components...")
    
    try:
        from cli import ResearchCLI
        # Test CLI initialization without API key (should not crash)
        try:
            # This should work for basic initialization
            api_key = os.getenv("OPENAI_API_KEY", "test-key")
            cli = ResearchCLI(api_key)
            assert cli.agent is not None, "CLI agent not initialized"
            print("   ✅ CLI initialization")
        except Exception as e:
            print(f"   ❌ CLI initialization failed: {e}")
            return False
        
        # Test markdown formatting
        test_result = {
            "query": "Test Query",
            "timestamp": "2024-01-01T12:00:00",
            "num_sources": 2,
            "research_time": 30.5,
            "synthesis": "This is a test synthesis.",
            "sources": [
                {
                    "title": "Test Paper 1", 
                    "authors": ["Author One"],
                    "year": "2024",
                    "url": "https://example.com"
                },
                {
                    "title": "Test Paper 2",
                    "authors": ["Author Two", "Author Three"],
                    "year": "2023"
                }
            ]
        }
        
        cli.current_result = test_result
        markdown = cli.format_markdown_report()
        
        assert "# Research Report" in markdown, "Markdown formatting failed"
        assert "Test Query" in markdown, "Query not in markdown"
        assert "Test Paper 1" in markdown, "Sources not in markdown"
        
        print("   ✅ Markdown formatting")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        return False


async def run_system_test():
    """Run comprehensive system test."""
    print("🧪 Enhanced Autonomous Research Agent - System Test")
    print("=" * 60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment first
    print("\n🔍 Environment Check:")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("❌ OPENAI_API_KEY not set - some tests will be skipped")
    
    # Check imports
    if not test_imports():
        print("❌ Critical import failure - cannot continue")
        return False
    
    # Check dependencies
    ml_available = test_dependencies()
    
    results = []
    
    # Track test results - synchronous tests
    sync_tests = [
        ("Citation Manager", test_citation_manager),
        ("Content Extractor", test_content_extractor),
        ("CLI Components", test_cli_components)
    ]
    
    # Add async tests
    async_tests = [
        ("Basic Research Agent", test_research_agent),
        ("Enhanced Research Agent", test_enhanced_research_agent)
    ]
    
    # Run synchronous tests
    print("\n🔬 Running Synchronous Tests:")
    for test_name, test_func in sync_tests:
        try:
            result = test_func()
            results.append((test_name, result, "sync"))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            logger.error(f"{test_name} crash: {traceback.format_exc()}")
            results.append((test_name, False, "sync"))
    
    # Run asynchronous tests
    print("\n🔬 Running Asynchronous Tests:")
    for test_name, test_func in async_tests:
        try:
            result = await test_func()
            results.append((test_name, result, "async"))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            logger.error(f"{test_name} crash: {traceback.format_exc()}")
            results.append((test_name, False, "async"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("-" * 40)
    
    passed = 0
    total = len(results)
    critical_passed = 0
    critical_total = 0
    
    for test_name, result, test_type in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        
        # Enhanced agent is critical
        if test_name == "Enhanced Research Agent":
            critical_total += 1
            if result:
                critical_passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed ({100*passed//total if total > 0 else 0}%)")
    
    # Provide specific guidance
    if critical_passed == critical_total and critical_total > 0:
        print("🎉 Enhanced Research Agent is working! The system is ready for research.")
        print("\n🚀 Quick Start Guide:")
        print("1. Run the enhanced CLI: python cli.py --enhanced")
        print("2. Try a query: research \"explain backpropagation in neural networks\"")
        print("3. Export results: export txt")
        
        if not ml_available:
            print("\n💡 For better results, install ML dependencies:")
            print("   pip install sentence-transformers scikit-learn")
            
    elif not api_key:
        print("\n⚠️ Cannot test research functionality without OpenAI API key")
        print("Set OPENAI_API_KEY environment variable and re-run tests")
        
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        print("\n🛠️ Troubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check your OpenAI API key is valid")
        print("3. Check internet connectivity for ArXiv/web searches")
        print("4. Run with debug mode: python test_system.py")
    
    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return True if critical tests passed
    return critical_passed == critical_total and critical_total > 0


async def main():
    """Main test function."""
    try:
        success = await run_system_test()
        if success:
            print("\n✅ System tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ System tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        logger.error(f"Test runner error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())