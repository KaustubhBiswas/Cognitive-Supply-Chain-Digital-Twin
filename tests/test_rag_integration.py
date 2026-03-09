"""
Tests for RAG integration with cognitive tools and analyst.
"""

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGToolsIntegration:
    """Test RAG tools integration with cognitive layer."""
    
    def test_rag_tools_importable(self):
        """Test that RAG tools can be imported."""
        from src.cognition.tools import (get_best_practices,
                                         get_disruption_context, get_rag_stats,
                                         is_rag_available,
                                         search_supply_chain_knowledge)
        
        assert search_supply_chain_knowledge is not None
        assert get_disruption_context is not None
        assert get_best_practices is not None
        assert callable(is_rag_available)
        assert callable(get_rag_stats)
    
    def test_rag_not_available_without_init(self):
        """Test that RAG is not available without initialization."""
        from src.cognition.tools import get_rag_stats, is_rag_available

        # Without initialization, RAG should not be available
        # (unless previously initialized in same session)
        stats = get_rag_stats()
        if not stats.get("available"):
            assert is_rag_available() == False
    
    def test_search_knowledge_without_retriever(self):
        """Test search returns error when retriever not initialized."""
        # Reset retriever to None
        import src.cognition.tools as tools_module
        from src.cognition.tools import search_supply_chain_knowledge
        original = tools_module._rag_retriever
        tools_module._rag_retriever = None
        
        try:
            result = search_supply_chain_knowledge.invoke({
                "query": "semiconductor shortage mitigation"
            })
            
            assert isinstance(result, dict)
            assert result["success"] == False
            assert "not initialized" in result.get("error", "")
            assert result["results"] == []
        finally:
            tools_module._rag_retriever = original
    
    def test_get_disruption_context_without_retriever(self):
        """Test disruption context returns error when retriever not initialized."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import get_disruption_context
        original = tools_module._rag_retriever
        tools_module._rag_retriever = None
        
        try:
            result = get_disruption_context.invoke({
                "disruption_type": "stockout",
                "severity": "high",
            })
            
            assert isinstance(result, dict)
            assert result["success"] == False
            assert result["mitigation_strategies"] == []
            assert result["similar_cases"] == []
        finally:
            tools_module._rag_retriever = original
    
    def test_get_best_practices_without_retriever(self):
        """Test best practices returns error when retriever not initialized."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import get_best_practices
        original = tools_module._rag_retriever
        tools_module._rag_retriever = None
        
        try:
            result = get_best_practices.invoke({
                "topic": "inventory management"
            })
            
            assert isinstance(result, dict)
            assert result["success"] == False
            assert result["practices"] == []
        finally:
            tools_module._rag_retriever = original
    
    def test_get_all_tools_without_rag(self):
        """Test get_all_tools returns base tools when RAG not available."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import get_all_tools
        original = tools_module._rag_retriever
        tools_module._rag_retriever = None
        
        try:
            tools = get_all_tools()
            tool_names = [t.name for t in tools]
            
            # Should have base tools and JIT tools
            assert "forecast_demand" in tool_names
            assert "get_jit_recommendations" in tool_names
            
            # Should NOT have RAG tools when retriever is None
            assert "search_supply_chain_knowledge" not in tool_names
            assert "get_disruption_context" not in tool_names
            assert "get_best_practices" not in tool_names
        finally:
            tools_module._rag_retriever = original


class TestRAGToolsWithMockRetriever:
    """Test RAG tools with mock retriever."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        from src.cognition.rag import (QueryAnalysis, QueryIntent,
                                       SearchResult, SearchResults)
        
        mock = MagicMock()
        
        # Mock retrieve method
        mock.retrieve.return_value = SearchResults(
            query="test query",
            results=[
                SearchResult(
                    chunk_id="c1",
                    doc_id="doc1",
                    content="Best practice for inventory management: maintain safety stock levels.",
                    score=0.95,
                    collection="best_practices",
                    metadata={"doc_type": "best_practice", "severity": "medium"},
                )
            ],
            search_time_ms=5.0,
            collections_searched=["best_practices"],
            total_results=1,
        )
        
        # Mock get_context_for_llm
        mock.get_context_for_llm.return_value = "Relevant context: Best practice for inventory management..."
        
        # Mock query_router.analyze_query
        from src.cognition.rag import CollectionType
        mock.query_router.analyze_query.return_value = QueryAnalysis(
            original_query="test query",
            normalized_query="test query",
            intent=QueryIntent.BEST_PRACTICE,
            confidence=0.9,
            detected_entities=[],
            detected_disruption_type=None,
            detected_region=None,
            is_urgent=False,
            suggested_collections=[CollectionType.BEST_PRACTICES],
        )
        
        # Mock retrieve_for_disruption
        mock.retrieve_for_disruption.return_value = SearchResults(
            query="disruption query",
            results=[
                SearchResult(
                    chunk_id="c2",
                    doc_id="doc2",
                    content="Mitigation strategy: Increase safety stock by 20%.",
                    score=0.88,
                    collection="best_practices",
                    metadata={"doc_type": "best_practice"},
                )
            ],
            search_time_ms=3.0,
            collections_searched=["best_practices", "case_studies"],
            total_results=1,
        )
        
        # Mock retrieve_best_practices
        mock.retrieve_best_practices.return_value = SearchResults(
            query="best practices query",
            results=[
                SearchResult(
                    chunk_id="c3",
                    doc_id="doc3",
                    content="Use ABC analysis for inventory classification.",
                    score=0.92,
                    collection="best_practices",
                    metadata={"doc_type": "best_practice", "industry": "manufacturing"},
                )
            ],
            search_time_ms=2.0,
            collections_searched=["best_practices"],
            total_results=1,
        )
        
        return mock
    
    def test_search_knowledge_with_mock(self, mock_retriever):
        """Test search with mock retriever."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import search_supply_chain_knowledge
        
        original = tools_module._rag_retriever
        tools_module._rag_retriever = mock_retriever
        
        try:
            result = search_supply_chain_knowledge.invoke({
                "query": "inventory management best practices",
                "n_results": 5,
            })
            
            assert result["success"] == True
            assert len(result["results"]) > 0
            assert "context" in result
            assert "query_analysis" in result
            assert result["query_analysis"]["intent"] == "best_practice"
        finally:
            tools_module._rag_retriever = original
    
    def test_get_disruption_context_with_mock(self, mock_retriever):
        """Test disruption context with mock retriever."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import get_disruption_context
        
        original = tools_module._rag_retriever
        tools_module._rag_retriever = mock_retriever
        
        try:
            result = get_disruption_context.invoke({
                "disruption_type": "stockout",
                "severity": "high",
                "include_mitigation": True,
            })
            
            assert result["success"] == True
            assert result["disruption_type"] == "stockout"
            assert result["severity"] == "high"
        finally:
            tools_module._rag_retriever = original
    
    def test_get_best_practices_with_mock(self, mock_retriever):
        """Test best practices with mock retriever."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import get_best_practices
        
        original = tools_module._rag_retriever
        tools_module._rag_retriever = mock_retriever
        
        try:
            result = get_best_practices.invoke({
                "topic": "inventory classification",
                "industry": "manufacturing",
                "max_results": 3,
            })
            
            assert result["success"] == True
            assert result["topic_analyzed"] == "inventory classification"
            assert result["industry_filter"] == "manufacturing"
            assert len(result["practices"]) > 0
        finally:
            tools_module._rag_retriever = original
    
    def test_get_all_tools_with_rag(self, mock_retriever):
        """Test get_all_tools includes RAG tools when available."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import get_all_tools
        
        original = tools_module._rag_retriever
        tools_module._rag_retriever = mock_retriever
        
        try:
            tools = get_all_tools()
            tool_names = [t.name for t in tools]
            
            # Should have RAG tools
            assert "search_supply_chain_knowledge" in tool_names
            assert "get_disruption_context" in tool_names
            assert "get_best_practices" in tool_names
        finally:
            tools_module._rag_retriever = original


class TestAnalystRAGIntegration:
    """Test analyst integration with RAG context."""
    
    def test_analyst_prompt_includes_rag(self):
        """Test analyst prompt file includes RAG tool references."""
        prompt_path = Path(__file__).parent.parent / "src" / "cognition" / "prompts" / "analyst.txt"
        
        if prompt_path.exists():
            content = prompt_path.read_text()
            
            assert "RAG Knowledge Base Tools" in content
            assert "search_supply_chain_knowledge" in content
            assert "get_disruption_context" in content
            assert "get_best_practices" in content
            assert "rag_context" in content
    
    def test_gather_analysis_handles_no_rag(self):
        """Test _gather_analysis_data works when RAG is unavailable."""
        from src.cognition.analyst import _gather_analysis_data
        from src.cognition.tools import initialize_tools

        # Create minimal state
        state = {
            "current_alert": {
                "alert_type": "stockout",
                "severity": "high",
                "affected_nodes": [1, 2],
            }
        }
        
        # Should not raise even without RAG
        try:
            data = _gather_analysis_data(state)
            assert isinstance(data, dict)
            # May or may not have rag_context depending on state
        except Exception as e:
            # Should only fail for expected reasons (simulation not init)
            assert "simulation" in str(e).lower() or "_simulation" in str(e).lower()


class TestInitializeToolsWithRAG:
    """Test initialize_tools with RAG retriever."""
    
    def test_initialize_tools_accepts_rag_retriever(self):
        """Test that initialize_tools accepts rag_retriever parameter."""
        import inspect

        from src.cognition.tools import initialize_tools
        
        sig = inspect.signature(initialize_tools)
        params = list(sig.parameters.keys())
        
        assert "rag_retriever" in params
    
    def test_initialize_with_mock_rag(self):
        """Test initializing with mock RAG retriever."""
        import src.cognition.tools as tools_module
        from src.cognition.tools import initialize_tools, is_rag_available
        
        mock_sim = MagicMock()
        mock_retriever = MagicMock()
        
        original = tools_module._rag_retriever
        
        try:
            initialize_tools(
                simulation=mock_sim,
                rag_retriever=mock_retriever,
            )
            
            assert is_rag_available() == True
            assert tools_module._rag_retriever is mock_retriever
        finally:
            tools_module._rag_retriever = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
