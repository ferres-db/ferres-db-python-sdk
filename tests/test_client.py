"""Tests for VectorDBClient."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from vector_db_client import (
    VectorDBClient,
    Point,
    Collection,
    CollectionListItem,
    SearchResult,
    UpsertResult,
    DistanceMetric,
    ApiKeyInfo,
    CreateKeyResponse,
    EstimateSearchResponse,
    QueryCostEstimate,
    CostBreakdown,
    HistoricalLatency,
    SearchExplanation,
    ExplainResult,
    FilterExplanation,
    ConditionResult,
    IndexStats,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    InvalidDimensionError,
    InvalidPayloadError,
    BudgetExceededError,
    ConnectionError,
)


@pytest.fixture
def client():
    """Create a test client."""
    return VectorDBClient(base_url="http://localhost:3000", timeout=30)


@pytest.fixture
def mock_response():
    """Create a mock response helper."""
    def _create(status_code: int, json_data: Dict[str, Any] = None):
        response = AsyncMock(spec=httpx.Response)
        response.status_code = status_code
        response.is_success = 200 <= status_code < 300
        if json_data:
            # Make json() an async function that returns the data
            async def json_mock():
                return json_data
            response.json = json_mock
        else:
            async def json_mock():
                return {}
            response.json = json_mock
        response.text = ""
        return response
    return _create


@pytest.mark.asyncio
async def test_create_collection_success(client, mock_response):
    """Test successful collection creation."""
    response_data = {
        "name": "test-collection",
        "dimension": 128,
        "distance": "Cosine",
        "created_at": 1234567890,
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(201, response_data)
        
        collection = await client.create_collection(
            name="test-collection",
            dimension=128,
            distance=DistanceMetric.COSINE,
        )
        
        assert collection.name == "test-collection"
        assert collection.dimension == 128
        assert collection.distance == DistanceMetric.COSINE
        assert collection.created_at == 1234567890
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "POST"
        assert "/api/v1/collections" in call_args.kwargs["url"]
        assert call_args.kwargs["json"] == {
            "name": "test-collection",
            "dimension": 128,
            "distance": "Cosine",
        }


@pytest.mark.asyncio
async def test_create_collection_with_bm25_options(client, mock_response):
    """Test collection creation with enable_bm25 and bm25_text_field."""
    response_data = {
        "name": "docs",
        "dimension": 384,
        "distance": "Cosine",
        "created_at": 1234567890,
    }
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(201, response_data)
        await client.create_collection(
            name="docs",
            dimension=384,
            distance=DistanceMetric.COSINE,
            enable_bm25=True,
            bm25_text_field="content",
        )
        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == {
            "name": "docs",
            "dimension": 384,
            "distance": "Cosine",
            "enable_bm25": True,
            "bm25_text_field": "content",
        }


def test_client_with_api_key_sets_authorization_header():
    """Test that client with api_key sets Authorization header."""
    c = VectorDBClient(
        base_url="http://localhost:8080",
        api_key="ferres_sk_test123",
    )
    assert c.client.headers.get("Authorization") == "Bearer ferres_sk_test123"


def test_client_without_api_key_has_no_authorization_header():
    """Test that client without api_key has no Authorization header."""
    c = VectorDBClient(base_url="http://localhost:8080")
    assert "Authorization" not in (c.client.headers or {})


@pytest.mark.asyncio
async def test_list_keys(client, mock_response):
    """Test listing API keys."""
    response_data = [
        {"id": 1, "name": "key1", "key_prefix": "ferres_sk_ab", "created_at": 1000},
        {"id": 2, "name": "key2", "key_prefix": "ferres_sk_cd", "created_at": 2000},
    ]
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        keys = await client.list_keys()
        assert len(keys) == 2
        assert keys[0].id == 1 and keys[0].name == "key1"
        assert keys[1].id == 2 and keys[1].name == "key2"
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "GET"
        assert "/api/v1/keys" in call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_create_key(client, mock_response):
    """Test creating an API key."""
    response_data = {
        "id": 1,
        "name": "my-key",
        "key": "ferres_sk_raw_once",
        "key_prefix": "ferres_sk_ra",
        "created_at": 1234567890,
    }
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        result = await client.create_key("my-key")
        assert result.id == 1
        assert result.name == "my-key"
        assert result.key == "ferres_sk_raw_once"
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "POST"
        assert call_args.kwargs["json"] == {"name": "my-key"}


@pytest.mark.asyncio
async def test_create_key_trims_name(client, mock_response):
    """Test that create_key trims the name."""
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, {"id": 1, "name": "x", "key": "k", "key_prefix": "p", "created_at": 0})
        await client.create_key("  trimmed  ")
        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == {"name": "trimmed"}


@pytest.mark.asyncio
async def test_create_key_empty_name_raises(client):
    """Test that create_key with empty name raises InvalidPayloadError."""
    with pytest.raises(InvalidPayloadError):
        await client.create_key("")
    with pytest.raises(InvalidPayloadError):
        await client.create_key("   ")


@pytest.mark.asyncio
async def test_delete_key(client, mock_response):
    """Test deleting an API key."""
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200)
        await client.delete_key(42)
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "DELETE"
        assert "/api/v1/keys/42" in call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_create_collection_already_exists(client, mock_response):
    """Test collection creation when collection already exists."""
    error_data = {
        "error": "collection_already_exists",
        "message": "collection 'test-collection' already exists",
        "code": 409,
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(409, error_data)
        
        with pytest.raises(CollectionAlreadyExistsError) as exc_info:
            await client.create_collection(
                name="test-collection",
                dimension=128,
                distance=DistanceMetric.COSINE,
            )
        
        assert "test-collection" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_collections(client, mock_response):
    """Test listing collections."""
    response_data = {
        "collections": [
            {
                "name": "collection-1",
                "dimension": 128,
                "num_points": 100,
                "created_at": 1234567890,
            },
            {
                "name": "collection-2",
                "dimension": 256,
                "num_points": 50,
                "created_at": 1234567891,
            },
        ]
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        
        collections = await client.list_collections()
        
        assert len(collections) == 2
        assert collections[0].name == "collection-1"
        assert collections[0].dimension == 128
        assert collections[0].num_points == 100
        assert collections[1].name == "collection-2"
        assert collections[1].dimension == 256


@pytest.mark.asyncio
async def test_delete_collection(client, mock_response):
    """Test deleting a collection."""
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(204)
        
        await client.delete_collection("test-collection")
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "DELETE"
        assert "/api/v1/collections/test-collection" in call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_delete_collection_not_found(client, mock_response):
    """Test deleting a non-existent collection."""
    error_data = {
        "error": "collection_not_found",
        "message": "collection 'test-collection' not found",
        "code": 404,
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(404, error_data)
        
        with pytest.raises(CollectionNotFoundError) as exc_info:
            await client.delete_collection("test-collection")
        
        assert "test-collection" in str(exc_info.value)


@pytest.mark.asyncio
async def test_upsert_points_single_batch(client, mock_response):
    """Test upserting points in a single batch."""
    points = [
        Point(id="1", vector=[0.1, 0.2, 0.3], metadata={"text": "hello"}),
        Point(id="2", vector=[0.4, 0.5, 0.6], metadata={"text": "world"}),
    ]
    
    response_data = {
        "upserted": 2,
        "failed": [],
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        
        result = await client.upsert_points("test-collection", points)
        
        assert result.upserted == 2
        assert len(result.failed) == 0
        
        # Verify request payload
        call_args = mock_request.call_args
        assert call_args.kwargs["json"]["points"] == [
            {"id": "1", "vector": [0.1, 0.2, 0.3], "metadata": {"text": "hello"}},
            {"id": "2", "vector": [0.4, 0.5, 0.6], "metadata": {"text": "world"}},
        ]


@pytest.mark.asyncio
async def test_upsert_points_large_batch(client, mock_response):
    """Test upserting points with automatic batching."""
    # Create 2500 points (should be split into 3 batches: 1000, 1000, 500)
    points = [
        Point(id=str(i), vector=[0.1, 0.2, 0.3], metadata={"index": i})
        for i in range(2500)
    ]
    
    response_data = {
        "upserted": 1000,
        "failed": [],
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        
        result = await client.upsert_points("test-collection", points)
        
        # Should have been called 3 times (1000 + 1000 + 500)
        assert mock_request.call_count == 3
        assert result.upserted == 3000  # 3 batches * 1000
        assert len(result.failed) == 0


@pytest.mark.asyncio
async def test_upsert_points_empty_list(client):
    """Test upserting empty list."""
    result = await client.upsert_points("test-collection", [])
    
    assert result.upserted == 0
    assert len(result.failed) == 0


@pytest.mark.asyncio
async def test_upsert_points_with_failures(client, mock_response):
    """Test upserting points with some failures."""
    points = [
        Point(id="1", vector=[0.1, 0.2, 0.3], metadata={}),
        Point(id="2", vector=[0.4, 0.5, 0.6], metadata={}),
    ]
    
    response_data = {
        "upserted": 1,
        "failed": [
            {"id": "2", "reason": "dimension mismatch: expected 3, got 3"},
        ],
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        
        result = await client.upsert_points("test-collection", points)
        
        assert result.upserted == 1
        assert len(result.failed) == 1
        assert result.failed[0]["id"] == "2"


@pytest.mark.asyncio
async def test_delete_points(client, mock_response):
    """Test deleting points."""
    ids = ["1", "2", "3"]
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200)
        
        await client.delete_points("test-collection", ids)
        
        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == {"ids": ids}


@pytest.mark.asyncio
async def test_delete_points_empty_list(client):
    """Test deleting points with empty list."""
    with pytest.raises(InvalidPayloadError) as exc_info:
        await client.delete_points("test-collection", [])
    
    assert "cannot be empty" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_search(client, mock_response):
    """Test searching for similar vectors."""
    response_data = {
        "results": [
            {
                "id": "1",
                "score": 0.95,
                "metadata": {"text": "hello"},
            },
            {
                "id": "2",
                "score": 0.85,
                "metadata": {"text": "world"},
            },
        ],
        "took_ms": 10,
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        
        results = await client.search(
            collection="test-collection",
            vector=[0.1, 0.2, 0.3],
            limit=10,
        )
        
        assert len(results) == 2
        assert results[0].id == "1"
        assert results[0].score == 0.95
        assert results[0].metadata == {"text": "hello"}
        
        # Verify request payload
        call_args = mock_request.call_args
        assert call_args.kwargs["json"]["vector"] == [0.1, 0.2, 0.3]
        assert call_args.kwargs["json"]["limit"] == 10


@pytest.mark.asyncio
async def test_search_with_filter(client, mock_response):
    """Test searching with metadata filter."""
    response_data = {
        "results": [
            {
                "id": "1",
                "score": 0.95,
                "metadata": {"category": "A"},
            },
        ],
        "took_ms": 5,
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)
        
        results = await client.search(
            collection="test-collection",
            vector=[0.1, 0.2, 0.3],
            limit=10,
            filter={"category": "A"},
        )
        
        assert len(results) == 1
        
        # Verify filter was included in request
        call_args = mock_request.call_args
        assert call_args.kwargs["json"]["filter"] == {"category": "A"}


@pytest.mark.asyncio
async def test_retry_on_server_error(client, mock_response):
    """Test automatic retry on server errors."""
    # First two calls return 500, third succeeds
    responses = [
        mock_response(500),
        mock_response(500),
        mock_response(200, {"collections": []}),
    ]
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = responses
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            collections = await client.list_collections()
            
            # Should have retried twice (3 total calls)
            assert mock_request.call_count == 3
            # Should have slept twice (exponential backoff)
            assert mock_sleep.call_count == 2
            assert len(collections) == 0


@pytest.mark.asyncio
async def test_retry_on_timeout(client, mock_response):
    """Test automatic retry on timeout."""
    import httpx
    
    # First call times out, second succeeds
    timeout_error = httpx.TimeoutException("Request timeout")
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = [
            timeout_error,
            mock_response(200, {"collections": []}),
        ]
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            collections = await client.list_collections()
            
            assert mock_request.call_count == 2
            assert len(collections) == 0


@pytest.mark.asyncio
async def test_no_retry_on_client_error(client, mock_response):
    """Test that client errors (4xx) are not retried."""
    error_data = {
        "error": "collection_not_found",
        "message": "collection 'test' not found",
        "code": 404,
    }
    
    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(404, error_data)
        
        with pytest.raises(CollectionNotFoundError):
            await client.delete_collection("test")
        
        # Should not retry on 4xx errors
        assert mock_request.call_count == 1


@pytest.mark.asyncio
async def test_context_manager(client, mock_response):
    """Test using client as async context manager."""
    # Create a new client for this test
    new_client = VectorDBClient(base_url="http://localhost:3000")
    with patch.object(new_client.client, "aclose", new_callable=AsyncMock) as mock_close:
        async with new_client as ctx_client:
            assert isinstance(ctx_client, VectorDBClient)
        
        # Should close on exit
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_close_method(client):
    """Test explicit close method."""
    with patch.object(client.client, "aclose", new_callable=AsyncMock) as mock_close:
        await client.close()
        mock_close.assert_called_once()


# ─── Search with budget_ms ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_with_budget_ms(client, mock_response):
    """Test search sends budget_ms when provided."""
    response_data = {
        "results": [
            {"id": "1", "score": 0.95, "metadata": {"text": "hello"}},
        ],
        "took_ms": 3,
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)

        results = await client.search(
            collection="test-collection",
            vector=[0.1, 0.2, 0.3],
            limit=10,
            budget_ms=50,
        )

        assert len(results) == 1
        call_args = mock_request.call_args
        assert call_args.kwargs["json"]["budget_ms"] == 50


@pytest.mark.asyncio
async def test_search_without_budget_ms_omits_field(client, mock_response):
    """Test search does not include budget_ms when not provided."""
    response_data = {"results": [], "took_ms": 1}

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)

        await client.search(
            collection="test-collection",
            vector=[0.1, 0.2],
            limit=5,
        )

        call_args = mock_request.call_args
        assert "budget_ms" not in call_args.kwargs["json"]


@pytest.mark.asyncio
async def test_search_budget_exceeded_error(client, mock_response):
    """Test that search raises BudgetExceededError when budget is exceeded."""
    error_data = {
        "error": "budget_exceeded",
        "message": "estimated cost (12.5ms) exceeds budget (5ms)",
        "code": 422,
        "estimate": {
            "estimated_ms": 12.5,
            "confidence_range": [6.25, 18.75],
            "is_expensive": True,
            "recommendations": ["Considere reduzir limit"],
            "breakdown": {
                "index_scan_cost": 10.0,
                "filter_cost": 0.5,
                "hydration_cost": 1.0,
                "network_overhead": 1.0,
            },
        },
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(422, error_data)

        with pytest.raises(BudgetExceededError) as exc_info:
            await client.search(
                collection="test-collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                budget_ms=5,
            )

        assert exc_info.value.code == 422
        assert "12.5ms" in str(exc_info.value)
        assert exc_info.value.estimate["estimated_ms"] == 12.5


# ─── Estimate Search Cost ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_estimate_search_cost(client, mock_response):
    """Test estimate_search_cost returns a valid estimate."""
    response_data = {
        "estimated_ms": 2.35,
        "confidence_range": [1.17, 8.0],
        "estimated_memory_bytes": 45320,
        "estimated_nodes_visited": 575,
        "is_expensive": False,
        "recommendations": [],
        "breakdown": {
            "index_scan_cost": 1.76,
            "filter_cost": 0.0003,
            "hydration_cost": 0.01,
            "network_overhead": 0.1,
        },
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)

        result = await client.estimate_search_cost(
            collection="docs",
            limit=10,
        )

        assert isinstance(result, EstimateSearchResponse)
        assert result.estimate.estimated_ms == 2.35
        assert result.estimate.is_expensive is False
        assert result.estimate.estimated_nodes_visited == 575
        assert result.estimate.breakdown.index_scan_cost == 1.76
        assert result.historical_latency is None

        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == {"limit": 10}


@pytest.mark.asyncio
async def test_estimate_search_cost_with_filter_and_history(client, mock_response):
    """Test estimate_search_cost with filter and include_history."""
    response_data = {
        "estimated_ms": 5.0,
        "confidence_range": [2.5, 15.0],
        "estimated_memory_bytes": 90000,
        "estimated_nodes_visited": 1000,
        "is_expensive": True,
        "recommendations": ["Reduza limit para melhor performance"],
        "breakdown": {
            "index_scan_cost": 3.5,
            "filter_cost": 0.5,
            "hydration_cost": 0.5,
            "network_overhead": 0.5,
        },
        "historical_latency": {
            "p50_ms": 2.0,
            "p95_ms": 8.0,
            "p99_ms": 15.0,
            "avg_ms": 3.2,
            "total_queries": 1520,
        },
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)

        result = await client.estimate_search_cost(
            collection="docs",
            limit=100,
            filter={"category": "tech"},
            include_history=True,
        )

        assert result.estimate.is_expensive is True
        assert len(result.estimate.recommendations) == 1
        assert result.historical_latency is not None
        assert result.historical_latency.p50_ms == 2.0
        assert result.historical_latency.total_queries == 1520

        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == {
            "limit": 100,
            "filter": {"category": "tech"},
            "include_history": True,
        }


@pytest.mark.asyncio
async def test_estimate_search_cost_collection_not_found(client, mock_response):
    """Test estimate returns CollectionNotFoundError for unknown collection."""
    error_data = {
        "error": "collection_not_found",
        "message": "collection 'nope' not found",
        "code": 404,
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(404, error_data)

        with pytest.raises(CollectionNotFoundError):
            await client.estimate_search_cost(collection="nope", limit=10)


# ─── Explain Search ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_explain(client, mock_response):
    """Test search_explain returns a valid explanation."""
    response_data = {
        "query_vector_norm": 0.245,
        "distance_metric": "Cosine",
        "candidates_scanned": 30,
        "candidates_after_filter": 5,
        "results": [
            {
                "id": "doc-1",
                "score": 0.12,
                "distance_metric": "Cosine",
                "raw_distance": 0.12,
                "score_breakdown": {"vector_score": 0.12},
                "filter_evaluation": None,
                "rank_before_filter": 1,
                "rank_after_filter": 1,
            }
        ],
        "index_stats": {
            "total_points": 1000,
            "hnsw_layers": 16,
            "ef_search_used": 50,
            "tombstones_skipped": 0,
        },
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)

        explanation = await client.search_explain(
            collection="docs",
            vector=[0.1, 0.2, -0.1],
            limit=5,
        )

        assert isinstance(explanation, SearchExplanation)
        assert explanation.query_vector_norm == 0.245
        assert explanation.distance_metric == "Cosine"
        assert explanation.candidates_scanned == 30
        assert explanation.candidates_after_filter == 5
        assert len(explanation.results) == 1
        assert explanation.results[0].id == "doc-1"
        assert explanation.results[0].score_breakdown["vector_score"] == 0.12
        assert explanation.index_stats.total_points == 1000
        assert explanation.index_stats.ef_search_used == 50

        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == {
            "vector": [0.1, 0.2, -0.1],
            "limit": 5,
        }


@pytest.mark.asyncio
async def test_search_explain_with_filter(client, mock_response):
    """Test search_explain with filter returns per-condition evaluation."""
    response_data = {
        "query_vector_norm": 0.3,
        "distance_metric": "Cosine",
        "candidates_scanned": 50,
        "candidates_after_filter": 3,
        "results": [
            {
                "id": "doc-1",
                "score": 0.85,
                "distance_metric": "Cosine",
                "raw_distance": 0.85,
                "score_breakdown": {"vector_score": 0.85},
                "filter_evaluation": {
                    "conditions": [
                        {
                            "field": "category",
                            "operator": "$eq",
                            "expected": "tech",
                            "actual": "tech",
                            "passed": True,
                        },
                        {
                            "field": "price",
                            "operator": "$gte",
                            "expected": 10,
                            "actual": 25,
                            "passed": True,
                        },
                    ],
                    "passed": True,
                },
                "rank_before_filter": 1,
                "rank_after_filter": 1,
            },
        ],
        "index_stats": {
            "total_points": 5000,
            "hnsw_layers": 20,
            "ef_search_used": 100,
            "tombstones_skipped": 3,
        },
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(200, response_data)

        explanation = await client.search_explain(
            collection="products",
            vector=[0.1, 0.2, -0.1],
            limit=10,
            filter={"category": "tech", "price": {"$gte": 10}},
        )

        assert explanation.candidates_after_filter == 3
        result = explanation.results[0]
        assert result.filter_evaluation is not None
        assert result.filter_evaluation.passed is True
        assert len(result.filter_evaluation.conditions) == 2
        assert result.filter_evaluation.conditions[0].field == "category"
        assert result.filter_evaluation.conditions[0].passed is True
        assert result.filter_evaluation.conditions[1].operator == "$gte"
        assert explanation.index_stats.tombstones_skipped == 3

        call_args = mock_request.call_args
        assert call_args.kwargs["json"]["filter"] == {
            "category": "tech",
            "price": {"$gte": 10},
        }


@pytest.mark.asyncio
async def test_search_explain_collection_not_found(client, mock_response):
    """Test search_explain returns CollectionNotFoundError for unknown collection."""
    error_data = {
        "error": "collection_not_found",
        "message": "collection 'nope' not found",
        "code": 404,
    }

    with patch.object(client.client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response(404, error_data)

        with pytest.raises(CollectionNotFoundError):
            await client.search_explain(
                collection="nope",
                vector=[0.1, 0.2],
                limit=5,
            )
