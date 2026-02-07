"""Tests for models."""

import pytest
from vector_db_client import (
    Point,
    Collection,
    CollectionListItem,
    CollectionDetail,
    CollectionStats,
    PointDetail,
    ListPointsResult,
    SearchResult,
    SearchResponse,
    UpsertResult,
    DeletePointsResult,
    DistanceMetric,
    ScalarType,
    ScalarQuantizationConfig,
    QuantizationConfig,
    WsAckMessage,
    WsEventMessage,
    WsErrorMessage,
)


def test_point_to_dict():
    """Test Point.to_dict()."""
    point = Point(
        id="test-1",
        vector=[0.1, 0.2, 0.3],
        metadata={"text": "hello", "score": 0.95},
    )
    
    result = point.to_dict()
    
    assert result == {
        "id": "test-1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"text": "hello", "score": 0.95},
    }


def test_point_from_dict():
    """Test Point.from_dict()."""
    data = {
        "id": "test-1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"text": "hello"},
    }
    
    point = Point.from_dict(data)
    
    assert point.id == "test-1"
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.metadata == {"text": "hello"}


def test_point_from_dict_no_metadata():
    """Test Point.from_dict() with missing metadata."""
    data = {
        "id": "test-1",
        "vector": [0.1, 0.2, 0.3],
    }
    
    point = Point.from_dict(data)
    
    assert point.metadata == {}


def test_collection_to_dict():
    """Test Collection.to_dict()."""
    collection = Collection(
        name="test-collection",
        dimension=128,
        distance=DistanceMetric.COSINE,
    )
    
    result = collection.to_dict()
    
    assert result == {
        "name": "test-collection",
        "dimension": 128,
        "distance": "Cosine",
    }


def test_collection_list_item_from_dict():
    """Test CollectionListItem.from_dict()."""
    data = {
        "name": "test-collection",
        "dimension": 128,
        "num_points": 100,
        "created_at": 1234567890,
    }
    
    item = CollectionListItem.from_dict(data)
    
    assert item.name == "test-collection"
    assert item.dimension == 128
    assert item.num_points == 100
    assert item.created_at == 1234567890


def test_search_result_from_dict():
    """Test SearchResult.from_dict()."""
    data = {
        "id": "test-1",
        "score": 0.95,
        "metadata": {"text": "hello"},
    }
    
    result = SearchResult.from_dict(data)
    
    assert result.id == "test-1"
    assert result.score == 0.95
    assert result.metadata == {"text": "hello"}


def test_upsert_result_from_dict():
    """Test UpsertResult.from_dict()."""
    data = {
        "upserted": 5,
        "failed": [
            {"id": "bad-1", "reason": "dimension mismatch"},
            {"id": "bad-2", "reason": "invalid vector"},
        ],
    }
    
    result = UpsertResult.from_dict(data)
    
    assert result.upserted == 5
    assert len(result.failed) == 2
    assert result.failed[0]["id"] == "bad-1"
    assert result.failed[1]["reason"] == "invalid vector"


def test_upsert_result_from_dict_no_failed():
    """Test UpsertResult.from_dict() with missing failed field."""
    data = {
        "upserted": 5,
    }
    
    result = UpsertResult.from_dict(data)
    
    assert result.upserted == 5
    assert result.failed == []


def test_distance_metric_values():
    """Test DistanceMetric enum values."""
    assert DistanceMetric.COSINE.value == "Cosine"
    assert DistanceMetric.DOT_PRODUCT.value == "DotProduct"
    assert DistanceMetric.EUCLIDEAN.value == "Euclidean"


def test_collection_list_item_with_distance():
    """Test CollectionListItem.from_dict() parses distance."""
    data = {
        "name": "test",
        "dimension": 128,
        "num_points": 10,
        "created_at": 1234567890,
        "distance": "Euclidean",
    }
    item = CollectionListItem.from_dict(data)
    assert item.distance == DistanceMetric.EUCLIDEAN


def test_collection_list_item_without_distance_defaults_to_cosine():
    """Test CollectionListItem defaults distance to Cosine if missing."""
    data = {
        "name": "test",
        "dimension": 128,
        "num_points": 10,
        "created_at": 1234567890,
    }
    item = CollectionListItem.from_dict(data)
    assert item.distance == DistanceMetric.COSINE


# ─── Quantization Models ──────────────────────────────────────────────────


def test_scalar_type_values():
    """Test ScalarType enum."""
    assert ScalarType.INT8.value == "Int8"


def test_scalar_quantization_config_to_dict():
    """Test ScalarQuantizationConfig.to_dict()."""
    config = ScalarQuantizationConfig(
        dtype=ScalarType.INT8,
        always_ram=True,
        quantile=95.0,
    )
    result = config.to_dict()
    assert result == {"dtype": "Int8", "always_ram": True, "quantile": 95.0}


def test_scalar_quantization_config_to_dict_defaults_omitted():
    """Test that default values are omitted from to_dict()."""
    config = ScalarQuantizationConfig()
    result = config.to_dict()
    # always_ram=False and quantile=99.5 are defaults, should be omitted
    assert result == {"dtype": "Int8"}


def test_scalar_quantization_config_from_dict():
    """Test ScalarQuantizationConfig.from_dict()."""
    config = ScalarQuantizationConfig.from_dict({
        "dtype": "Int8",
        "always_ram": True,
        "quantile": 90.0,
    })
    assert config.dtype == ScalarType.INT8
    assert config.always_ram is True
    assert config.quantile == 90.0


def test_quantization_config_none():
    """Test QuantizationConfig.none() serialization."""
    config = QuantizationConfig.none()
    assert config.to_dict() == "None"


def test_quantization_config_scalar():
    """Test QuantizationConfig.scalar_int8() serialization."""
    config = QuantizationConfig.scalar_int8(always_ram=True, quantile=95.0)
    result = config.to_dict()
    assert result == {"Scalar": {"dtype": "Int8", "always_ram": True, "quantile": 95.0}}


def test_quantization_config_from_dict_none():
    """Test QuantizationConfig.from_dict() with 'None'."""
    config = QuantizationConfig.from_dict("None")
    assert config.scalar is None


def test_quantization_config_from_dict_scalar():
    """Test QuantizationConfig.from_dict() with Scalar."""
    config = QuantizationConfig.from_dict({
        "Scalar": {"dtype": "Int8", "always_ram": False, "quantile": 99.5}
    })
    assert config.scalar is not None
    assert config.scalar.dtype == ScalarType.INT8


def test_quantization_config_roundtrip():
    """Test serialization round-trip."""
    original = QuantizationConfig.scalar_int8(always_ram=True, quantile=97.5)
    serialized = original.to_dict()
    restored = QuantizationConfig.from_dict(serialized)
    assert restored.scalar is not None
    assert restored.scalar.always_ram is True
    assert restored.scalar.quantile == 97.5


# ─── New Response Models ──────────────────────────────────────────────────


def test_collection_detail_from_dict():
    """Test CollectionDetail.from_dict()."""
    data = {
        "name": "my-col",
        "dimension": 128,
        "num_points": 5000,
        "last_updated": 1700000000,
        "distance": "DotProduct",
        "stats": {"index_size_bytes": 2048},
    }
    detail = CollectionDetail.from_dict(data)
    assert detail.name == "my-col"
    assert detail.distance == DistanceMetric.DOT_PRODUCT
    assert detail.stats.index_size_bytes == 2048


def test_point_detail_from_dict():
    """Test PointDetail.from_dict()."""
    data = {
        "id": "p1",
        "vector": [0.1, 0.2],
        "metadata": {"key": "val"},
        "created_at": 1700000000,
    }
    point = PointDetail.from_dict(data)
    assert point.id == "p1"
    assert point.vector == [0.1, 0.2]
    assert point.created_at == 1700000000


def test_list_points_result_from_dict():
    """Test ListPointsResult.from_dict()."""
    data = {
        "points": [
            {"id": "p1", "vector": [0.1], "metadata": {}, "created_at": 1000},
        ],
        "total": 100,
        "limit": 10,
        "offset": 0,
        "has_more": True,
    }
    result = ListPointsResult.from_dict(data)
    assert len(result.points) == 1
    assert result.total == 100
    assert result.has_more is True


def test_search_response_from_dict():
    """Test SearchResponse.from_dict()."""
    data = {
        "results": [{"id": "r1", "score": 0.9, "metadata": {}}],
        "took_ms": 5,
        "query_id": "qid-1",
    }
    resp = SearchResponse.from_dict(data)
    assert len(resp.results) == 1
    assert resp.took_ms == 5
    assert resp.query_id == "qid-1"


def test_search_response_from_dict_no_query_id():
    """Test SearchResponse.from_dict() without query_id."""
    data = {
        "results": [],
        "took_ms": 1,
    }
    resp = SearchResponse.from_dict(data)
    assert resp.query_id is None


def test_delete_points_result_from_dict():
    """Test DeletePointsResult.from_dict()."""
    result = DeletePointsResult.from_dict({"deleted": 5})
    assert result.deleted == 5


# ─── WebSocket Message Models ─────────────────────────────────────────────


def test_ws_ack_message_from_dict():
    """Test WsAckMessage.from_dict()."""
    ack = WsAckMessage.from_dict({"upserted": 10, "failed": 2, "took_ms": 5})
    assert ack.upserted == 10
    assert ack.failed == 2
    assert ack.took_ms == 5


def test_ws_event_message_from_dict():
    """Test WsEventMessage.from_dict()."""
    event = WsEventMessage.from_dict({
        "collection": "docs",
        "action": "upsert",
        "point_ids": ["p1", "p2"],
        "timestamp": 1700000000,
    })
    assert event.collection == "docs"
    assert event.action == "upsert"
    assert event.point_ids == ["p1", "p2"]
    assert event.timestamp == 1700000000


def test_ws_error_message_from_dict():
    """Test WsErrorMessage.from_dict()."""
    error = WsErrorMessage.from_dict({"message": "not found", "code": 404})
    assert error.message == "not found"
    assert error.code == 404
