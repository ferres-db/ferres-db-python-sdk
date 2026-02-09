"""Models for the VectorDB client."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class DistanceMetric(str, Enum):
    """Distance metrics supported by FerresDB."""
    
    COSINE = "Cosine"
    DOT_PRODUCT = "DotProduct"
    EUCLIDEAN = "Euclidean"


# ─── Scalar Quantization (SQ8) ────────────────────────────────────────────


class ScalarType(str, Enum):
    """Scalar data types for quantization."""

    INT8 = "Int8"


@dataclass
class ScalarQuantizationConfig:
    """Configuration for scalar quantization (SQ8).

    Attributes:
        dtype: Scalar data type (currently only Int8).
        always_ram: Keep original f32 vectors in RAM for re-ranking (default: False).
        quantile: Percentile used for min/max calibration, 0-100 (default: 99.5).
    """

    dtype: ScalarType = ScalarType.INT8
    always_ram: bool = False
    quantile: float = 99.5

    def to_dict(self) -> dict:
        """Serialize to the server's JSON format."""
        d: Dict[str, Any] = {"dtype": self.dtype.value}
        if self.always_ram:
            d["always_ram"] = True
        if self.quantile != 99.5:
            d["quantile"] = self.quantile
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ScalarQuantizationConfig":
        """Create from a dictionary."""
        return cls(
            dtype=ScalarType(data["dtype"]),
            always_ram=data.get("always_ram", False),
            quantile=data.get("quantile", 99.5),
        )


@dataclass
class QuantizationConfig:
    """Vector quantization configuration.

    Use ``QuantizationConfig.none()`` for no quantization (default), or
    ``QuantizationConfig.scalar(...)`` to enable SQ8 compression.

    The server serializes this as a tagged enum:
      - ``"None"``
      - ``{"Scalar": {"dtype": "Int8", ...}}``
    """

    scalar: Optional[ScalarQuantizationConfig] = None

    # ── Convenience constructors ──────────────────────────────────────

    @classmethod
    def none(cls) -> "QuantizationConfig":
        """No quantization (default)."""
        return cls(scalar=None)

    @classmethod
    def scalar_int8(
        cls,
        always_ram: bool = False,
        quantile: float = 99.5,
    ) -> "QuantizationConfig":
        """Enable SQ8 (Int8) scalar quantization."""
        return cls(
            scalar=ScalarQuantizationConfig(
                dtype=ScalarType.INT8,
                always_ram=always_ram,
                quantile=quantile,
            )
        )

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> Any:
        """Serialize to the server's tagged-enum JSON format."""
        if self.scalar is None:
            return "None"
        return {"Scalar": self.scalar.to_dict()}

    @classmethod
    def from_dict(cls, data: Any) -> "QuantizationConfig":
        """Deserialize from the server's tagged-enum JSON format."""
        if data is None or data == "None":
            return cls.none()
        if isinstance(data, dict) and "Scalar" in data:
            return cls(scalar=ScalarQuantizationConfig.from_dict(data["Scalar"]))
        return cls.none()


# ─── Tiered Storage ────────────────────────────────────────────────────────


@dataclass
class TieredStorageConfig:
    """Configuration for tiered storage.

    When enabled, points are automatically moved between storage tiers
    (Hot/Warm/Cold) based on access frequency.

    Attributes:
        enabled: Enable tiered storage (default: False).
        hot_threshold_hours: Points accessed within this many hours stay in Hot (RAM).
            Default: 24.
        warm_threshold_hours: Points accessed within this many hours stay in Warm (mmap).
            Default: 168 (7 days).
        compaction_interval_secs: Interval between automatic compaction runs (seconds).
            Default: 3600 (1 hour).
    """

    enabled: bool = False
    hot_threshold_hours: int = 24
    warm_threshold_hours: int = 168
    compaction_interval_secs: int = 3600

    def to_dict(self) -> dict:
        """Serialize to the server's JSON format."""
        d: Dict[str, Any] = {"enabled": self.enabled}
        if self.hot_threshold_hours != 24:
            d["hot_threshold_hours"] = self.hot_threshold_hours
        if self.warm_threshold_hours != 168:
            d["warm_threshold_hours"] = self.warm_threshold_hours
        if self.compaction_interval_secs != 3600:
            d["compaction_interval_secs"] = self.compaction_interval_secs
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TieredStorageConfig":
        """Create from a dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            hot_threshold_hours=data.get("hot_threshold_hours", 24),
            warm_threshold_hours=data.get("warm_threshold_hours", 168),
            compaction_interval_secs=data.get("compaction_interval_secs", 3600),
        )


@dataclass
class TierDistribution:
    """Distribution of points across storage tiers.

    Attributes:
        hot: Number of points in the Hot tier (RAM).
        warm: Number of points in the Warm tier (mmap).
        cold: Number of points in the Cold tier (disk).
        hot_memory_bytes: Estimated memory usage for the Hot tier (bytes).
        warm_memory_bytes: Estimated memory usage for the Warm tier (bytes).
        cold_memory_bytes: Estimated memory usage for the Cold tier (bytes).
    """

    hot: int
    warm: int
    cold: int
    hot_memory_bytes: int
    warm_memory_bytes: int
    cold_memory_bytes: int

    @classmethod
    def from_dict(cls, data: dict) -> "TierDistribution":
        """Create from a dictionary."""
        return cls(
            hot=data["hot"],
            warm=data["warm"],
            cold=data["cold"],
            hot_memory_bytes=data["hot_memory_bytes"],
            warm_memory_bytes=data["warm_memory_bytes"],
            cold_memory_bytes=data["cold_memory_bytes"],
        )


@dataclass
class Point:
    """Represents a vector point with ID, vector, and metadata."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    namespace: Optional[str] = None
    ttl: Optional[int] = None
    vectors: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> dict:
        """Convert point to dictionary for API requests."""
        d: Dict[str, Any] = {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
        }
        if self.namespace is not None:
            d["namespace"] = self.namespace
        if self.ttl is not None:
            d["ttl"] = self.ttl
        if self.vectors is not None:
            d["vectors"] = self.vectors
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Point":
        """Create a Point from a dictionary."""
        return cls(
            id=data["id"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace"),
            ttl=data.get("ttl"),
            vectors=data.get("vectors"),
        )


@dataclass
class Collection:
    """Represents a collection configuration."""
    
    name: str
    dimension: int
    distance: DistanceMetric
    created_at: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert collection to dictionary for API requests."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "distance": self.distance.value,
        }


@dataclass
class CollectionListItem:
    """Represents a collection in a list."""
    
    name: str
    dimension: int
    num_points: int
    created_at: int
    distance: DistanceMetric = DistanceMetric.COSINE
    
    @classmethod
    def from_dict(cls, data: dict) -> "CollectionListItem":
        """Create a CollectionListItem from a dictionary."""
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            num_points=data["num_points"],
            created_at=data["created_at"],
            distance=DistanceMetric(data["distance"]) if "distance" in data else DistanceMetric.COSINE,
        )


# ─── Collection Detail (GET /collections/{name}) ──────────────────────────


@dataclass
class CollectionStats:
    """Statistics about a collection's index."""

    index_size_bytes: int

    @classmethod
    def from_dict(cls, data: dict) -> "CollectionStats":
        return cls(index_size_bytes=data["index_size_bytes"])


@dataclass
class CollectionDetail:
    """Detailed information about a single collection."""

    name: str
    dimension: int
    num_points: int
    last_updated: int
    distance: DistanceMetric
    stats: CollectionStats

    @classmethod
    def from_dict(cls, data: dict) -> "CollectionDetail":
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            num_points=data["num_points"],
            last_updated=data["last_updated"],
            distance=DistanceMetric(data["distance"]),
            stats=CollectionStats.from_dict(data["stats"]),
        )


# ─── Point Detail (GET /collections/{name}/points/{id}) ───────────────────


@dataclass
class PointDetail:
    """Full point data including vector, metadata, and creation timestamp."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    created_at: int
    namespace: Optional[str] = None
    vectors: Optional[Dict[str, List[float]]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PointDetail":
        return cls(
            id=data["id"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
            namespace=data.get("namespace"),
            vectors=data.get("vectors"),
        )


# ─── List Points (GET /collections/{name}/points) ─────────────────────────


@dataclass
class ListPointsResult:
    """Paginated list of points."""

    points: List[PointDetail]
    total: int
    limit: int
    offset: int
    has_more: bool

    @classmethod
    def from_dict(cls, data: dict) -> "ListPointsResult":
        return cls(
            points=[PointDetail.from_dict(p) for p in data["points"]],
            total=data["total"],
            limit=data["limit"],
            offset=data["offset"],
            has_more=data["has_more"],
        )


@dataclass
class SearchResult:
    """Represents a search result."""

    id: str
    score: float
    metadata: Dict[str, Any]
    namespace: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Create a SearchResult from a dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            metadata=data["metadata"],
            namespace=data.get("namespace"),
        )


@dataclass
class UpsertResult:
    """Result of an upsert operation."""

    upserted: int
    failed: List[Dict[str, str]]

    @classmethod
    def from_dict(cls, data: dict) -> "UpsertResult":
        """Create an UpsertResult from a dictionary."""
        return cls(
            upserted=data["upserted"],
            failed=data.get("failed", []),
        )


@dataclass
class ApiKeyInfo:
    """API key metadata (without the raw key value)."""

    id: int
    name: str
    key_prefix: str
    created_at: int

    @classmethod
    def from_dict(cls, data: dict) -> "ApiKeyInfo":
        """Create an ApiKeyInfo from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            key_prefix=data["key_prefix"],
            created_at=data["created_at"],
        )


@dataclass
class CreateKeyResponse:
    """Response when creating an API key (includes raw key once)."""

    id: int
    name: str
    key: str
    key_prefix: str
    created_at: int

    @classmethod
    def from_dict(cls, data: dict) -> "CreateKeyResponse":
        """Create a CreateKeyResponse from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            key=data["key"],
            key_prefix=data["key_prefix"],
            created_at=data["created_at"],
        )


# ─── Query Cost Estimation ────────────────────────────────────────────────


@dataclass
class CostBreakdown:
    """Breakdown of individual cost components for a search query."""

    index_scan_cost: float
    filter_cost: float
    hydration_cost: float
    network_overhead: float

    @classmethod
    def from_dict(cls, data: dict) -> "CostBreakdown":
        """Create a CostBreakdown from a dictionary."""
        return cls(
            index_scan_cost=data["index_scan_cost"],
            filter_cost=data["filter_cost"],
            hydration_cost=data["hydration_cost"],
            network_overhead=data["network_overhead"],
        )


@dataclass
class QueryCostEstimate:
    """Estimated cost of a search query before execution."""

    estimated_ms: float
    confidence_range: List[float]
    estimated_memory_bytes: int
    estimated_nodes_visited: int
    is_expensive: bool
    recommendations: List[str]
    breakdown: CostBreakdown

    @classmethod
    def from_dict(cls, data: dict) -> "QueryCostEstimate":
        """Create a QueryCostEstimate from a dictionary."""
        return cls(
            estimated_ms=data["estimated_ms"],
            confidence_range=data["confidence_range"],
            estimated_memory_bytes=data["estimated_memory_bytes"],
            estimated_nodes_visited=data["estimated_nodes_visited"],
            is_expensive=data["is_expensive"],
            recommendations=data.get("recommendations", []),
            breakdown=CostBreakdown.from_dict(data["breakdown"]),
        )


@dataclass
class HistoricalLatency:
    """Historical latency percentiles for a collection."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    total_queries: int

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalLatency":
        """Create a HistoricalLatency from a dictionary."""
        return cls(
            p50_ms=data["p50_ms"],
            p95_ms=data["p95_ms"],
            p99_ms=data["p99_ms"],
            avg_ms=data["avg_ms"],
            total_queries=data["total_queries"],
        )


@dataclass
class EstimateSearchResponse:
    """Response from the search cost estimation endpoint."""

    estimate: QueryCostEstimate
    historical_latency: Optional[HistoricalLatency] = None

    @classmethod
    def from_dict(cls, data: dict) -> "EstimateSearchResponse":
        """Create an EstimateSearchResponse from a dictionary.

        The API response uses a flat structure (estimate fields are at the top level
        via serde flatten), so we need to parse accordingly.
        """
        # The estimate fields are flattened into the top-level response
        estimate = QueryCostEstimate.from_dict(data)
        historical = None
        if "historical_latency" in data and data["historical_latency"] is not None:
            historical = HistoricalLatency.from_dict(data["historical_latency"])
        return cls(estimate=estimate, historical_latency=historical)


# ─── Explain Query ─────────────────────────────────────────────────────────


@dataclass
class ConditionResult:
    """Result of evaluating a single filter condition against a point."""

    field: str
    operator: str
    expected: Any
    actual: Any
    passed: bool

    @classmethod
    def from_dict(cls, data: dict) -> "ConditionResult":
        """Create a ConditionResult from a dictionary."""
        return cls(
            field=data["field"],
            operator=data["operator"],
            expected=data["expected"],
            actual=data.get("actual"),
            passed=data["passed"],
        )


@dataclass
class FilterExplanation:
    """Detailed evaluation of filter conditions for a search result."""

    conditions: List[ConditionResult]
    passed: bool

    @classmethod
    def from_dict(cls, data: dict) -> "FilterExplanation":
        """Create a FilterExplanation from a dictionary."""
        return cls(
            conditions=[ConditionResult.from_dict(c) for c in data["conditions"]],
            passed=data["passed"],
        )


@dataclass
class ExplainResult:
    """A single search result with detailed explanation."""

    id: str
    score: float
    distance_metric: str
    raw_distance: float
    score_breakdown: Dict[str, float]
    rank_before_filter: int
    rank_after_filter: int
    filter_evaluation: Optional[FilterExplanation] = None
    similarity: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ExplainResult":
        """Create an ExplainResult from a dictionary."""
        filter_eval = None
        if "filter_evaluation" in data and data["filter_evaluation"] is not None:
            filter_eval = FilterExplanation.from_dict(data["filter_evaluation"])
        return cls(
            id=data["id"],
            score=data["score"],
            distance_metric=data["distance_metric"],
            raw_distance=data["raw_distance"],
            score_breakdown=data["score_breakdown"],
            rank_before_filter=data["rank_before_filter"],
            rank_after_filter=data["rank_after_filter"],
            filter_evaluation=filter_eval,
            similarity=data.get("similarity"),
        )


@dataclass
class IndexStats:
    """Statistics about the HNSW index at the time of search."""

    total_points: int
    hnsw_layers: int
    ef_search_used: int
    tombstones_skipped: int

    @classmethod
    def from_dict(cls, data: dict) -> "IndexStats":
        """Create an IndexStats from a dictionary."""
        return cls(
            total_points=data["total_points"],
            hnsw_layers=data["hnsw_layers"],
            ef_search_used=data["ef_search_used"],
            tombstones_skipped=data["tombstones_skipped"],
        )


@dataclass
class SearchExplanation:
    """Complete explanation of a search query's results."""

    query_vector_norm: float
    distance_metric: str
    candidates_scanned: int
    candidates_after_filter: int
    results: List[ExplainResult]
    index_stats: IndexStats

    @classmethod
    def from_dict(cls, data: dict) -> "SearchExplanation":
        """Create a SearchExplanation from a dictionary."""
        return cls(
            query_vector_norm=data["query_vector_norm"],
            distance_metric=data["distance_metric"],
            candidates_scanned=data["candidates_scanned"],
            candidates_after_filter=data["candidates_after_filter"],
            results=[ExplainResult.from_dict(r) for r in data["results"]],
            index_stats=IndexStats.from_dict(data["index_stats"]),
        )


# ─── Search Response (includes query_id) ──────────────────────────────────


@dataclass
class SearchResponse:
    """Full search response including results, timing, and optional query ID."""

    results: List[SearchResult]
    took_ms: int
    query_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResponse":
        return cls(
            results=[SearchResult.from_dict(r) for r in data["results"]],
            took_ms=data["took_ms"],
            query_id=data.get("query_id"),
        )


# ─── Delete Points Result ─────────────────────────────────────────────────


@dataclass
class DeletePointsResult:
    """Result of a delete points operation."""

    deleted: int

    @classmethod
    def from_dict(cls, data: dict) -> "DeletePointsResult":
        return cls(deleted=data["deleted"])


# ─── Reindex ─────────────────────────────────────────────────────────────────


class ReindexStatus(str, Enum):
    """Status of a background reindex job."""

    QUEUED = "Queued"
    BUILDING = "Building"
    SWAPPING = "Swapping"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class ReindexStats:
    """Statistics about a reindex operation."""

    points_processed: int
    points_total: int
    tombstones_cleaned: int
    old_index_size_bytes: int
    new_index_size_bytes: int

    @classmethod
    def from_dict(cls, data: dict) -> "ReindexStats":
        return cls(
            points_processed=data["points_processed"],
            points_total=data["points_total"],
            tombstones_cleaned=data["tombstones_cleaned"],
            old_index_size_bytes=data["old_index_size_bytes"],
            new_index_size_bytes=data["new_index_size_bytes"],
        )


@dataclass
class ReindexJob:
    """A background reindex job."""

    id: str
    collection: str
    status: ReindexStatus
    progress: float
    started_at: int
    completed_at: Optional[int]
    error: Optional[str]
    stats: ReindexStats

    @classmethod
    def from_dict(cls, data: dict) -> "ReindexJob":
        return cls(
            id=data["id"],
            collection=data["collection"],
            status=ReindexStatus(data["status"]),
            progress=data["progress"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            stats=ReindexStats.from_dict(data["stats"]),
        )


@dataclass
class StartReindexResponse:
    """Response from starting a reindex job."""

    job_id: str
    collection: str
    status: ReindexStatus
    message: str

    @classmethod
    def from_dict(cls, data: dict) -> "StartReindexResponse":
        return cls(
            job_id=data["job_id"],
            collection=data["collection"],
            status=ReindexStatus(data["status"]),
            message=data["message"],
        )


# ─── WebSocket Messages ───────────────────────────────────────────────────


@dataclass
class WsAckMessage:
    """Acknowledgement from the server after an upsert or subscribe via WebSocket."""

    upserted: int
    failed: int
    took_ms: int

    @classmethod
    def from_dict(cls, data: dict) -> "WsAckMessage":
        return cls(
            upserted=data["upserted"],
            failed=data["failed"],
            took_ms=data["took_ms"],
        )


@dataclass
class WsEventMessage:
    """Real-time event notification received via WebSocket subscription."""

    collection: str
    action: str
    point_ids: List[str]
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict) -> "WsEventMessage":
        return cls(
            collection=data["collection"],
            action=data["action"],
            point_ids=data["point_ids"],
            timestamp=data["timestamp"],
        )


@dataclass
class WsErrorMessage:
    """Error message received via WebSocket."""

    message: str
    code: int

    @classmethod
    def from_dict(cls, data: dict) -> "WsErrorMessage":
        return cls(
            message=data["message"],
            code=data["code"],
        )
