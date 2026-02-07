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


@dataclass
class Point:
    """Represents a vector point with ID, vector, and metadata."""
    
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert point to dictionary for API requests."""
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Point":
        """Create a Point from a dictionary."""
        return cls(
            id=data["id"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
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
    
    @classmethod
    def from_dict(cls, data: dict) -> "CollectionListItem":
        """Create a CollectionListItem from a dictionary."""
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            num_points=data["num_points"],
            created_at=data["created_at"],
        )


@dataclass
class SearchResult:
    """Represents a search result."""
    
    id: str
    score: float
    metadata: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Create a SearchResult from a dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            metadata=data["metadata"],
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
