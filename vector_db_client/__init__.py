"""VectorDB Python SDK."""

from .client import VectorDBClient
from .models import (
    Point,
    Collection,
    CollectionListItem,
    SearchResult,
    UpsertResult,
    DistanceMetric,
    ApiKeyInfo,
    CreateKeyResponse,
    CostBreakdown,
    QueryCostEstimate,
    HistoricalLatency,
    EstimateSearchResponse,
    ConditionResult,
    FilterExplanation,
    ExplainResult,
    IndexStats,
    SearchExplanation,
)
from .exceptions import (
    VectorDBError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    InvalidDimensionError,
    InvalidPayloadError,
    InternalError,
    BudgetExceededError,
    ConnectionError,
)

__all__ = [
    "VectorDBClient",
    # Models
    "Point",
    "Collection",
    "CollectionListItem",
    "SearchResult",
    "UpsertResult",
    "DistanceMetric",
    "ApiKeyInfo",
    "CreateKeyResponse",
    # Cost estimation
    "CostBreakdown",
    "QueryCostEstimate",
    "HistoricalLatency",
    "EstimateSearchResponse",
    # Explain query
    "ConditionResult",
    "FilterExplanation",
    "ExplainResult",
    "IndexStats",
    "SearchExplanation",
    # Exceptions
    "VectorDBError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "InvalidDimensionError",
    "InvalidPayloadError",
    "InternalError",
    "BudgetExceededError",
    "ConnectionError",
]

__version__ = "0.1.0"
