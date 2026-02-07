"""VectorDB Python SDK."""

from .client import VectorDBClient
from .realtime import RealtimeClient
from .models import (
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
    # Quantization
    ScalarType,
    ScalarQuantizationConfig,
    QuantizationConfig,
    # API keys
    ApiKeyInfo,
    CreateKeyResponse,
    # Cost estimation
    CostBreakdown,
    QueryCostEstimate,
    HistoricalLatency,
    EstimateSearchResponse,
    # Explain query
    ConditionResult,
    FilterExplanation,
    ExplainResult,
    IndexStats,
    SearchExplanation,
    # WebSocket messages
    WsAckMessage,
    WsEventMessage,
    WsErrorMessage,
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
    # Clients
    "VectorDBClient",
    "RealtimeClient",
    # Models
    "Point",
    "Collection",
    "CollectionListItem",
    "CollectionDetail",
    "CollectionStats",
    "PointDetail",
    "ListPointsResult",
    "SearchResult",
    "SearchResponse",
    "UpsertResult",
    "DeletePointsResult",
    "DistanceMetric",
    # Quantization
    "ScalarType",
    "ScalarQuantizationConfig",
    "QuantizationConfig",
    # API keys
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
    # WebSocket messages
    "WsAckMessage",
    "WsEventMessage",
    "WsErrorMessage",
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

__version__ = "0.2.0"
