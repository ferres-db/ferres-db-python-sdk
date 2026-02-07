"""VectorDB client for Python."""

import asyncio
import time
import json
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin

import httpx
import structlog

from .models import (
    Point,
    Collection,
    CollectionListItem,
    CollectionDetail,
    PointDetail,
    ListPointsResult,
    SearchResult,
    SearchResponse,
    UpsertResult,
    DeletePointsResult,
    DistanceMetric,
    QuantizationConfig,
    ApiKeyInfo,
    CreateKeyResponse,
    EstimateSearchResponse,
    QueryCostEstimate,
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

# Configure structlog
logger = structlog.get_logger()


class VectorDBClient:
    """Client for interacting with FerresDB vector database."""
    
    # Maximum batch size for upsert operations
    MAX_BATCH_SIZE = 1000
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the VectorDB client.

        Args:
            base_url: Base URL of the FerresDB server (e.g., "http://localhost:8080")
            api_key: Optional API key for authentication (Authorization: Bearer <key>).
                     Required for all data routes (collections, points, keys).
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )
        self.logger = logger.bind(client="VectorDBClient")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _handle_error(self, response: httpx.Response) -> None:
        """Handle API errors and raise appropriate exceptions."""
        try:
            error_data = await response.json()
            error_type = error_data.get("error", "unknown")
            message = error_data.get("message", "Unknown error")
            code = error_data.get("code", response.status_code)
        except Exception:
            error_type = "unknown"
            message = response.text or f"HTTP {response.status_code}"
            code = response.status_code
        
        error_map = {
            "collection_not_found": CollectionNotFoundError,
            "collection_already_exists": CollectionAlreadyExistsError,
            "invalid_dimension": InvalidDimensionError,
            "invalid_payload": InvalidPayloadError,
            "internal_error": InternalError,
        }
        
        # Handle budget_exceeded specially (includes estimate in body)
        if error_type == "budget_exceeded":
            estimate = error_data.get("estimate", {}) if isinstance(error_data, dict) else {}
            raise BudgetExceededError(message, estimate=estimate)
        
        error_class = error_map.get(error_type, VectorDBError)
        
        if error_class == CollectionNotFoundError:
            # Try to extract collection name from message
            if "collection '" in message:
                collection_name = message.split("collection '")[1].split("'")[0]
                raise CollectionNotFoundError(collection_name)
        
        if error_class == CollectionAlreadyExistsError:
            if "collection '" in message:
                collection_name = message.split("collection '")[1].split("'")[0]
                raise CollectionAlreadyExistsError(collection_name)
        
        raise error_class(message)
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[dict] = None,
        max_retries: int = 3,
    ) -> httpx.Response:
        """
        Make an HTTP request with automatic retry and exponential backoff.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path (e.g., "/api/v1/collections")
            json_data: Optional JSON payload
            max_retries: Maximum number of retry attempts
        
        Returns:
            HTTP response
        
        Raises:
            ConnectionError: If connection fails after retries
            VectorDBError: For API errors
        """
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(
                    "making_request",
                    method=method,
                    url=url,
                    attempt=attempt + 1,
                )
                
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=json_data,
                )
                
                # Success
                if response.is_success:
                    self.logger.debug(
                        "request_success",
                        method=method,
                        url=url,
                        status_code=response.status_code,
                    )
                    return response
                
                # Client errors (4xx) should not be retried
                if 400 <= response.status_code < 500:
                    await self._handle_error(response)
                
                # Server errors (5xx) should be retried
                if response.status_code >= 500:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(
                            "server_error_retrying",
                            status_code=response.status_code,
                            attempt=attempt + 1,
                            wait_time=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        await self._handle_error(response)
                
                # Other errors
                await self._handle_error(response)
                
            except httpx.TimeoutException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        "timeout_retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise ConnectionError(f"Request timeout after {max_retries + 1} attempts: {e}")
            
            except httpx.RequestError as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        "connection_error_retrying",
                        error=str(e),
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise ConnectionError(f"Connection error after {max_retries + 1} attempts: {e}")
        
        raise ConnectionError(f"Request failed after {max_retries + 1} attempts")
    
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: DistanceMetric,
        enable_bm25: Optional[bool] = None,
        bm25_text_field: Optional[str] = None,
        quantization: Optional[QuantizationConfig] = None,
    ) -> Collection:
        """
        Create a new collection.

        Args:
            name: Collection name (alphanumeric, hyphens, underscores only)
            dimension: Vector dimension (1-4096)
            distance: Distance metric to use
            enable_bm25: Optional; enable BM25 index for hybrid search (default: False)
            bm25_text_field: Optional; metadata key used as text for BM25 (default: "text")
            quantization: Optional; quantization config (use ``QuantizationConfig.scalar_int8()``
                to enable SQ8 compression with ~4x memory savings)

        Returns:
            Created collection

        Raises:
            CollectionAlreadyExistsError: If collection already exists
            InvalidDimensionError: If dimension is invalid
            InvalidPayloadError: If payload is invalid
        """
        payload: Dict[str, Any] = {
            "name": name,
            "dimension": dimension,
            "distance": distance.value,
        }
        if enable_bm25 is not None:
            payload["enable_bm25"] = enable_bm25
        if bm25_text_field is not None:
            payload["bm25_text_field"] = bm25_text_field
        if quantization is not None:
            payload["quantization"] = quantization.to_dict()

        response = await self._request("POST", "/api/v1/collections", json_data=payload)
        data = await response.json()

        return Collection(
            name=data["name"],
            dimension=data["dimension"],
            distance=DistanceMetric(data["distance"]),
            created_at=data.get("created_at"),
        )
    
    async def list_collections(self) -> List[CollectionListItem]:
        """
        List all collections.
        
        Returns:
            List of collections
        """
        response = await self._request("GET", "/api/v1/collections")
        data = await response.json()
        
        return [
            CollectionListItem.from_dict(item)
            for item in data["collections"]
        ]
    
    async def get_collection(self, name: str) -> CollectionDetail:
        """
        Get detailed information about a single collection.

        Args:
            name: Collection name

        Returns:
            CollectionDetail with dimension, num_points, last_updated, distance, and stats.

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        response = await self._request("GET", f"/api/v1/collections/{name}")
        data = await response.json()
        return CollectionDetail.from_dict(data)

    async def delete_collection(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name
        
        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        await self._request("DELETE", f"/api/v1/collections/{name}")
    
    async def upsert_points(
        self,
        collection: str,
        points: List[Point],
    ) -> UpsertResult:
        """
        Upsert points into a collection.
        
        Automatically batches points if the list exceeds MAX_BATCH_SIZE (1000).
        
        Args:
            collection: Collection name
            points: List of points to upsert
        
        Returns:
            Upsert result with upserted count and failed points
        
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            InvalidDimensionError: If vector dimensions don't match
        """
        if not points:
            return UpsertResult(upserted=0, failed=[])
        
        # Batch points if necessary
        if len(points) <= self.MAX_BATCH_SIZE:
            return await self._upsert_batch(collection, points)
        
        # Split into batches
        total_upserted = 0
        all_failed = []
        
        for i in range(0, len(points), self.MAX_BATCH_SIZE):
            batch = points[i:i + self.MAX_BATCH_SIZE]
            result = await self._upsert_batch(collection, batch)
            total_upserted += result.upserted
            all_failed.extend(result.failed)
        
        return UpsertResult(upserted=total_upserted, failed=all_failed)
    
    async def _upsert_batch(
        self,
        collection: str,
        points: List[Point],
    ) -> UpsertResult:
        """Upsert a single batch of points."""
        payload = {
            "points": [point.to_dict() for point in points],
        }
        
        response = await self._request(
            "POST",
            f"/api/v1/collections/{collection}/points",
            json_data=payload,
        )
        data = await response.json()
        
        return UpsertResult.from_dict(data)
    
    async def delete_points(
        self,
        collection: str,
        ids: List[str],
    ) -> DeletePointsResult:
        """
        Delete points from a collection by IDs.
        
        Args:
            collection: Collection name
            ids: List of point IDs to delete
        
        Returns:
            DeletePointsResult with the number of points actually deleted.
        
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            InvalidPayloadError: If ids list is empty
        """
        if not ids:
            raise InvalidPayloadError("ids cannot be empty")
        
        payload = {"ids": ids}
        
        response = await self._request(
            "DELETE",
            f"/api/v1/collections/{collection}/points",
            json_data=payload,
        )
        data = await response.json()
        return DeletePointsResult.from_dict(data)
    
    async def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        budget_ms: Optional[int] = None,
    ) -> SearchResponse:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection: Collection name
            vector: Query vector
            limit: Maximum number of results (default: 10)
            filter: Optional metadata filter (equality matching)
            budget_ms: Optional latency budget in milliseconds. If the estimated
                cost exceeds this budget, the server returns 422 without executing
                the search. Catch ``BudgetExceededError`` for the detailed estimate.
        
        Returns:
            SearchResponse with results, took_ms, and optional query_id.
        
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            InvalidDimensionError: If vector dimension doesn't match collection
            BudgetExceededError: If budget_ms is set and estimated cost exceeds it
        """
        payload: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
        }
        
        if filter is not None:
            payload["filter"] = filter
        
        if budget_ms is not None:
            payload["budget_ms"] = budget_ms
        
        response = await self._request(
            "POST",
            f"/api/v1/collections/{collection}/search",
            json_data=payload,
        )
        data = await response.json()
        
        return SearchResponse.from_dict(data)

    async def list_keys(self) -> List[ApiKeyInfo]:
        """
        List API keys (metadata only; requires valid API key with Editor/Admin role).

        Returns:
            List of API key info (id, name, key_prefix, created_at).
        """
        response = await self._request("GET", "/api/v1/keys")
        data = await response.json()
        return [ApiKeyInfo.from_dict(item) for item in data]

    async def create_key(self, name: str) -> CreateKeyResponse:
        """
        Create a new API key. The raw key is returned only once; store it securely.

        Args:
            name: Display name for the key (trimmed).

        Returns:
            CreateKeyResponse including the raw `key` (only time it is returned).

        Raises:
            InvalidPayloadError: If name is empty after trimming.
        """
        trimmed = name.strip()
        if not trimmed:
            raise InvalidPayloadError("name is required")
        response = await self._request(
            "POST", "/api/v1/keys", json_data={"name": trimmed}
        )
        data = await response.json()
        return CreateKeyResponse.from_dict(data)

    async def delete_key(self, key_id: int) -> None:
        """
        Delete an API key by id.

        Args:
            key_id: Numeric id of the key (from list_keys or create_key).
        """
        await self._request("DELETE", f"/api/v1/keys/{key_id}")

    # ─── Query Cost Estimation ──────────────────────────────────────────────

    async def estimate_search_cost(
        self,
        collection: str,
        limit: int,
        filter: Optional[Dict[str, Any]] = None,
        include_history: bool = False,
    ) -> EstimateSearchResponse:
        """
        Estimate the cost of a search query **before** executing it.

        Returns estimated latency, memory consumption, HNSW nodes visited,
        whether the query is "expensive", and optimisation recommendations.

        Args:
            collection: Collection name
            limit: Number of results that would be requested
            filter: Optional metadata filter (same format as search)
            include_history: If True, include historical latency percentiles
                (p50/p95/p99/avg) in the response.

        Returns:
            EstimateSearchResponse with cost estimate and optional history.

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        payload: Dict[str, Any] = {"limit": limit}
        if filter is not None:
            payload["filter"] = filter
        if include_history:
            payload["include_history"] = True

        response = await self._request(
            "POST",
            f"/api/v1/collections/{collection}/search/estimate",
            json_data=payload,
        )
        data = await response.json()

        return EstimateSearchResponse.from_dict(data)

    # ─── Explain Query ──────────────────────────────────────────────────────

    async def search_explain(
        self,
        collection: str,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchExplanation:
        """
        Search with a detailed explanation of each result.

        Returns **why** each result was returned (or filtered): score
        breakdown, per-condition filter evaluation, ranking before/after
        filters, and HNSW index statistics.

        Args:
            collection: Collection name
            vector: Query vector
            limit: Maximum number of results (default: 10)
            filter: Optional metadata filter

        Returns:
            SearchExplanation with full breakdown per result.

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            InvalidDimensionError: If vector dimension doesn't match collection
        """
        payload: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
        }
        if filter is not None:
            payload["filter"] = filter

        response = await self._request(
            "POST",
            f"/api/v1/collections/{collection}/search/explain",
            json_data=payload,
        )
        data = await response.json()

        return SearchExplanation.from_dict(data)

    # ─── Hybrid Search ─────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        collection: str,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        alpha: float = 0.5,
    ) -> SearchResponse:
        """
        Perform a hybrid search combining BM25 keyword matching and vector similarity.

        Requires the collection to have been created with ``enable_bm25=True``.

        Args:
            collection: Collection name (must have BM25 enabled)
            query_text: Text query for BM25 keyword matching
            query_vector: Query vector for similarity search
            limit: Maximum number of results (default: 10)
            alpha: Weight for vector search score (0.0-1.0). ``1-alpha`` is used
                for the BM25 keyword score. Default: 0.5.

        Returns:
            SearchResponse with combined results, took_ms, and optional query_id.

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            InvalidDimensionError: If vector dimension doesn't match collection
        """
        payload: Dict[str, Any] = {
            "query_text": query_text,
            "query_vector": query_vector,
            "limit": limit,
            "alpha": alpha,
        }

        response = await self._request(
            "POST",
            f"/api/v1/collections/{collection}/search/hybrid",
            json_data=payload,
        )
        data = await response.json()

        return SearchResponse.from_dict(data)

    # ─── Point Operations ──────────────────────────────────────────────────

    async def get_point(
        self,
        collection: str,
        point_id: str,
    ) -> PointDetail:
        """
        Get a single point by ID, including its vector and metadata.

        Args:
            collection: Collection name
            point_id: Unique point ID

        Returns:
            PointDetail with id, vector, metadata, and created_at.

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        response = await self._request(
            "GET",
            f"/api/v1/collections/{collection}/points/{point_id}",
        )
        data = await response.json()
        return PointDetail.from_dict(data)

    async def list_points(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> ListPointsResult:
        """
        List points in a collection with pagination.

        Args:
            collection: Collection name
            limit: Maximum number of points per page (default: 100, max: 1000)
            offset: Number of points to skip (default: 0)
            filter: Optional metadata filter as a dict (will be JSON-encoded
                as a query parameter)

        Returns:
            ListPointsResult with points, total, limit, offset, has_more.

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if filter is not None:
            params["filter"] = json.dumps(filter)

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        response = await self._request(
            "GET",
            f"/api/v1/collections/{collection}/points?{query_string}",
        )
        data = await response.json()
        return ListPointsResult.from_dict(data)
