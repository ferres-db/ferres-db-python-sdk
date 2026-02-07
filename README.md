# FerresDB Python SDK

Python SDK for interacting with FerresDB vector database.

## Installation

```bash
pip install ferres-db-python
```

Or install from source:

```bash
cd sdk/python
pip install -e .
```

## Authentication

All data routes (collections, points, search, API keys) require authentication. Pass the **API key** when creating the client; the SDK sends the `Authorization: Bearer <api_key>` header on every request.

```python
client = VectorDBClient(
    base_url="http://localhost:8080",
    api_key="ferres_sk_...",  # required for protected routes
)
```

Without `api_key`, the server will respond with 401 on protected routes.

## Running FerresDB with Docker

To use the SDK against a real FerresDB instance, you can run the official images.

### Pull images

```bash
docker pull ferresdb/ferres-db-core
docker pull ferresdb/ferres-db-frontend
```

### Start the backend (API)

```bash
docker run -d \
  --name ferres-db-core \
  -p 8080:8080 \
  -e PORT=8080 \
  -e STORAGE_PATH=/data \
  -e FERRESDB_API_KEYS=ferres_sk_your_key_here \
  -v ferres-data:/data \
  ferresdb/ferres-db-core
```

- **API:** http://localhost:8080

### Start the frontend (dashboard)

```bash
docker run -d \
  --name ferres-db-frontend \
  -p 3000:80 \
  -e VITE_API_BASE_URL=http://localhost:8080 \
  -e VITE_API_KEY=ferres_sk_your_key_here \
  ferresdb/ferres-db-frontend
```

- **Dashboard:** http://localhost:3000

### Use the SDK

With the backend running at `http://localhost:8080` and the same API key:

```python
from vector_db_client import VectorDBClient

client = VectorDBClient(
    base_url="http://localhost:8080",
    api_key="ferres_sk_your_key_here",
)
# create collections, upsert, search, etc.
```

## Quick Start

```python
import asyncio
from vector_db_client import VectorDBClient, Point, DistanceMetric

async def main():
    # Create client (api_key required for collections, points, etc.)
    async with VectorDBClient(
        base_url="http://localhost:8080",
        api_key="ferres_sk_...",
    ) as client:
        # Create a collection
        collection = await client.create_collection(
            name="my-collection",
            dimension=128,
            distance=DistanceMetric.COSINE,
        )

        # Upsert points
        points = [
            Point(id="1", vector=[0.1, 0.2, 0.3], metadata={"text": "hello"}),
            Point(id="2", vector=[0.4, 0.5, 0.6], metadata={"text": "world"}),
        ]
        result = await client.upsert_points("my-collection", points)
        print(f"Upserted {result.upserted} points")

        # Search for similar vectors
        results = await client.search(
            collection="my-collection",
            vector=[0.1, 0.2, 0.3],
            limit=10,
        )
        for result in results:
            print(f"ID: {result.id}, Score: {result.score}")

asyncio.run(main())
```

## Features

- **Type hints**: Full type annotations for better IDE support
- **Automatic retry**: Exponential backoff for transient failures
- **Structured logging**: Uses structlog for better observability
- **Automatic batching**: Large upsert operations are automatically split into batches
- **Async/await**: Built on httpx for async operations

## API Reference

### VectorDBClient

#### `__init__(base_url: str, api_key: str = None, timeout: int = 30)`

Initialize the client.

- `base_url`: Base URL of the FerresDB server (e.g., "http://localhost:8080")
- `api_key`: Optional API key for authentication (recommended for all data routes)
- `timeout`: Request timeout in seconds

#### `create_collection(name: str, dimension: int, distance: DistanceMetric, enable_bm25: bool = None, bm25_text_field: str = None) -> Collection`

Create a new collection. Use `enable_bm25=True` and `bm25_text_field="content"` for hybrid search.

#### `list_collections() -> List[CollectionListItem]`

List all collections.

#### `list_keys() -> List[ApiKeyInfo]`

List API keys (metadata only; requires Editor/Admin). Returns `id`, `name`, `key_prefix`, `created_at`.

#### `create_key(name: str) -> CreateKeyResponse`

Create a new API key. The raw `key` is returned only once; store it securely.

#### `delete_key(key_id: int) -> None`

Delete an API key by id (from `list_keys` or `create_key`).

#### `delete_collection(name: str) -> None`

Delete a collection.

#### `upsert_points(collection: str, points: List[Point]) -> UpsertResult`

Upsert points into a collection. Automatically batches if more than 1000 points.

#### `delete_points(collection: str, ids: List[str]) -> None`

Delete points by IDs.

#### `search(collection: str, vector: List[float], limit: int = 10, filter: dict = None) -> List[SearchResult]`

Search for similar vectors.

## Models

### Point

- `id: str`: Point identifier
- `vector: List[float]`: Vector coordinates
- `metadata: Dict[str, Any]`: Arbitrary metadata

### Collection

- `name: str`: Collection name
- `dimension: int`: Vector dimension
- `distance: DistanceMetric`: Distance metric

### SearchResult

- `id: str`: Point ID
- `score: float`: Similarity score
- `metadata: Dict[str, Any]`: Point metadata

### ApiKeyInfo

- `id: int`: Key id
- `name: str`: Display name
- `key_prefix: str`: Prefix (raw key never returned in list)
- `created_at: int`: Unix timestamp

### CreateKeyResponse

- `id: int`, `name: str`, `key: str`, `key_prefix: str`, `created_at: int` — `key` is the raw secret (returned only on create).

## Exceptions

- `VectorDBError`: Base exception
- `CollectionNotFoundError`: Collection not found (404)
- `CollectionAlreadyExistsError`: Collection already exists (409)
- `InvalidDimensionError`: Invalid dimension (400)
- `InvalidPayloadError`: Invalid payload (400)
- `InternalError`: Internal server error (500)
- `ConnectionError`: Connection error

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black vector_db_client tests
```

Type checking:

```bash
mypy vector_db_client
```

## gRPC API

O FerresDB também oferece uma API gRPC nativa (porta 50051 por padrão) como alternativa à API REST. Este SDK usa a API REST; para usar gRPC diretamente, gere stubs de cliente a partir do arquivo `proto/ferresdb.proto` no repositório do servidor (requer `--features grpc` no build do server).

Para gerar stubs gRPC em Python:

```bash
pip install grpcio-tools
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/ferresdb.proto
```
