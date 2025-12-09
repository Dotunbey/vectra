from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from .engine import QuantizedIndex

# Global state
index = None
DIMENSION = 384  # Standard embedding size (e.g., all-MiniLM-L6-v2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    # Initialize index on startup
    index = QuantizedIndex(dim=DIMENSION)
    yield
    # Cleanup if necessary (e.g., persist to disk)
    index = None

app = FastAPI(lifespan=lifespan, title="Vectra Engine")

class IngestRequest(BaseModel):
    vectors: list[list[float]]
    ids: list[str]

class SearchRequest(BaseModel):
    vector: list[float]
    k: int = 5

@app.post("/ingest", status_code=201)
async def ingest_vectors(payload: IngestRequest):
    if len(payload.vectors) != len(payload.ids):
        raise HTTPException(400, "Count mismatch between vectors and IDs")
    
    try:
        await index.add(payload.vectors, payload.ids)
        return {"status": "indexed", "count": len(payload.ids)}
    except MemoryError:
        raise HTTPException(507, "Index at capacity")

@app.post("/search")
async def search_vectors(payload: SearchRequest):
    results = await index.search(payload.vector, k=payload.k)
    return {
        "results": [{"id": r[0], "score": round(r[1], 4)} for r in results]
    }

@app.get("/health")
async def health():
    return {"status": "online", "items": len(index.ids)}
