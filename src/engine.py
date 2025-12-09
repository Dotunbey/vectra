import asyncio
import numpy as np
from typing import List, Tuple, Optional

class QuantizedIndex:
    __slots__ = ('dim', 'vectors', 'ids', '_lock')

    def __init__(self, dim: int, capacity: int = 100_000):
        self.dim = dim
        # Pre-allocate memory for performance (int8 for quantization)
        self.vectors = np.zeros((capacity, dim), dtype=np.int8)
        self.ids = [] 
        self._lock = asyncio.Lock()

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        # Avoid division by zero
        return v / (norm + 1e-10)

    def _quantize(self, v: np.ndarray) -> np.ndarray:
        # Scalar quantization: float32 [-1.0, 1.0] -> int8 [-127, 127]
        return (v * 127).astype(np.int8)

    async def add(self, vectors: List[List[float]], ids: List[str]):
        """Async ingestion with thread-safe locking."""
        arr = np.array(vectors, dtype=np.float32)
        
        # CPU-bound math ops offloaded to prevent blocking event loop
        loop = asyncio.get_running_loop()
        normalized = await loop.run_in_executor(None, self._normalize, arr)
        quantized = await loop.run_in_executor(None, self._quantize, normalized)

        async with self._lock:
            start_idx = len(self.ids)
            end_idx = start_idx + len(ids)
            
            if end_idx > len(self.vectors):
                raise MemoryError("Index capacity exceeded.")

            self.vectors[start_idx:end_idx] = quantized
            self.ids.extend(ids)

    async def search(self, query: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Performs approximate dot product search over quantized vectors.
        Complexity: O(N) vectorized.
        """
        q_arr = np.array([query], dtype=np.float32)
        
        loop = asyncio.get_running_loop()
        # Normalize query but keep as float for higher precision during dot product
        q_norm = await loop.run_in_executor(None, self._normalize, q_arr)
        
        # Vectorized dot product: (1, D) @ (N, D).T -> (1, N)
        # Using float32 for accumulation to prevent overflow
        scores = np.dot(q_norm, self.vectors[:len(self.ids)].T.astype(np.float32) / 127.0)
        
        # Get top-k indices
        top_k_idxs = np.argsort(scores[0])[-k:][::-1]
        
        return [(self.ids[i], float(scores[0][i])) for i in top_k_idxs]
