import requests
import numpy as np
import time

BASE_URL = "http://127.0.0.1:8000"
DIM = 384

def test_vectra():
    print(f"🚀 Connecting to Vectra Engine...")

    # 1. Create Dummy Data
    # We create 1000 random vectors
    data_size = 1000
    vectors = np.random.rand(data_size, DIM).tolist()
    ids = [f"doc_{i}" for i in range(data_size)]

    # 2. Ingest
    start = time.time()
    resp = requests.post(f"{BASE_URL}/ingest", json={"vectors": vectors, "ids": ids})
    print(f"✅ Ingested {data_size} vectors in {time.time()-start:.3f}s. Response: {resp.json()}")

    # 3. Search
    query = np.random.rand(DIM).tolist()
    start = time.time()
    resp = requests.post(f"{BASE_URL}/search", json={"vector": query, "k": 3})
    results = resp.json()['results']
    
    print(f"\n🔍 Search complete in {time.time()-start:.3f}s")
    print("Top 3 Matches:")
    for r in results:
        print(f" - ID: {r['id']} | Similarity: {r['score']}")

if __name__ == "__main__":
    test_vectra()