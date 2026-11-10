#!/usr/bin/env python3
"""Замеры FAISS для лекции 6: точный перебор, IVF, IVF+PQ, HNSW.

Данные 1: заголовки ParaPhraser (около 14,7 тыс.), модель paraphrase-multilingual-MiniLM-L12-v2, d = 384.
Данные 2: синтетическая смесь гауссиан, d = 384, N до 200 тыс., для роста времени с размером базы.
Время запроса меряется в один поток. Полнота@10 считается против точного перебора.
Запуск: python3 scripts/ann_bench.py   (несколько минут на CPU)
"""
import os, sys, json, time, pathlib
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, faiss
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent; DATA = ROOT / "data"
K = 10

def headlines():
    path = DATA / "emb_headlines.npy"
    if path.exists():
        return np.load(path)
    texts = set()
    for f in ["paraphraser_train.jsonl", "paraphraser_test.jsonl"]:
        for line in open(DATA / f, encoding="utf-8"):
            d = json.loads(line); texts.add(d["text_1"]); texts.add(d["text_2"])
    texts = sorted(texts)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    X = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    np.save(path, X); json.dump(texts, open(DATA / "headlines.json", "w"), ensure_ascii=False)
    return X

def split(X, nq=1000, seed=0):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(X))
    return X[perm[nq:]].copy(), X[perm[:nq]].copy()

def truth(xb, xq):
    idx = faiss.IndexFlatIP(xb.shape[1]); idx.add(xb); _, I = idx.search(xq, K); return I

def recall(I, gt):
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(I, gt)]))

def timed_search(index, xq, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); _, I = index.search(xq, K); best = min(best, time.perf_counter() - t0)
    return I, best / len(xq) * 1e3  # мс на запрос

def size_mb(index):
    return len(faiss.serialize_index(index)) / 2 ** 20

def bench_headlines():
    X = headlines(); xb, xq = split(X); d = xb.shape[1]; gt = truth(xb, xq)
    faiss.omp_set_num_threads(1)
    out = {"N": int(len(xb)), "d": int(d), "nq": int(len(xq))}
    flat = faiss.IndexFlatIP(d); t0 = time.perf_counter(); flat.add(xb); tb = time.perf_counter() - t0
    I, ms = timed_search(flat, xq); out["flat"] = {"build_s": round(tb, 3), "ms": round(ms, 4), "recall": recall(I, gt), "mb": round(size_mb(flat), 1)}
    # IVF
    faiss.omp_set_num_threads(8)
    nlist = 128; q = faiss.IndexFlatIP(d); ivf = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter(); ivf.train(xb); ivf.add(xb); tb = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    out["ivf"] = {"nlist": nlist, "build_s": round(tb, 2), "mb": round(size_mb(ivf), 1), "curve": []}
    for nprobe in [1, 2, 4, 8, 16, 32, 64, 128]:
        ivf.nprobe = nprobe; I, ms = timed_search(ivf, xq)
        out["ivf"]["curve"].append({"nprobe": nprobe, "recall": round(recall(I, gt), 4), "ms": round(ms, 4)})
    # IVF + PQ
    faiss.omp_set_num_threads(8)
    q2 = faiss.IndexFlatIP(d); ivfpq = faiss.IndexIVFPQ(q2, d, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter(); ivfpq.train(xb); ivfpq.add(xb); tb = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)
    out["ivfpq"] = {"nlist": nlist, "m": 48, "nbits": 8, "build_s": round(tb, 2), "mb": round(size_mb(ivfpq), 2), "curve": []}
    for nprobe in [1, 4, 16, 64]:
        ivfpq.nprobe = nprobe; I, ms = timed_search(ivfpq, xq)
        out["ivfpq"]["curve"].append({"nprobe": nprobe, "recall": round(recall(I, gt), 4), "ms": round(ms, 4)})
    # HNSW
    out["hnsw"] = {}
    for M in [8, 16, 32]:
        faiss.omp_set_num_threads(8)
        h = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 40
        t0 = time.perf_counter(); h.add(xb); tb = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        levels = faiss.vector_to_array(h.hnsw.levels)
        rec = {"build_s": round(tb, 2), "mb": round(size_mb(h), 1), "max_level": int(levels.max() - 1),
               "per_level": {int(l): int((levels - 1 >= l).sum()) for l in range(int(levels.max()))}, "curve": []}
        for ef in [4, 8, 16, 32, 64, 128, 256]:
            h.hnsw.efSearch = ef; I, ms = timed_search(h, xq)
            rec["curve"].append({"ef": ef, "recall": round(recall(I, gt), 4), "ms": round(ms, 4)})
        out["hnsw"][M] = rec
    return out

def synthetic(n, d=384, centers=1000, seed=1):
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, (centers, d)).astype("float32")
    X = C[rng.integers(0, centers, n)] + rng.normal(0, 0.6, (n, d)).astype("float32")
    faiss.normalize_L2(X); return X

def bench_scaling():
    out = []
    for n in [10_000, 50_000, 100_000, 200_000]:
        X = synthetic(n + 500); xb, xq = X[500:], X[:500]
        faiss.omp_set_num_threads(16); gt = truth(xb, xq)
        flat = faiss.IndexFlatIP(xb.shape[1]); flat.add(xb)
        h = faiss.IndexHNSWFlat(xb.shape[1], 32, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 40
        t0 = time.perf_counter(); h.add(xb); tb = time.perf_counter() - t0
        faiss.omp_set_num_threads(1); h.hnsw.efSearch = 64
        _, msf = timed_search(flat, xq, 2); I, msh = timed_search(h, xq, 2)
        out.append({"N": n, "flat_ms": round(msf, 4), "hnsw_ms": round(msh, 4), "hnsw_recall": round(recall(I, gt), 4),
                    "hnsw_build_s": round(tb, 1), "flat_mb": round(xb.nbytes / 2 ** 20, 1), "hnsw_mb": round(size_mb(h), 1)})
        print(out[-1], flush=True)
    return out

if __name__ == "__main__":
    res = {"headlines": bench_headlines()}
    print(json.dumps({k: v for k, v in res["headlines"].items() if k != "hnsw"}, ensure_ascii=False)[:1500], flush=True)
    print({M: r["curve"] for M, r in res["headlines"]["hnsw"].items()}, flush=True)
    res["scaling"] = bench_scaling()
    json.dump(res, open(DATA / "ann_bench.json", "w"), ensure_ascii=False, indent=1)
