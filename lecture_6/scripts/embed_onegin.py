#!/usr/bin/env python3
"""Векторизация строф «Онегина» тремя моделями и сравнение с BM25F на 16 запросах занятия 5.

Модели берутся из локального кэша Hugging Face, сеть не нужна. Обучения нет, только кодирование на CPU.
Результат: data/emb_<модель>.npy, data/eval_onegin.json. Запуск: python3 scripts/embed_onegin.py
"""
import os, sys, csv, json, time, pathlib
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from search_bm25 import Pipeline, Index, load_qrels, precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k
from sentence_transformers import SentenceTransformer

MODELS = {  # короткое имя: (имя в кэше, префикс запроса, префикс документа)
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "scirus": ("mlsa-iai-msu-lab/sci-rus-tiny", "", ""),
    "bgem3": ("BAAI/bge-m3", "", ""),
}

def load():
    rows = list(csv.DictReader(open(ROOT / "data" / "onegin_stanzas.csv", encoding="utf-8")))
    return rows, load_qrels(ROOT / "data" / "queries_onegin.csv")

def metrics(rank_fn, qrels):
    m = {"P@5": 0, "R@10": 0, "MRR": 0, "NDCG@10": 0}; per = {}
    for qid, q, rel in qrels:
        r = rank_fn(q)
        m["P@5"] += precision_at_k(r, rel, 5); m["R@10"] += recall_at_k(r, rel, 10)
        m["MRR"] += reciprocal_rank(r, rel); n = ndcg_at_k(r, rel, 10); m["NDCG@10"] += n; per[qid] = round(n, 3)
    return {k: round(v / len(qrels), 3) for k, v in m.items()}, per

def main():
    rows, qrels = load(); ids = [r["id"] for r in rows]
    texts = [r["текст"].replace(" / ", "\n") for r in rows]
    idx = Index([{"название": r["название"], "текст": r["текст"]} for r in rows], Pipeline(), {"название": 2.0, "текст": 1.0}, ids)
    res = {"BM25F": metrics(lambda q: [d for d, _ in idx.bm25f(q, k=10)], qrels)}
    info = {}
    for short, (name, qp, dp) in MODELS.items():
        t0 = time.time(); m = SentenceTransformer(name, device="cpu"); tl = time.time() - t0
        t0 = time.time(); E = m.encode([dp + t for t in texts], batch_size=16, normalize_embeddings=True, show_progress_bar=False); te = time.time() - t0
        np.save(ROOT / "data" / f"emb_{short}.npy", E.astype("float32"))
        Q = {qid: m.encode([qp + q], normalize_embeddings=True)[0] for qid, q, _ in qrels}
        def rank(q, Q=Q, E=E):
            qid = [i for i, qq, _ in qrels if qq == q][0]
            s = E @ Q[qid]; return [ids[i] for i in np.argsort(-s)[:10]]
        res[short] = metrics(rank, qrels)
        tok = m.tokenizer; lens = [len(tok(t)["input_ids"]) for t in texts]
        info[short] = {"model": name, "dim": int(E.shape[1]), "max_seq_length": int(m.max_seq_length),
                       "load_s": round(tl, 1), "encode_s": round(te, 1),
                       "stanzas_truncated": int(sum(l > m.max_seq_length for l in lens)), "median_tokens": int(np.median(lens))}
        print(short, info[short], res[short][0], flush=True)
    print("BM25F", res["BM25F"][0])
    json.dump({"metrics": {k: v[0] for k, v in res.items()}, "per_query": {k: v[1] for k, v in res.items()}, "info": info},
              open(ROOT / "data" / "eval_onegin.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
