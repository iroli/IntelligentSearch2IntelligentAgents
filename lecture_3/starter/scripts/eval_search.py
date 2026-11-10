#!/usr/bin/env python3
"""Метрики поиска по словам на тестовой выборке. Запуск из папки задания:

    python3 scripts/eval_search.py                 # таблица четырёх вариантов
    python3 scripts/eval_search.py --per-query     # и таблица по запросам для BM25F
    python3 scripts/eval_search.py --k1 1.2 --b 0.5

Читает kg/corpus.csv (поля «название» и «формулировка») и kg/queries.csv.
Печатает точность@5, полноту@5, MRR, MAP, NDCG@10 для двух конвейеров и двух ранжирований.
Числа из этой таблицы переносятся в kg/SEARCH.md без правок.
"""
import argparse
import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
from search_bm25 import Pipeline, Index, load_qrels, evaluate, precision_at_k, reciprocal_rank, ndcg_at_k  # noqa: E402

FIELDS = {"название": 2.0, "формулировка": 1.0}


def read_corpus(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    docs = [{f: r.get(f, "") for f in FIELDS} for r in rows]
    return docs, [r["id"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=HERE / "kg" / "corpus.csv")
    ap.add_argument("--queries", default=HERE / "kg" / "queries.csv")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--per-query", action="store_true")
    a = ap.parse_args()

    docs, ids = read_corpus(a.corpus)
    qrels = load_qrels(a.queries)
    full = Index(docs, Pipeline(stop=True, stem=True), FIELDS, ids)
    raw = Index(docs, Pipeline(stop=False, stem=False), FIELDS, ids)
    variants = {
        "BM25, без нормализации": lambda q: raw.bm25(q, k=10, k1=a.k1, b=a.b),
        "BM25, стоп-слова и основы": lambda q: full.bm25(q, k=10, k1=a.k1, b=a.b),
        "BM25F, без нормализации": lambda q: raw.bm25f(q, k=10, k1=a.k1, b=a.b),
        "BM25F, стоп-слова и основы": lambda q: full.bm25f(q, k=10, k1=a.k1, b=a.b),
    }
    print(f"корпус: {len(docs)} документов, выборка: {len(qrels)} запросов, k1 = {a.k1}, b = {a.b}")
    print(f"{'вариант':30} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'MAP':>6} {'NDCG@10':>8}")
    for name, fn in variants.items():
        m = evaluate(fn, qrels)
        print(f"{name:30} {m['P@5']:6.3f} {m['R@5']:6.3f} {m['MRR']:6.3f} {m['MAP']:6.3f} {m['NDCG@10']:8.3f}")
    if a.per_query:
        print(f"\nBM25F по запросам:\n{'id':5} {'P@5':>5} {'RR':>5} {'NDCG@10':>8}  первые пять")
        for qid, q, rel in qrels:
            ranked = [d for d, _ in full.bm25f(q, k=10, k1=a.k1, b=a.b)]
            print(f"{qid:5} {precision_at_k(ranked, rel, 5):5.2f} {reciprocal_rank(ranked, rel):5.2f} "
                  f"{ndcg_at_k(ranked, rel, 10):8.2f}  {' '.join(ranked[:5]) or '—'}")


if __name__ == "__main__":
    main()
