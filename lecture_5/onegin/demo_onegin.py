#!/usr/bin/env python3
"""«Евгений Онегин» как корпус для поиска по словам. Живой пример лекции 5.

360 строф романа (восемь глав и два письма) суть документы, запросы суть строки, имена, вопросы о событиях
и описания по смыслу. Запуск из папки lecture5/onegin:

    python3 demo_onegin.py              # весь показ: закон Ципфа, запросы, метрики, гонка индекса с перебором
    python3 demo_onegin.py --live       # свои запросы: ввод запроса, выдача с вкладами слов, пустая строка завершает
    python3 demo_onegin.py --zipf       # только рисунок img/zipf.png

Нужны pandas, matplotlib, nltk и модуль ../search_bm25.py.
"""
import collections
import csv
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from search_bm25 import Pipeline, Index, tokenize, load_qrels, evaluate, precision_at_k, reciprocal_rank, ndcg_at_k  # noqa: E402

FIELDS = {"название": 2.0, "текст": 1.0}


def load():
    rows = list(csv.DictReader(open(HERE / "data" / "onegin_stanzas.csv", encoding="utf-8")))
    docs = [{"название": r["название"], "текст": r["текст"]} for r in rows]
    ids = [r["id"] for r in rows]
    text = {r["id"]: r["текст"] for r in rows}
    return rows, docs, ids, text


def show(idx, text, q, k=5, explain=True):
    print(f"\nзапрос: {q}\n  основы: {idx.pipe(q)}")
    out = idx.bm25f(q, k=k)
    if not out:
        print("  выдача пуста: ни одной общей основы")
    for d, s in out:
        first = text[d].split(" / ")[0]
        print(f"  {s:6.2f}  {d:9} {first[:60]}")
        if explain:
            parts = idx.bm25_score(q, idx.ids.index(d), explain=True)
            print("          вклад:", ", ".join(f"{w} {v:.2f}" for w, v in sorted(parts.items(), key=lambda x: -x[1])))


def zipf(docs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = collections.Counter(w for d in docs for w in tokenize(d["текст"]))
    total, uniq = sum(c.values()), len(c)
    hapax = sum(1 for v in c.values() if v == 1)
    freqs = sorted(c.values(), reverse=True)
    print(f"закон Ципфа: {total} слов, {uniq} словоформ, {hapax} встречаются один раз ({100 * hapax / uniq:.0f} %)")
    print("самые частые:", ", ".join(f"{w} {n}" for w, n in c.most_common(10)))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(range(1, len(freqs) + 1), freqs, ".", ms=3, color="#2346c8")
    ax.loglog([1, len(freqs)], [freqs[0], freqs[0] / len(freqs)], "--", color="gray", lw=1, label="частота ~ 1/ранг")
    ax.set_xlabel("ранг слова"); ax.set_ylabel("частота"); ax.set_title("«Евгений Онегин»: 360 строф, 24 тыс. слов", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    (HERE / "img").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(HERE / "img" / "zipf.png", dpi=160)
    print("рисунок: img/zipf.png")


def race(docs, idx, queries):
    """Перебор строк против индекса на корпусе, размноженном в 100 раз: 36 000 документов."""
    big = [d["текст"] for d in docs] * 100
    big_docs = [{"название": "", "текст": t} for t in big]
    t0 = time.time(); big_idx = Index(big_docs, Pipeline(), {"текст": 1.0}); t_build = time.time() - t0
    print(f"\nгонка: {len(big)} документов, индекс построен за {t_build:.1f} с")
    for q in queries:
        words = [w for w in tokenize(q)]
        t0 = time.time(); n_scan = sum(1 for t in big if all(w in t.lower() for w in words)); t_scan = time.time() - t0
        t0 = time.time(); n_idx = len(big_idx.boolean_and(q)); t_idx = time.time() - t0
        t0 = time.time(); big_idx.bm25(q, k=5); t_rank = time.time() - t0
        print(f"  {q!r:35} перебор {t_scan * 1000:7.1f} мс ({n_scan})   индекс И {t_idx * 1000:6.1f} мс ({n_idx})   BM25 {t_rank * 1000:6.1f} мс")


def main():
    rows, docs, ids, text = load()
    print(f"корпус: {len(docs)} строф, глав {len({r['глава'] for r in rows})}")
    idx = Index(docs, Pipeline(), FIELDS, ids)
    raw = Index(docs, Pipeline(stop=False, stem=False), FIELDS, ids)
    print(f"словарь: {len(raw.vocab)} словоформ, {len(idx.vocab)} основ, средняя длина строфы {idx.avgdl:.0f} слов")

    if "--zipf" in sys.argv:
        zipf(docs); return
    if "--live" in sys.argv:
        print("свои запросы (пустая строка завершает):")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            show(idx, text, q)
        return

    zipf(docs)
    print("\n=== 1. Точная строка находится всегда")
    for q in ("мой дядя самых честных правил", "Любви все возрасты покорны"):
        show(idx, text, q, k=3, explain=False)
    print("\n=== 2. Имя и событие: слова есть в строфах")
    for q in ("кто такой Зарецкий", "дуэль Онегина и Ленского"):
        show(idx, text, q, k=5, explain=True)
    print("\n=== 3. Ловушка конвейера: основа стоп-слова")
    q = "кто такой Зарецкий"
    print(f"  {q!r}: стоп-слова убраны до стемминга → {Pipeline()(q)}; «такой» дал основу «так», а она и есть стоп-слово")
    print(f"  с отбрасыванием основ стоп-слов → {Pipeline(stop_stems=True)(q)}")
    idx2 = Index(docs, Pipeline(stop_stems=True), FIELDS, ids)
    for label, ix in (("до:   ", idx), ("после:", idx2)):
        ranked = [d for d, _ in ix.bm25f(q, k=10)]
        pos = ranked.index("6.IV") + 1 if "6.IV" in ranked else None
        print(f"  выдача {label} {ranked[:5]}   6.IV «Зарецкий, некогда буян» на позиции {pos}")
    print("  Основа «так» весила больше имени: она редка среди основ, потому что слово «так» обычно убирается как стоп-слово.")
    print("\n=== 4. Смысл без слов: синонимия, перефразировка, вопрос о сюжете")
    for q in ("имя героини романа", "описание осени", "ответ Онегина на письмо Татьяны"):
        show(idx, text, q, k=5, explain=False)

    qrels = load_qrels(HERE / "data" / "queries_onegin.csv")
    print(f"\n=== 5. Метрики на {len(qrels)} запросах")
    variants = {
        "TF-IDF, косинус": lambda q: idx.tfidf(q, k=10),
        "BM25": lambda q: idx.bm25(q, k=10),
        "BM25F, название ×2": lambda q: idx.bm25f(q, k=10),
        "BM25 без стоп-слов и основ": lambda q: raw.bm25(q, k=10),
    }
    print(f"{'вариант':30} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'MAP':>6} {'NDCG@10':>8}")
    for name, fn in variants.items():
        m = evaluate(fn, qrels)
        print(f"{name:30} {m['P@5']:6.3f} {m['R@5']:6.3f} {m['MRR']:6.3f} {m['MAP']:6.3f} {m['NDCG@10']:8.3f}")
    print("\nBM25F по запросам:")
    for qid, q, rel in qrels:
        ranked = [d for d, _ in idx.bm25f(q, k=10)]
        print(f"  {qid} {precision_at_k(ranked, rel, 5):4.2f} {reciprocal_rank(ranked, rel):4.2f} {ndcg_at_k(ranked, rel, 10):5.2f}  {q[:34]:34} → {' '.join(ranked[:4]) or '—'}")

    race(docs, idx, ["дядя правил", "Татьяна письмо", "Москва"])


if __name__ == "__main__":
    main()
