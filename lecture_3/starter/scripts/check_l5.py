#!/usr/bin/env python3
"""Проверка задания лекции 5: индекс, тестовая выборка, метрики, провалы. 21 пункт.

    python3 scripts/check_l5.py

Проверяется форма: файлы на месте, выборка размечена, скрипт оценки работает и его числа есть в отчёте.
Смысл (причины провалов, чтение таблицы) проверяет человек.
"""
import csv
import importlib.util
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
ok_count = 0
total = 0


def check(cond, msg):
    global ok_count, total
    total += 1
    ok_count += bool(cond)
    print(("  ok  " if cond else "  --  ") + msg)


def tokens(s):
    return set(re.findall(r"[а-яёa-z0-9]+", s.lower().replace("ё", "е")))


def main():
    kg, src, scripts = HERE / "kg", HERE / "src", HERE / "scripts"

    # ---------------- корпус и выборка
    corpus = kg / "corpus.csv"
    check(corpus.exists(), "kg/corpus.csv есть")
    rows = list(csv.DictReader(open(corpus, encoding="utf-8"))) if corpus.exists() else []
    ids = {r["id"] for r in rows}
    titles = {r["id"]: r.get("название", "") for r in rows}

    qpath = kg / "queries.csv"
    check(qpath.exists(), "kg/queries.csv есть")
    qrows = list(csv.DictReader(open(qpath, encoding="utf-8"))) if qpath.exists() else []
    check({"id", "запрос", "релевантные"} <= set(qrows[0].keys()) if qrows else False,
          "в queries.csv есть столбцы id, запрос, релевантные")
    check(len(qrows) >= 15, f"запросов не меньше 15 (сейчас {len(qrows)})")

    rel_ok, grade_ok, no_title_overlap, parsed = True, True, 0, []
    for r in qrows:
        rel = {}
        for item in (r.get("релевантные") or "").split(";"):
            if ":" not in item:
                rel_ok = False
                continue
            d, g = item.split(":")
            d, g = d.strip(), g.strip()
            if d not in ids:
                rel_ok = False
            if g not in ("1", "2"):
                grade_ok = False
            rel[d] = g
        parsed.append((r["запрос"], rel))
        qt = tokens(r["запрос"])
        if rel and all(not (qt & tokens(titles.get(d, ""))) for d in rel):
            no_title_overlap += 1
    check(rel_ok, "у каждого запроса релевантные документы записаны как id:градация и id есть в корпусе")
    check(grade_ok, "градации только 1 и 2")
    check(all(rel for _, rel in parsed), "у каждого запроса есть хотя бы один релевантный документ")
    check(no_title_overlap >= 3,
          f"не меньше трёх запросов без общих слов с названиями релевантных документов (сейчас {no_title_overlap})")
    grades = [g for _, rel in parsed for g in rel.values()]
    check("1" in grades and "2" in grades, "в разметке есть обе градации, 1 и 2")

    # ---------------- код поиска
    mod = src / "search_bm25.py"
    check(mod.exists(), "src/search_bm25.py есть")
    index_ok = pipe_ok = params_ok = False
    if mod.exists():
        spec = importlib.util.spec_from_file_location("search_bm25", mod)
        m = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(m)
            index_ok = hasattr(m, "Index") and hasattr(m, "Pipeline")
            pipe_ok = index_ok and m.Pipeline(stop=True, stem=True).stem
            params_ok = index_ok and "k1" in m.Index.bm25.__code__.co_varnames and "b" in m.Index.bm25.__code__.co_varnames
        except Exception as e:
            print("     ошибка импорта:", e)
    check(index_ok, "в модуле есть классы Index и Pipeline")
    check(pipe_ok, "конвейер со стеммингом работает (nltk установлен)")
    check(params_ok, "у bm25 есть параметры k1 и b")

    # ---------------- скрипт оценки
    ev = scripts / "eval_search.py"
    check(ev.exists(), "scripts/eval_search.py есть")
    out = ""
    if ev.exists():
        try:
            out = subprocess.run([sys.executable, str(ev)], capture_output=True, text=True, timeout=120, cwd=HERE).stdout
        except Exception as e:
            print("     ошибка запуска:", e)
    check(all(k in out for k in ("P@5", "MRR", "NDCG@10")), "скрипт оценки печатает P@5, MRR и NDCG@10")
    numbers = re.findall(r"\b0\.\d{3}\b|\b1\.000\b", out)
    check(len(numbers) >= 10, "скрипт печатает таблицу не меньше чем из двух вариантов")

    # ---------------- отчёт
    rep = kg / "SEARCH.md"
    check(rep.exists(), "kg/SEARCH.md есть")
    text = open(rep, encoding="utf-8").read() if rep.exists() else ""
    in_report = [n for n in set(numbers) if n in text]
    check(len(in_report) >= 5, f"числа из скрипта оценки присутствуют в SEARCH.md (найдено {len(in_report)})")
    causes = ("синоним", "перефраз", "формул", "связ", "конвейер", "стоп-слов", "основ")
    section = text.lower().split("провал", 1)[1] if "провал" in text.lower() else ""
    items = re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S", section)
    check(len(items) >= 3 and sum(c in section for c in causes) >= 2,
          "в SEARCH.md описаны три провала с причинами")
    check("k1" in text and ("стемм" in text.lower() or "основ" in text.lower()),
          "в SEARCH.md записаны параметры и конвейер")

    diary = HERE / "diary.md"
    dtext = open(diary, encoding="utf-8").read().lower() if diary.exists() else ""
    check(any(w in dtext for w in ("поиск", "bm25", "выборк")), "в diary.md есть запись о поиске")

    print(f"\nитого: {ok_count} из {total}")


if __name__ == "__main__":
    main()
