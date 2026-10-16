#!/usr/bin/env python3
"""Проверка критериев ДЗ1: корпус утверждений и README с пятью вопросами.

Запуск из корня репозитория:
    python3 scripts/check_hw1.py                      # kg/corpus.csv и README.md
    python3 scripts/check_hw1.py --corpus path.csv    # другой корпус
    python3 scripts/check_hw1.py --readme path.md     # другой README
Код возврата 0 — все пункты зачтены, 1 — есть незачтённые.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REQUIRED_COLUMNS = ["id", "тип", "название", "формулировка", "зависит_от", "источник"]
MIN_ROWS = 30
MIN_WITH_DEPS = 10
QUESTION_TYPES = {"факт": r"\(1\)|\bфакт", "зависимость": r"\(2\)|\bзависимост", "смысл": r"\(3\)|\bсмысл"}


def check_corpus(path: Path):
    results = []
    if not path.exists():
        return [("корпус существует", False, f"нет файла {path}")]
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    results.append(("колонки id, тип, название, формулировка, зависит_от, источник", not missing,
                    "нет колонок: " + ", ".join(missing) if missing else f"{len(columns)} колонок"))
    if missing:
        return results

    results.append((f"корпус ≥ {MIN_ROWS} утверждений", len(rows) >= MIN_ROWS, f"строк: {len(rows)}"))

    ids = [r["id"].strip() for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    results.append(("все id уникальны", not dup, "повторы: " + ", ".join(dup) if dup else "повторов нет"))

    empty_form = [r["id"] for r in rows if not r["формулировка"].strip()]
    results.append(("у каждой строки есть формулировка", not empty_form,
                    "пустые: " + ", ".join(empty_form) if empty_form else "все заполнены"))

    empty_src = [r["id"] for r in rows if not r["источник"].strip()]
    results.append(("у каждой строки есть источник", not empty_src,
                    "пустые: " + ", ".join(empty_src) if empty_src else "все заполнены"))

    idset = set(ids)
    broken, with_deps, self_ref = [], 0, []
    for r in rows:
        deps = [d.strip() for d in r["зависит_от"].split(";") if d.strip()]
        if deps:
            with_deps += 1
        for d in deps:
            if d not in idset:
                broken.append(f"{r['id']}→{d}")
            if d == r["id"].strip():
                self_ref.append(r["id"])
    results.append(("все ссылки в зависит_от существуют", not broken,
                    "битые: " + ", ".join(broken[:8]) + (" …" if len(broken) > 8 else "") if broken else "все ссылки найдены"))
    results.append(("нет ссылок на самого себя", not self_ref, ", ".join(self_ref) if self_ref else "нет"))
    results.append((f"≥ {MIN_WITH_DEPS} строк с непустым зависит_от", with_deps >= MIN_WITH_DEPS, f"строк с зависимостями: {with_deps}"))

    types = sorted({r["тип"].strip() for r in rows if r["тип"].strip()})
    results.append(("≥ 2 типов утверждений", len(types) >= 2, "типы: " + ", ".join(types) if types else "колонка тип пуста"))
    return results


def check_readme(path: Path):
    results = []
    if not path.exists():
        return [("README существует", False, f"нет файла {path}")]
    text = path.read_text(encoding="utf-8")

    # строки таблицы вопросов: начинаются с "| <номер> |"
    q_rows = [ln for ln in text.splitlines() if re.match(r"^\|\s*\d+\s*\|", ln)]
    filled = [ln for ln in q_rows if len([c for c in ln.split("|")[1:-1] if c.strip()]) >= 4]
    results.append(("5 вопросов исследователя заполнены (тип, вопрос, ответ, источник)", len(filled) >= 5,
                    f"заполненных строк: {len(filled)} из {len(q_rows)}"))

    joined = "\n".join(filled).lower()
    missing_types = [t for t, pat in QUESTION_TYPES.items() if not re.search(pat, joined)]
    results.append(("среди вопросов есть типы факт, зависимость, смысл", not missing_types,
                    "нет типа: " + ", ".join(missing_types) if missing_types else "все три есть"))

    has_area = bool(re.search(r"\*\*Название:\*\*\s*[^<\n]{3,}", text))
    results.append(("указана под-область (без угловых скобок-заглушек)", has_area, "заполнено" if has_area else "поле «Название» пустое или содержит заглушку"))
    return results


def check_files(root: Path):
    results = []
    for rel, desc in [("kg/MAP.md", "карта знаний"), ("AGENTS.md", "инструкции агенту"), ("diary.md", "журнал")]:
        p = root / rel
        ok = p.exists() and len(p.read_text(encoding="utf-8").strip()) > 200
        results.append((f"{rel} ({desc}) есть и заполнен", ok, "есть" if ok else "нет файла или он почти пустой"))
    diary = root / "diary.md"
    if diary.exists():
        placeholders = diary.read_text(encoding="utf-8").count("<")
        results.append(("в diary.md нет незаполненных заглушек <…>", placeholders <= 2, f"заглушек: {placeholders}"))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="kg/corpus.csv")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--root", default=".", help="корень репозитория (для MAP.md, AGENTS.md, diary.md)")
    args = ap.parse_args()
    root = Path(args.root)

    sections = [("Корпус", check_corpus(Path(args.corpus))),
                ("README", check_readme(Path(args.readme))),
                ("Файлы проекта", check_files(root))]
    total = passed = 0
    for title, res in sections:
        print(f"\n== {title}")
        for name, ok, note in res:
            total += 1; passed += ok
            print(f"  [{'зачтено' if ok else 'НЕ зачтено'}] {name} — {note}")
    print(f"\nИтого: {passed} из {total} пунктов зачтено. " + ("ДЗ1 можно сдавать." if passed == total else "Исправьте незачтённые пункты."))
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
