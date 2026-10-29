#!/usr/bin/env python3
"""Проверка критериев задания лекции 3: вопросы, онтология v2, формы SHACL, отчёты.

Запуск из корня репозитория:
    python3 scripts/check_l3.py                       # kg/ontology.ttl, kg/shapes.ttl, kg/QUESTIONS.md, kg/VALIDATION.md
    python3 scripts/check_l3.py --no-reasoner         # без машины вывода
Код возврата 0: все пункты зачтены. Нужны rdflib и pyshacl. Для машины вывода нужны owlready2 и Java.
"""
import argparse, re, shutil, sys
from pathlib import Path

MIN_SHAPES, MIN_QUESTIONS = 3, 3
Q_SECTIONS = ["Вопросы", "Вопрос — класс — свойство", "Выражения классов"]
V_SECTIONS = ["Машина вывода", "Проверка по формам, данные без ошибки", "Проверка по формам, намеренная ошибка"]


def load(path: Path):
    import rdflib
    g = rdflib.Graph()
    g.parse(str(path), format="turtle" if path.suffix in (".ttl", ".turtle") else "xml")
    return g


def check_questions(root: Path):
    p = root / "kg" / "QUESTIONS.md"
    if not p.exists():
        return [("kg/QUESTIONS.md существует", False, "нет файла")]
    text = p.read_text(encoding="utf-8")
    res = []
    missing = [s for s in Q_SECTIONS if not re.search(rf"^##\s+{re.escape(s)}", text, flags=re.M)]
    res.append(("в QUESTIONS.md есть все разделы", not missing, "нет разделов: " + ", ".join(missing) if missing else "разделы на месте"))
    rows = [l for l in text.splitlines() if re.match(r"^\|\s*\d+\s*\|", l)]
    filled = [l for l in rows if len([c for c in l.split("|")[1:-1] if c.strip() and "<" not in c]) >= 5]
    res.append((f"≥ {MIN_QUESTIONS} вопросов с ответом, источником и средством", len(filled) >= MIN_QUESTIONS, f"заполненных строк: {len(filled)}"))
    placeholders = len(re.findall(r"<[^<>\n]{2,}>", text))
    res.append(("в QUESTIONS.md нет заглушек <…>", placeholders == 0, f"заглушек: {placeholders}"))
    res.append(("есть выражение класса (EquivalentTo)", "EquivalentTo" in text, "найдено" if "EquivalentTo" in text else "запишите хотя бы один выводимый класс"))
    return res


def check_ontology(path: Path):
    from rdflib import RDF, OWL
    if not path.exists():
        return [("файл онтологии существует", False, f"нет файла {path}")], None
    try:
        g = load(path)
    except Exception as e:
        return [("файл онтологии разбирается", False, str(e)[:120])], None
    res = [("файл онтологии разбирается", True, f"троек: {len(g)}")]
    eq = list(g.subject_objects(OWL.equivalentClass))
    res.append(("есть выводимый класс (owl:equivalentClass)", len(eq) >= 1, f"определений: {len(eq)}"))
    return res, g


def check_shapes(root: Path, data_path: Path):
    from rdflib import RDF
    import rdflib
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    p = root / "kg" / "shapes.ttl"
    if not p.exists():
        return [("kg/shapes.ttl существует", False, "нет файла")]
    try:
        sg = load(p)
    except Exception as e:
        return [("файл форм разбирается", False, str(e)[:120])]
    res = [("файл форм разбирается", True, f"троек: {len(sg)}")]
    node_shapes = set(sg.subjects(RDF.type, SH.NodeShape))
    props = list(sg.subject_objects(SH.property))
    sparql = list(sg.subject_objects(SH.sparql))
    n = len(props) + len(sparql)
    res.append((f"форм (sh:property или sh:sparql) ≥ {MIN_SHAPES}", n >= MIN_SHAPES, f"узловых форм: {len(node_shapes)}, свойств форм: {len(props)}, запросов: {len(sparql)}"))
    kinds = {"число связей": [SH.minCount, SH.maxCount, SH.qualifiedMinCount],
             "тип или значения": [SH["class"], SH.datatype, SH["in"], SH.hasValue, SH.pattern],
             "сравнение или составная": [SH.lessThan, SH.lessThanOrEquals, SH.equals, SH.disjoint, SH.qualifiedValueShape, SH.sparql]}
    found = {k: any(list(sg.subject_objects(pr)) for pr in v) for k, v in kinds.items()}
    res.append(("есть формы трёх видов", all(found.values()), ", ".join(f"{k}: {'да' if v else 'нет'}" for k, v in found.items())))
    messages = [o for _, o in sg.subject_objects(SH.message)]
    res.append(("у форм есть сообщения", len(messages) >= MIN_SHAPES, f"сообщений: {len(messages)}"))
    if data_path.exists():
        try:
            from pyshacl import validate
            ok, rep, text = validate(load(data_path), shacl_graph=sg, inference="none")
            res.append(("проверка данных без ошибки проходит", bool(ok), "Conforms: True" if ok else "есть нарушения: " + re.sub(r"\s+", " ", text)[:160]))
        except ImportError:
            res.append(("проверка pySHACL", None, "pyshacl не установлен, пункт пропущен"))
    return res


def check_reasoner(path: Path):
    if shutil.which("java") is None:
        return [("машина вывода", None, "Java не найдена, пункт пропущен")]
    try:
        import warnings; warnings.filterwarnings("ignore")
        import owlready2 as o2
    except ImportError:
        return [("машина вывода", None, "owlready2 не установлен, пункт пропущен")]
    import io, contextlib, tempfile
    tmp = Path(tempfile.mkdtemp()) / "onto.owl"
    load(path).serialize(destination=str(tmp), format="xml")
    world = o2.World()
    onto = world.get_ontology(tmp.as_uri()).load()
    before = len(world.as_rdflib_graph())
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with onto:
                o2.sync_reasoner_pellet(world, infer_property_values=True, debug=0)
    except o2.OwlReadyInconsistentOntologyError:
        return [("запись непротиворечива", False, "машина вывода сообщила о противоречии")]
    except Exception as e:
        return [("машина вывода", None, f"не запустилась: {str(e)[:80]}")]
    unsat = [c.name for c in onto.classes() if o2.Nothing in c.equivalent_to]
    after = len(world.as_rdflib_graph())
    return [("запись непротиворечива", True, "противоречий нет"),
            ("нет неудовлетворимых классов", not unsat, "все классы выполнимы" if not unsat else "эквивалентны Nothing: " + ", ".join(unsat)),
            ("машина вывода получила новые факты", after > before, f"троек до вывода: {before}, после: {after}")]


def check_validation(root: Path):
    p = root / "kg" / "VALIDATION.md"
    if not p.exists():
        return [("kg/VALIDATION.md существует", False, "нет файла")]
    text = p.read_text(encoding="utf-8")
    res = []
    missing = [s for s in V_SECTIONS if not re.search(rf"^##\s+{re.escape(s)}", text, flags=re.M)]
    res.append(("в VALIDATION.md есть все разделы", not missing, "нет разделов: " + ", ".join(missing) if missing else "разделы на месте"))
    res.append(("в VALIDATION.md нет заглушек <…>", not re.findall(r"<[^<>\n]{2,}>", text), "заглушек нет" if not re.findall(r"<[^<>\n]{2,}>", text) else "заполните заглушки"))
    res.append(("приложены оба отчёта проверки", text.count("Conforms: True") >= 1 and "Conforms: False" in text, "есть True и False" if ("Conforms: True" in text and "Conforms: False" in text) else "нужны отчёт без ошибки и отчёт с ошибкой"))
    for name in ("diary.md", "AGENTS.md"):
        res.append((f"{name} существует", (root / name).exists(), "есть" if (root / name).exists() else "нет файла"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--ontology", default=None)
    ap.add_argument("--no-reasoner", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    onto_path = Path(args.ontology) if args.ontology else root / "kg" / "ontology.ttl"
    blocks = [("Вопросы", check_questions(root))]
    res, g = check_ontology(onto_path)
    blocks.append(("Онтология", res))
    if g is not None and not args.no_reasoner:
        blocks.append(("Машина вывода", check_reasoner(onto_path)))
    blocks.append(("Формы SHACL", check_shapes(root, onto_path)))
    blocks.append(("Отчёты", check_validation(root)))
    passed = total = skipped = 0
    for title, res in blocks:
        print(f"\n== {title}")
        for name, ok, note in res:
            if ok is None:
                skipped += 1; print(f"  [пропущено] {name} — {note}"); continue
            total += 1; passed += bool(ok)
            print(f"  [{'зачтено' if ok else 'НЕ зачтено'}] {name} — {note}")
    print(f"\nИтого: {passed} из {total} пунктов зачтено" + (f", пропущено {skipped}" if skipped else "") + ".")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
