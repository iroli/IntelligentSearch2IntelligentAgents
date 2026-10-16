#!/usr/bin/env python3
"""Проверка критериев задания лекции 2: онтология своей под-области.

Запуск из корня репозитория:
    python3 scripts/check_l2.py                          # kg/ontology.ttl и kg/ONTOLOGY.md
    python3 scripts/check_l2.py --ontology path.ttl      # другой файл онтологии (.ttl или .owl)
    python3 scripts/check_l2.py --no-reasoner            # без запуска машины вывода
Код возврата 0: все пункты зачтены. Код возврата 1: есть незачтённые.
Нужен rdflib. Для проверки непротиворечивости нужны owlready2 и Java.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

MIN_CLASSES, MIN_PROPERTIES, MIN_RESTRICTIONS, MIN_INDIVIDUALS = 6, 3, 2, 5
SECTIONS = ["Классы", "Свойства", "Высказывания", "Выведенный факт", "Противоречие"]


def load(path: Path):
    import rdflib
    g = rdflib.Graph()
    fmt = "turtle" if path.suffix in (".ttl", ".turtle") else "xml"
    g.parse(str(path), format=fmt)
    return g


def check_ontology(path: Path):
    from rdflib import RDF, RDFS, OWL, URIRef, BNode
    if not path.exists():
        return [("файл онтологии существует", False, f"нет файла {path}")], None
    try:
        g = load(path)
    except Exception as e:  # файл не разбирается
        return [("файл онтологии разбирается", False, str(e)[:120])], None
    res = [("файл онтологии разбирается", True, f"троек: {len(g)}")]

    classes = {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    res.append((f"классов ≥ {MIN_CLASSES}", len(classes) >= MIN_CLASSES, f"классов: {len(classes)}"))

    sub = [(s, o) for s, o in g.subject_objects(RDFS.subClassOf) if s in classes and o in classes]
    parents = {s: o for s, o in sub}
    two_levels = any(o in parents for _, o in sub)
    res.append(("есть иерархия классов", bool(sub), f"связей «подкласс» между классами: {len(sub)}"))
    res.append(("иерархия глубже одного уровня", two_levels, "есть цепочка A ⊑ B ⊑ C" if two_levels else "все подклассы на одном уровне"))

    oprops = set(g.subjects(RDF.type, OWL.ObjectProperty))
    dprops = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    res.append((f"свойств ≥ {MIN_PROPERTIES}", len(oprops | dprops) >= MIN_PROPERTIES,
                f"объектных: {len(oprops)}, свойств данных: {len(dprops)}"))

    char_types = [OWL.TransitiveProperty, OWL.SymmetricProperty, OWL.FunctionalProperty,
                  OWL.InverseFunctionalProperty, OWL.AsymmetricProperty, OWL.ReflexiveProperty, OWL.IrreflexiveProperty]
    with_char = {p for t in char_types for p in g.subjects(RDF.type, t)} | set(g.subjects(OWL.inverseOf, None))
    res.append(("есть свойство с характеристикой", bool(with_char), f"свойств с характеристикой: {len(with_char)}"))

    with_dr = {p for p in oprops | dprops if (p, RDFS.domain, None) in g or (p, RDFS.range, None) in g}
    res.append(("у свойств указаны области", len(with_dr) >= 1, f"свойств с областью определения или значений: {len(with_dr)}"))

    restrictions = set(g.subjects(RDF.type, OWL.Restriction))
    res.append((f"ограничений ≥ {MIN_RESTRICTIONS}", len(restrictions) >= MIN_RESTRICTIONS, f"ограничений: {len(restrictions)}"))

    disjoint = len(set(g.subjects(RDF.type, OWL.AllDisjointClasses))) + len(list(g.subject_objects(OWL.disjointWith)))
    res.append(("объявлена дизъюнктность", disjoint >= 1, f"записей о дизъюнктности: {disjoint}"))

    inds = {s for s in g.subjects(RDF.type, OWL.NamedIndividual) if isinstance(s, URIRef)}
    inds |= {s for s, o in g.subject_objects(RDF.type) if o in classes and isinstance(s, URIRef)}
    res.append((f"индивидов ≥ {MIN_INDIVIDUALS}", len(inds) >= MIN_INDIVIDUALS, f"индивидов: {len(inds)}"))

    as_class = inds & classes
    res.append(("индивиды не стоят на месте классов", not as_class,
                "совпадают: " + ", ".join(str(x).split("#")[-1] for x in sorted(as_class)[:5]) if as_class else "совпадений нет"))

    facts = sum(1 for s, p, o in g if s in inds and p in oprops and o in inds)
    res.append(("между индивидами есть связи", facts >= 3, f"фактов о парах индивидов: {facts}"))
    return res, g


def check_reasoner(path: Path):
    """Запускает машину вывода, если есть owlready2 и Java."""
    if shutil.which("java") is None:
        return [("запись непротиворечива (машина вывода)", None, "Java не найдена, пункт пропущен")]
    try:
        import warnings; warnings.filterwarnings("ignore")
        import owlready2 as o2
    except ImportError:
        return [("запись непротиворечива (машина вывода)", None, "owlready2 не установлен, пункт пропущен")]
    import io, contextlib, tempfile, rdflib
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
        return [("запись непротиворечива (машина вывода)", False, "машина вывода сообщила о противоречии")]
    except Exception as e:
        return [("запись непротиворечива (машина вывода)", None, f"машина вывода не запустилась: {str(e)[:80]}")]
    after = len(world.as_rdflib_graph())
    return [("запись непротиворечива (машина вывода)", True, "противоречий нет"),
            ("машина вывода получила новые факты", after > before, f"троек до вывода: {before}, после: {after}")]


def check_docs(root: Path):
    res = []
    p = root / "kg" / "ONTOLOGY.md"
    if not p.exists():
        return [("kg/ONTOLOGY.md существует", False, "нет файла")]
    text = p.read_text(encoding="utf-8")
    missing = [s for s in SECTIONS if not re.search(rf"^##\s+{re.escape(s)}", text, flags=re.M)]
    res.append(("в ONTOLOGY.md есть все разделы", not missing, "нет разделов: " + ", ".join(missing) if missing else "разделы на месте"))
    placeholders = len(re.findall(r"<[^<>\n]{2,}>", text))
    res.append(("в ONTOLOGY.md нет незаполненных заглушек <…>", placeholders == 0, f"заглушек: {placeholders}"))
    owl_lines = len(re.findall(r"SubClassOf|EquivalentTo|Characteristics|DisjointWith|some |only |min ", text))
    res.append(("высказывания записаны в OWL", owl_lines >= 2, f"строк с записью OWL: {owl_lines}"))
    for name in ("diary.md", "AGENTS.md"):
        f = root / name
        res.append((f"{name} существует", f.exists(), "есть" if f.exists() else "нет файла"))
    a = root / "AGENTS.md"
    if a.exists():
        has = bool(re.search(r"словар[ьяю] классов|классы онтологии", a.read_text(encoding="utf-8"), flags=re.I))
        res.append(("в AGENTS.md есть словарь классов", has, "найден" if has else "добавьте раздел со словарём классов онтологии"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="корень репозитория")
    ap.add_argument("--ontology", default=None, help="файл онтологии (по умолчанию kg/ontology.ttl)")
    ap.add_argument("--no-reasoner", action="store_true", help="не запускать машину вывода")
    args = ap.parse_args()
    root = Path(args.root)
    onto_path = Path(args.ontology) if args.ontology else root / "kg" / "ontology.ttl"

    blocks = []
    res, g = check_ontology(onto_path)
    blocks.append(("Онтология", res))
    if g is not None and not args.no_reasoner:
        blocks.append(("Машина вывода", check_reasoner(onto_path)))
    blocks.append(("Описание и журнал", check_docs(root)))

    passed = total = skipped = 0
    for title, res in blocks:
        print(f"\n== {title}")
        for name, ok, note in res:
            if ok is None:
                skipped += 1
                print(f"  [пропущено] {name} — {note}")
                continue
            total += 1
            passed += bool(ok)
            print(f"  [{'зачтено' if ok else 'НЕ зачтено'}] {name} — {note}")
    print(f"\nИтого: {passed} из {total} пунктов зачтено" + (f", пропущено {skipped}" if skipped else "") + ".")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
