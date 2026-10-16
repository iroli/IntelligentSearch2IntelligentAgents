"""Задание лекции 2 на стартовом наборе: онтология правил зачёта.

Запуск из корня репозитория (папка starter/):
    python3 src/build_ontology.py
Записывает kg/ontology.owl и kg/ontology.ttl, запускает машину вывода (нужна Java),
печатает выведенный факт и разбор намеренного противоречия.
"""
import contextlib, io, pathlib, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import rdflib
from owlready2 import *

corpus = pd.read_csv("kg/corpus.csv", dtype=str).fillna("")
vedomost = pd.read_csv("kg/data/vedomost.csv", dtype=str).fillna("")

world = World()
onto = world.get_ontology("http://example.org/zachet.owl")

with onto:
    # 1. классы: из столбца «тип» корпуса и из ведомости
    class Объект(Thing): pass
    class Положение(Объект): pass
    class Раздел(Положение): pass
    class Термин(Положение): pass
    class Требование(Положение): pass
    class Студент(Объект): pass
    class Сдача(Объект): pass
    AllDisjoint([Раздел, Термин, Требование])
    AllDisjoint([Положение, Студент, Сдача])

    # 2. свойства: стрелки карты знаний
    class зависит_от(ObjectProperty, TransitiveProperty):
        domain = [Положение]
        range = [Положение]
    class автор(ObjectProperty, FunctionalProperty):
        domain = [Сдача]
        range = [Студент]
    class балл(DataProperty, FunctionalProperty):
        domain = [Сдача]
        range = [int]

    # 3. ограничения: фразы «у каждого X есть Y»
    Сдача.is_a.append(автор.some(Студент))             # у каждой сдачи есть автор
    Требование.is_a.append(зависит_от.some(Раздел))    # каждое требование относится к разделу

    # 4. индивиды: все записи корпуса
    TYPES = {"Раздел": Раздел, "Термин": Термин, "Требование": Требование}
    pos = {}
    for _, r in corpus.iterrows():
        x = TYPES[r["тип"]](r["id"])
        x.label = [locstr(r["название"], lang="ru")]
        pos[r["id"]] = x
    for _, r in corpus.iterrows():
        pos[r["id"]].зависит_от = [pos[d] for d in r["зависит_от"].split(";") if d]

    # и три строки ведомости
    for _, r in vedomost[vedomost["сдано"] != ""].head(3).iterrows():
        st = onto[r["студент"]] or Студент(r["студент"])
        s = Сдача(f"сдача_{r['студент']}_{r['задание']}")
        s.автор = st
        s.балл = int(r["балл"])

# 5. сохранение до запуска машины вывода
onto.save(file="kg/ontology.owl", format="rdfxml")
g = rdflib.Graph()
g.parse("kg/ontology.owl", format="xml")
g.bind("", "http://example.org/zachet.owl#")
g.serialize(destination="kg/ontology.ttl", format="turtle")
print("троек:", len(g))

def run_reasoner(world, onto, explain=False):
    """Запускает Pellet. Возвращает (непротиворечива, объяснение)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            with onto:
                sync_reasoner_pellet(world, infer_property_values=True,
                                     debug=2 if explain else 0)
        return True, ""
    except OwlReadyInconsistentOntologyError as e:
        text = str(e)
        return False, text[text.find("Explanation"):] if "Explanation" in text else text

# 6. выведенный факт
R34 = pos["R34"]
written = sorted(x.name for x in R34.зависит_от)
ok, _ = run_reasoner(world, onto)
inferred = sorted(x.name for x in R34.зависит_от)
print("непротиворечива:", ok)
print("R34 зависит_от, записано:", written)
print("R34 зависит_от, после вывода:", inferred)
print("выведено:", sorted(set(inferred) - set(written)))

# 7. намеренное противоречие на свежей копии: в поле «автор» записано требование
world2 = World()
onto2 = world2.get_ontology(pathlib.Path("kg/ontology.owl").resolve().as_uri()).load()
with onto2:
    bad = onto2.Сдача("сдача_ошибка")
    bad.автор = onto2["R31"]
ok, msg = run_reasoner(world2, onto2, explain=True)
print("после ошибки, непротиворечива:", ok)
print(msg.replace("ontology.", "").strip()[:500])
destroy_entity(bad)
ok, _ = run_reasoner(world2, onto2)
print("после исправления, непротиворечива:", ok)
