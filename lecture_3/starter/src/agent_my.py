"""Мини-агент стартового набора: ассистент по правилам зачёта.

Запуск из корня репозитория (папка starter/):
    python3 src/agent_my.py
Без модели работает модель-заглушка: инструменты настоящие, ответ записан заранее.
"""
import os
import pandas as pd
from rdflib import Graph, Namespace, RDF
from search_bm25 import BM25
from agent_core import run_agent, openai_llm, scripted_llm

# ---------------------------------------------------------------- среда
corpus = pd.read_csv("kg/corpus.csv", dtype=str).fillna("")
vedomost = pd.read_csv("kg/data/vedomost.csv", dtype=str).fillna("")
rows = corpus.set_index("id")
bm25 = BM25((corpus["название"] + ". " + corpus["формулировка"]).tolist())

EX = Namespace("http://example.org/kg#")
g = Graph()
for _, row in corpus.iterrows():
    g.add((EX[row["id"]], RDF.type, EX[row["тип"]]))
    for dep in filter(None, row["зависит_от"].split(";")):
        g.add((EX[row["id"]], EX.dependsOn, EX[dep.strip()]))

# ---------------------------------------------------------------- инструменты
def search_words(query: str, k: int = 5):
    """Поиск по словам по корпусу правил."""
    return [{"id": corpus.loc[i, "id"], "название": corpus.loc[i, "название"], "bm25": round(s, 2)}
            for i, s in bm25.search(query, k=k)]

def graph_deps(doc_id: str):
    """На какие положения транзитивно опирается положение."""
    if doc_id not in rows.index:
        return {"ошибка": f"записи {doc_id} нет в корпусе"}
    q = ("PREFIX ex: <http://example.org/kg#> "
         f"SELECT ?dep WHERE {{ ex:{doc_id} ex:dependsOn+ ?dep . }}")
    ids = sorted(str(r.dep).split("#")[1] for r in g.query(q))
    return {"документ": doc_id, "опирается_на": [f"{i} {rows.loc[i, 'название']}" for i in ids]}

def get_statement(doc_id: str):
    """Полная запись корпуса: формулировка, прямые зависимости, источник."""
    if doc_id not in rows.index:
        return {"ошибка": f"записи {doc_id} нет в корпусе"}
    r = rows.loc[doc_id]
    return {"id": doc_id, "тип": r["тип"], "название": r["название"],
            "формулировка": r["формулировка"],
            "зависит_от": [d for d in r["зависит_от"].split(";") if d],
            "источник": r["источник"]}

RULES = {
    "3.1": ("сдано не менее трёх заданий", lambda d: d["сдана"].sum() >= 3),
    "3.2": ("задания сданы не позже срока", lambda d: not d.loc[d["сдана"], "опоздание"].any()),
    "3.3": ("сумма баллов не меньше 60", lambda d: d.loc[d["сдана"], "балл_число"].sum() >= 60),
}

def check_rules():
    """Применить требования 3.1, 3.2, 3.3 к ведомости. Вердикт по каждому студенту."""
    v = vedomost.copy()
    v["сдана"] = v["сдано"] != ""
    v["балл_число"] = pd.to_numeric(v["балл"], errors="coerce").fillna(0)
    v["опоздание"] = pd.to_datetime(v["сдано"], errors="coerce") > pd.to_datetime(v["срок"])
    out = []
    for student, d in v.groupby("студент", sort=True):
        failed = [f"{k} ({RULES[k][0]})" for k in RULES if not RULES[k][1](d)]
        out.append({"студент": student, "зачёт": "нет" if failed else "да",
                    "сдано заданий": int(d["сдана"].sum()),
                    "сумма баллов": int(d.loc[d["сдана"], "балл_число"].sum()),
                    "причина": ", ".join(failed) or "—"})
    return out

TOOLS = [
    {"type": "function", "function": {
        "name": "search_words",
        "description": "Поиск по словам (BM25) по корпусу правил зачёта. Возвращает id и названия.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "graph_deps",
        "description": "На какие положения транзитивно опирается положение с данным id.",
        "parameters": {"type": "object",
                       "properties": {"doc_id": {"type": "string", "description": "например R34"}},
                       "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "get_statement",
        "description": "Полная запись корпуса по id: формулировка, прямые зависимости, источник.",
        "parameters": {"type": "object",
                       "properties": {"doc_id": {"type": "string", "description": "например R34"}},
                       "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "check_rules",
        "description": "Применить требования 3.1, 3.2, 3.3 к ведомости. Вердикт по каждому студенту.",
        "parameters": {"type": "object", "properties": {}}}},
]
IMPL = {"search_words": search_words, "graph_deps": graph_deps,
        "get_statement": get_statement, "check_rules": check_rules}

# ---------------------------------------------------------------- инструкция и вопрос
SYSTEM = ("Ты ассистент преподавателя по правилам зачёта. Отвечай по-русски. Сначала вызови инструменты, затем отвечай только по их результатам. "
          "У каждого положения в ответе указывай id из корпуса. "
          "Помечай части ответа: [проверено: инструмент], [найдено: инструмент], [не подтверждено]. "
          "Положения, которых нет в результатах инструментов, не называй. "
          "Идентификатор требования 3.4 о выставлении зачёта: R34.")
QUESTION = "Кто из студентов получает зачёт, и на какие положения опирается требование о выставлении зачёта?"

PLAN = [("check_rules", {}), ("graph_deps", {"doc_id": "R34"})]
RECORD = ("[запись ответа модели]\n"
          "Зачёт получают Аня и Боря [проверено: check_rules]. Вика не получает по 3.2 (задание сдано позже срока), "
          "Гоша по 3.1 и 3.3 (сдано два задания, сумма 45), Даша по 3.3 (сумма 53) [проверено: check_rules]. "
          "Требование 3.4 (R34) опирается на 11 положений: требования R31, R32, R33, разделы S1, S2, S3 "
          "и термины T1–T5 [найдено: graph_deps].")

if __name__ == "__main__":
    llm = openai_llm if os.environ.get("LLM_API_KEY") else scripted_llm(PLAN, RECORD)
    run_agent(QUESTION, llm, SYSTEM, TOOLS, IMPL)
