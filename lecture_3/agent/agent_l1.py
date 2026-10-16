"""Мини-агент лекции 1: ассистент редактора сборника.

Запуск из папки, где лежит data/:
    python3 agent_l1.py            # без ключа: модель-заглушка, инструменты настоящие
    LLM_API_KEY=... python3 agent_l1.py   # с моделью через OpenAI-совместимый API
"""
import os, pathlib
import pandas as pd
from rdflib import Graph, Namespace, RDF
from search_bm25 import BM25
from agent_core import run_agent, openai_llm, scripted_llm

# ---------------------------------------------------------------- 1. данные (среда)
DATA = pathlib.Path("data")
reviews = pd.read_csv(DATA / "reviews.csv", dtype=str).fillna("")
corpus = pd.read_csv(DATA / "corpus_Aprime_standards.csv", dtype=str).fillna("")
claims = pd.read_csv(DATA / "paper_P2_claims.csv", dtype=str).fillna("")
title = corpus.set_index("id")["название"]

# ---------------------------------------------------------------- 2. инструменты (T)
# 2.1. поиск по словам
bm25 = BM25((corpus["название"] + ". " + corpus["формулировка"]).tolist())

def search_words(query: str, k: int = 5):
    """Поиск по словам: какие положения содержат слова запроса."""
    return [{"id": corpus.loc[i, "id"], "название": corpus.loc[i, "название"], "bm25": round(s, 2)}
            for i, s in bm25.search(query, k=k)]

# 2.2. граф зависимостей
EX = Namespace("http://example.org/kg#")
g = Graph()
for _, row in corpus.iterrows():
    g.add((EX[row["id"]], RDF.type, EX[row["тип"]]))
    for dep in filter(None, row["зависит_от"].split(";")):
        g.add((EX[row["id"]], EX.dependsOn, EX[dep.strip()]))

def graph_deps(doc_id: str):
    """На какие положения транзитивно опирается документ."""
    q = ("PREFIX ex: <http://example.org/kg#> "
         f"SELECT ?dep WHERE {{ ex:{doc_id} ex:dependsOn+ ?dep . }}")
    ids = sorted(str(r.dep).split("#")[1] for r in g.query(q))
    return {"документ": doc_id, "опирается_на": [f"{i} {title[i]}" for i in ids]}

# 2.3. правила положения над журналом
RULES = {
    "6.1": ("не менее двух сданных рецензий",
            lambda d: d["сдана"].sum() >= 2),
    "6.2": ("рецензент не из организации авторов",
            lambda d: not d.loc[d["сдана"], "своя_организация"].any()),
    "6.6": ("рецензии сданы в 21 день",
            lambda d: not d.loc[d["сдана"], "просрочена"].any()),
    "6.5": ("есть ответ авторов на замечания",
            lambda d: d["сдана"].sum() >= 2 and (d.loc[d["сдана"], "ответ_авторов"] == "да").all()),
}

def check_rules():
    """Применить правила 6.1, 6.2, 6.5, 6.6 к журналу. Вердикт по каждой статье."""
    rv = reviews.copy()
    rv["сдана"] = rv["сдано"] != ""
    rv["оценка_10"] = pd.to_numeric(rv["оценка"], errors="coerce") * 10 / pd.to_numeric(rv["шкала"], errors="coerce")
    rv["дней"] = (pd.to_datetime(rv["сдано"], errors="coerce") - pd.to_datetime(rv["назначено"])).dt.days
    rv["своя_организация"] = rv["рецензент_организация"] == rv["авторы_организация"]
    rv["просрочена"] = rv["дней"] > 21
    out = []
    for paper, d in rv.groupby("статья", sort=True):
        failed = [f"{k} ({RULES[k][0]})" for k in RULES if not RULES[k][1](d)]
        mean10 = d.loc[d["сдана"], "оценка_10"].mean()
        out.append({"статья": paper, "выносить": "нет" if failed else "да",
                    "средняя оценка (шкала 10)": round(float(mean10), 1) if pd.notna(mean10) else None,
                    "причина": ", ".join(failed) or "—"})
    return out

# 2.4. вердикты внешней проверки по утверждениям статьи P2
def lean_check_claims():
    """Вердикты внешнего верификатора доказательств (запись из файла)."""
    return [{"id": c["id"], "утверждение": c["формулировка"], "вердикт": c["вердикт_lean"]}
            for _, c in claims.iterrows()]

# ---------------------------------------------------------------- 3. описание инструментов для модели
TOOLS = [
    {"type": "function", "function": {
        "name": "search_words",
        "description": "Поиск по словам (BM25) по корпусу положения.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "graph_deps",
        "description": "На какие положения транзитивно опирается документ (граф знаний, зависит_от+).",
        "parameters": {"type": "object",
                       "properties": {"doc_id": {"type": "string", "description": "например R65"}},
                       "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "check_rules",
        "description": "Применить правила 6.1, 6.2, 6.5, 6.6 положения к журналу рецензий. Вердикт по каждой статье.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "lean_check_claims",
        "description": "Вердикты внешнего верификатора доказательств по утверждениям статьи P2.",
        "parameters": {"type": "object", "properties": {}}}},
]
IMPL = {"search_words": search_words, "graph_deps": graph_deps,
        "check_rules": check_rules, "lean_check_claims": lean_check_claims}

# ---------------------------------------------------------------- 4. инструкция (P) и цель (G)
SYSTEM = ("Ты ассистент редактора сборника. Отвечай по-русски. Сначала вызови инструменты, затем отвечай только по их результатам. "
          "Каждую часть ответа помечай: [проверено: инструмент] или [найдено: инструмент]. "
          "Документы и пункты, которых нет в результатах инструментов, не называй. "
          "Если после вызова инструментов что-то не подтвердилось, помечай это [не подтверждено]. "
          "Идентификатор требования 6.5 в корпусе: R65.")
QUESTION = ("Какие статьи можно выносить на решение редколлегии по данным журнала рецензий, "
            "и на какие положения опирается это требование? Назови документы и пункты.")

# ---------------------------------------------------------------- 5. запуск
PLAN = [("check_rules", {}),
        ("search_words", {"query": "принятие решения редколлегией требование"}),
        ("graph_deps", {"doc_id": "R65"}),
        ("lean_check_claims", {})]
RECORD = ("[запись ответа модели]\n"
          "Можно выносить на решение: P1, P2, P7 [проверено: check_rules]. Нельзя: P3 по 6.2 (рецензент из организации авторов), "
          "P4 по 6.1 и 6.5 (одна рецензия), P5 по 6.6 (просрочена), P6 по 6.5 (нет ответа авторов), P8 по 6.1 и 6.5 (рецензия не сдана) "
          "[проверено: check_rules]. Требование 6.5 (R65) опирается на 14 положений: требования 6.1, 6.3, 6.4, "
          "разделы 1, 3, 6 и восемь терминов [найдено: graph_deps]. По статье P2: утверждения C1 и C2 приняты, C3 отклонено, "
          "замечание рецензента обосновано [проверено: lean_check_claims]. Иных нормативных документов в результатах "
          "инструментов нет [не подтверждено].")

if __name__ == "__main__":
    if os.environ.get("LLM_API_KEY"):
        llm = openai_llm
    else:
        llm = scripted_llm(PLAN, RECORD)
    run_agent(QUESTION, llm, SYSTEM, TOOLS, IMPL)
