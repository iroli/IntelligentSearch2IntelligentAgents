"""Мини-агент лекции 2: ассистент редактора над онтологией.

Нужны: data/review.owl (его сохраняет блокнот лекции 2), owlready2, Java.
Запуск:
    python3 agent_l2.py                    # без ключа: модель-заглушка, инструменты настоящие
    LLM_API_KEY=... python3 agent_l2.py    # с моделью через OpenAI-совместимый API
"""
import contextlib, io, os, pathlib, re, warnings
warnings.filterwarnings("ignore")
from owlready2 import World, sync_reasoner_pellet, destroy_entity, OwlReadyInconsistentOntologyError
from agent_core import run_agent, openai_llm, scripted_llm   # цикл и модель прежние

# ---------------------------------------------------------------- 1. среда: онтология
world = World()
onto = world.get_ontology(pathlib.Path("data/review.owl").resolve().as_uri()).load()
state = {"проверено": False, "непротиворечива": None, "объяснение": ""}

def reason():
    """Запускает машину вывода Pellet. Результат запоминается в state."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            with onto:
                sync_reasoner_pellet(world, infer_property_values=True, debug=2)
        state.update(проверено=True, непротиворечива=True, объяснение="")
    except OwlReadyInconsistentOntologyError as e:
        text = str(e)
        text = text[text.find("Explanation"):] if "Explanation" in text else text
        text = re.sub(r"\s+", " ", text.replace("review.", ""))
        state.update(проверено=True, непротиворечива=False, объяснение=text[:300])

def ready():
    """Вывод выполнен и запись непротиворечива."""
    if not state["проверено"]:
        reason()
    return state["непротиворечива"]

# ---------------------------------------------------------------- 2. инструменты
def check_consistency():
    """Проверка непротиворечивости записи журнала и положения."""
    reason()
    out = {"непротиворечива": state["непротиворечива"]}
    if not state["непротиворечива"]:
        out["объяснение"] = state["объяснение"]
    return out

def infer(article: str):
    """Факты о статье, выведенные машиной вывода, с указанием правила."""
    if not ready():
        return {"ошибка": "запись противоречива, выводы недоступны"}
    a = onto[article]
    if a is None or onto.Статья not in a.is_a:
        return {"ошибка": f"статья {article} не найдена"}
    return {"статья": article,
            "выведено": [{"факт": f"{article} нарушает {x.name} ({x.label[0]})",
                          "правило": "правило 6.2 (SWRL)"} for x in a.нарушает],
            "организация_авторов": [o.name for o in a.организация_авторов]}

def graph_deps(doc_id: str):
    """От каких положений зависит данное положение (с учётом транзитивности)."""
    if not ready():
        return {"ошибка": "запись противоречива, выводы недоступны"}
    x = onto[doc_id]
    if x is None:
        return {"ошибка": f"положение {doc_id} не найдено"}
    return {"положение": doc_id,
            "зависит_от": sorted(f"{d.name} {d.label[0]}" for d in x.зависит_от),
            "основание": "транзитивность зависит_от"}

TOOLS = [
    {"type": "function", "function": {
        "name": "infer",
        "description": "Факты о статье, выведенные машиной вывода по онтологии, с указанием правила.",
        "parameters": {"type": "object",
                       "properties": {"article": {"type": "string", "description": "например P3"}},
                       "required": ["article"]}}},
    {"type": "function", "function": {
        "name": "graph_deps",
        "description": "От каких положений зависит данное положение (с учётом транзитивности).",
        "parameters": {"type": "object",
                       "properties": {"doc_id": {"type": "string", "description": "например R62"}},
                       "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "check_consistency",
        "description": "Проверка непротиворечивости записи журнала и положения.",
        "parameters": {"type": "object", "properties": {}}}},
]
IMPL = {"infer": infer, "graph_deps": graph_deps, "check_consistency": check_consistency}

# ---------------------------------------------------------------- 3. инструкция и вопрос
SYSTEM = ("Ты ассистент редактора сборника. Отвечай по-русски. Сначала вызови check_consistency. "
          "Если запись противоречива, сообщи об этом с пометкой [противоречие в данных] и не делай выводов о статьях. "
          "Иначе отвечай только по результатам инструментов. "
          "Помечай части ответа: [выведено: правило], [найдено: инструмент], [не подтверждено].")
QUESTION = ("Нарушает ли статья P3 требования положения о рецензировании, "
            "и на какие положения опирается нарушенное требование?")

# ---------------------------------------------------------------- 4. строка журнала с ошибкой
def add_error_row():
    """Новая рецензия на статью P8, в поле «рецензент» записана организация."""
    with onto:
        bad = onto.Рецензия("рец_P9_ошибка")
        bad.на_статью = onto.P8
        bad.рецензент = onto.ФИЦ_ИУ
    state["проверено"] = False
    return bad

def remove_error_row(bad):
    destroy_entity(bad)
    state["проверено"] = False

# ---------------------------------------------------------------- 5. запуск
def ask(plan, record):
    llm = openai_llm if os.environ.get("LLM_API_KEY") else scripted_llm(plan, record)
    return run_agent(QUESTION, llm, SYSTEM, TOOLS, IMPL)

if __name__ == "__main__":
    print("=== Прогон 1: вывод ===")
    ask([("check_consistency", {}), ("infer", {"article": "P3"}), ("graph_deps", {"doc_id": "R62"})],
        "[запись ответа модели]\n"
        "Запись непротиворечива [найдено: check_consistency]. Статья P3 нарушает требование 6.2 "
        "«Независимость рецензента» [выведено: правило 6.2]. Требование 6.2 опирается на раздел 6, раздел 3, "
        "раздел 1 и термины Рецензент, Автор, Статья, Редакционная коллегия [найдено: graph_deps]. "
        "О требованиях 6.1 и 6.6 инструменты сведений не дали [не подтверждено].")

    print("\n=== Прогон 2: журнал со строкой с ошибкой ===")
    bad = add_error_row()
    ask([("check_consistency", {})],
        "[запись ответа модели]\n"
        "Запись журнала противоречива [противоречие в данных]: индивид ФИЦ_ИУ указан рецензентом "
        "в строке рец_P9_ошибка и одновременно является организацией. "
        "Выводы о статьях не делаю до исправления строки.")
    remove_error_row(bad)
    print("\nпосле исправления строки:", check_consistency())
