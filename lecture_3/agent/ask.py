#!/usr/bin/env python3
"""Вопрос модели из терминала: без инструментов и с инструментами.

    python3 agent/ask.py --status                      что с моделью
    python3 agent/ask.py --plain "вопрос"              модель без инструментов (к вопросу прикладывается журнал)
    python3 agent/ask.py "вопрос"                      агент лекции 1 (поиск, граф, правила, внешняя проверка)
    python3 agent/ask.py --agent l2 "вопрос"           агент лекции 2 (онтология и машина вывода)
    python3 agent/ask.py --agent starter "вопрос"      агент стартового набора (правила зачёта)
    python3 agent/ask.py --agent l1                    без вопроса: диалог, пустая строка завершает

Запускать из папки занятия (где лежат data/ и starter/). Нужна модель: локальная Ollama (см. README.md)
или переменные LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))        # agent/
ROOT = os.path.dirname(HERE)                             # папка занятия: data/, starter/
os.chdir(ROOT)
sys.path.insert(0, HERE)
from llm_local import pick_llm, ask_plain, warm_up
from agent_core import run_agent

def load_agent(name):
    """Возвращает (SYSTEM, TOOLS, IMPL, вопрос по умолчанию)."""
    if name == "l1":
        import agent_l1 as a
    elif name == "l2":
        import agent_l2 as a
    elif name == "starter":
        os.chdir(os.path.join(ROOT, "starter"))
        sys.path.insert(0, os.path.join(ROOT, "starter", "src"))
        import agent_my as a
    else:
        sys.exit(f"неизвестный агент: {name}")
    return a.SYSTEM, a.TOOLS, a.IMPL, a.QUESTION

def journal_text():
    import pandas as pd
    j = pd.read_csv(os.path.join(ROOT, "data", "reviews.csv"), dtype=str).fillna("")
    return "Журнал рецензий:\n" + j.drop(columns=["шкала"]).to_csv(index=False)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="вопрос. Без него начинается диалог")
    ap.add_argument("--agent", default="l1", choices=["l1", "l2", "starter"])
    ap.add_argument("--plain", action="store_true", help="без инструментов")
    ap.add_argument("--status", action="store_true", help="показать состояние модели и выйти")
    ap.add_argument("--warm", action="store_true", help="прогреть модель коротким запросом")
    args = ap.parse_args()

    llm, note = pick_llm()
    print("модель:", note)
    if args.status:
        return
    if llm is None:
        sys.exit("модели нет. Записи прогонов: python3 agent_l1.py, python3 agent_l2.py, блокноты practice_*.ipynb")
    if args.warm:
        print(warm_up()); return

    if args.plain:
        ctx = journal_text()
        def answer(q):
            text, sec = ask_plain(q, ctx)
            print(f"\n{text}\n\n[без инструментов, {sec} с]")
    else:
        system, tools, impl, default_q = load_agent(args.agent)
        print("агент:", args.agent, "| инструменты:", ", ".join(impl))
        def answer(q):
            t = time.time()
            run_agent(q, llm, system, tools, impl)
            print(f"\n[агент {args.agent}, {round(time.time() - t, 1)} с]")

    if args.question:
        answer(args.question); return
    print("Диалог. Каждый вопрос задаётся заново, без памяти о прошлых. Пустая строка завершает.")
    while True:
        try:
            q = input("\nвопрос> ").strip()
        except EOFError:
            break
        if not q:
            break
        answer(q)

if __name__ == "__main__":
    main()
