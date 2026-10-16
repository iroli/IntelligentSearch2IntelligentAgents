"""Ядро мини-агента: модель и цикл. От предметной области не зависит."""
import json, os

# ---------------------------------------------------------------- модель
def openai_llm(messages, tools):
    """Настоящая модель через OpenAI-совместимый API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"))
    r = client.chat.completions.create(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                                       messages=messages, tools=tools)
    return r.choices[0].message.model_dump(exclude_none=True)

def scripted_llm(plan, answer):
    """Модель-заглушка: выдаёт вызовы из plan по одному, затем записанный ответ."""
    steps = iter(plan)
    def llm(messages, tools):
        step = next(steps, None)
        if step is None:
            return {"role": "assistant", "content": answer}
        name, args = step
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{len(messages)}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}
    return llm

# ---------------------------------------------------------------- ведущее приложение: цикл
def run_agent(question, llm, system, tools, impl, max_steps=8):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        m = llm(messages, tools)                      # модель решает, что делать дальше
        messages.append(m)
        calls = m.get("tool_calls") or []
        if not calls:                                 # вызовов нет: это ответ
            print(f"[{step}] ответ модели:\n{m['content']}")
            return messages
        for tc in calls:                              # приложение исполняет вызовы
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            try:
                result = impl[name](**args)
            except Exception as e:                    # ошибка возвращается модели как результат
                result = {"ошибка": f"{type(e).__name__}: {e}"}
            text = json.dumps(result, ensure_ascii=False)
            print(f"[{step}] модель вызвала {name}({args})")
            print(f"    <- {text[:160]}{'…' if len(text) > 160 else ''}")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
    print("лимит шагов исчерпан")
    return messages
