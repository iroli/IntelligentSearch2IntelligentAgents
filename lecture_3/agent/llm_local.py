"""Подключение модели для практикума: Ollama на этой машине или любой OpenAI-совместимый сервер.

pick_llm() возвращает (функция-модель или None, описание).
Если модель недоступна или отвечает ошибкой, блокноты показывают записи. Инструменты при этом работают по-настоящему.

Переменные окружения (все необязательны):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL   внешний сервер; при заданном ключе Ollama не используется
    OLLAMA_HOST                            адрес Ollama, по умолчанию 127.0.0.1:11434
    AGENT_MODEL                            имя модели в Ollama, по умолчанию qwen-agent
    LLM_TIMEOUT                            секунд на один ответ модели, по умолчанию 180
"""
import json, os, time, urllib.request
from agent_core import run_agent, scripted_llm

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen-agent")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))

def _local(url):
    return any(h in url for h in ("127.0.0.1", "localhost", "[::1]"))

def _client():
    """Клиент OpenAI. Для локального сервера прокси из переменных окружения не используется."""
    from openai import OpenAI
    base = os.environ.get("LLM_BASE_URL") or ""
    kw = dict(api_key=os.environ["LLM_API_KEY"], base_url=base or None, timeout=TIMEOUT, max_retries=0)
    if _local(base):
        import httpx
        kw["http_client"] = httpx.Client(trust_env=False, timeout=TIMEOUT)
    return OpenAI(**kw)

def ollama_models(host=OLLAMA_HOST):
    """Имена моделей на сервере Ollama. Пустой список, если сервер не отвечает. Прокси не используется."""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{host}/api/tags", timeout=2) as r:
            return sorted(m.get("name") or m.get("model") for m in json.load(r).get("models", []))
    except Exception:
        return []

def openai_llm(messages, tools):
    """Настоящая модель через OpenAI-совместимый API, с ограничением времени ответа."""
    client = _client()
    r = client.chat.completions.create(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"), messages=messages, tools=tools)
    return r.choices[0].message.model_dump(exclude_none=True)

NUDGE = "Ответ без вызова инструментов не принимается. Сначала вызови нужные инструменты, затем отвечай по их результатам."

def tools_first(llm, attempts=2):
    """Обёртка над моделью: пока в разговоре нет ни одного результата инструмента, ответ без вызовов
    не принимается. Модели возвращается напоминание, и она отвечает ещё раз (до attempts раз).
    Маленькие модели иначе отвечают сразу «[не подтверждено]», не заглянув в данные."""
    def wrapped(messages, tools):
        m = llm(messages, tools)
        if not tools or any(x.get("role") == "tool" for x in messages):
            return m
        extra = []
        for _ in range(attempts):
            if m.get("tool_calls"):
                return m
            print("    [ответ без вызова инструментов, модели отправлено напоминание]")
            extra += [m, {"role": "user", "content": NUDGE}]
            m = llm(messages + extra, tools)
        return m
    return wrapped

def pick_llm():
    """Выбирает модель: сначала переменные окружения, затем локальная Ollama, иначе записи."""
    if os.environ.get("LLM_API_KEY"):
        return tools_first(openai_llm), f"модель {os.environ.get('LLM_MODEL', 'gpt-4o-mini')} через {os.environ.get('LLM_BASE_URL', 'адрес по умолчанию')}"
    names = ollama_models()
    if not names:
        return None, "Ollama не отвечает или моделей нет: показываем записи, инструменты работают по-настоящему"
    want = AGENT_MODEL if ":" in AGENT_MODEL else AGENT_MODEL + ":latest"
    if want in names:
        chosen, note = AGENT_MODEL, f"локальная модель {AGENT_MODEL} через Ollama"
    else:
        # производной модели нет: берём первую подходящую, с предупреждением об окне контекста
        cands = [n for n in names if "qwen" in n.lower()] or names
        chosen = cands[0]
        note = (f"модели {AGENT_MODEL} нет, взята {chosen} через Ollama. Окно контекста у неё по умолчанию 2048: "
                f"для агента лучше собрать qwen-agent (ollama create qwen-agent -f Modelfile)")
    os.environ["LLM_API_KEY"] = "ollama"
    os.environ["LLM_BASE_URL"] = f"http://{OLLAMA_HOST}/v1"
    os.environ["LLM_MODEL"] = chosen
    return tools_first(openai_llm), note

def ask_plain(question, context="", system=None):
    """Вопрос модели без инструментов. Возвращает (текст ответа, секунды)."""
    client = _client()
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": question + ("\n\n" + context if context else "")}]
    t = time.time()
    r = client.chat.completions.create(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"), messages=messages)
    return r.choices[0].message.content, round(time.time() - t, 1)

def ask_plain_or_record(llm, question, context, record):
    """Вопрос без инструментов: живой ответ, при отказе модели запись."""
    if llm is None:
        print(record); return record
    try:
        text, sec = ask_plain(question, context)
        print(text, f"\n\n[без инструментов, {sec} с]"); return text
    except Exception as e:
        print(f"[модель не ответила: {type(e).__name__}. Показываем запись]\n")
        print(record); return record

def warm_up():
    """Короткий запрос, чтобы модель загрузилась в память до показа."""
    text, sec = ask_plain("Ответь одним словом: готов?")
    return f"модель ответила за {sec} с: {text.strip()[:40]}"

def run_live_or_record(llm, question, system, tools, impl, plan, record):
    """Прогон агента: с моделью, если она есть, иначе запись (инструменты вызываются по-настоящему).
    Если модель отвечает ошибкой или не укладывается в LLM_TIMEOUT, прогон повторяется по записи."""
    t = time.time()
    if llm is not None:
        try:
            messages = run_agent(question, llm, system, tools, impl)
            print(f"\n[время: {round(time.time() - t, 1)} с]")
            return messages
        except Exception as e:
            print(f"\n[модель не ответила: {type(e).__name__}: {str(e)[:120]}. Повторяем по записи]\n")
    else:
        print("[модели нет: порядок вызовов и текст ответа взяты из записи, результаты инструментов настоящие]")
    t = time.time()
    messages = run_agent(question, scripted_llm(plan, record), system, tools, impl)
    print(f"\n[время: {round(time.time() - t, 1)} с]")
    return messages
