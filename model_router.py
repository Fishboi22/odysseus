# model_router.py
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# ---------- Configuration ----------
DEFAULT_NUM_CTX = 16384
DEFAULT_MAX_TOKENS = 8192
APPEND_DONE_MARKER = True   # Set to False to disable

MODELS = {
    "fast": {
        "client": AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
        "model": "phi4-mini:3.8b",
        "temperature": 0.1,
        "system_prompt": "Answer in one very short sentence or just a few words. Be direct."
    },
    "general": {
        "client": AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
        "model": "FieldMouse-AI/qwen3.5:9b-Q5_K_M-instruct",
        "temperature": 0.4,
        "max_tokens": 16000,
        "system_prompt": (
            "You are a helpful, knowledgeable assistant. Provide clear, factual, unbiased explanations. "
            "Use examples or metaphors when helpful, but keep answers straightforward. "
            "Aim for easy‑to‑understand language."
        )
    },
    "reasoning": {
        "client": AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
        "model": "olmo-3:7b",
        "temperature": 0.4,
        "system_prompt": (
            "You are a reasoning assistant. Explain step by step clearly and concisely. "
            "Do not repeat the prompt, speculate about formatting, or include meta‑instructions. "
            "If the question is unclear, ask a follow up question for more information. "
            "Do not use web_search for simple mathematical equations. "
            "Before searching, think if you know the answer."
        )
    },
    "coding": {
        "client": AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
        "model": "qwen2.5-coder:7b",
        "temperature": 0.3,
        "system_prompt": (
            "You are a helpful coding tutor. Explain code concepts clearly, provide examples, "
            "and teach best practices. Output code with comments and explanations. "
            "Be patient and educational."
        )
    },
}

CLASSIFIER_MODEL = "phi4-mini:3.8b"

# ---------- Keyword fallback ----------
def keyword_category(prompt: str) -> str:
    pl = prompt.lower()
    if any(word in pl for word in ["```python", "def ", "class ", "function ", "code", "debug"]):
        return "coding"
    if any(word in pl for word in ["reason", "logic", "explain step", "deduce", "prove"]):
        return "reasoning"
    if any(word in pl for word in ["fast", "quick", "short"]):
        return "fast"
    return "general"

# ---------- Enhanced classifier: returns (category, needs_search) ----------
async def classify_with_small_llm(prompt: str) -> tuple[str, bool]:
    """Return (category, needs_search)."""
    client = AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
    try:
        resp = await client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a router. Perform two tasks:\n"
                    "1. Classify the user's request into exactly one of these categories: "
                    "fast, general, reasoning, coding.\n"
                    "2. Decide whether this request requires a live web search to answer correctly.\n"
                    "Output format:\n"
                    "CATEGORY: <category>\n"
                    "SEARCH: <true/false>"
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0,
        )
        output = resp.choices[0].message.content.strip()
        category = "general"
        needs_search = False
        for line in output.split('\n'):
            if line.startswith("CATEGORY:"):
                cat = line.split("CATEGORY:", 1)[1].strip().lower()
                if cat in MODELS:
                    category = cat
            elif line.startswith("SEARCH:"):
                val = line.split("SEARCH:", 1)[1].strip().lower()
                needs_search = (val == "true")
        return category, needs_search
    except Exception as e:
        print(f"LLM classifier error: {e}, falling back to keyword matcher")
        # Fallback to keyword matcher for category
        cat = keyword_category(prompt)
        # Heuristic: if query contains words like "latest", "news", "price", "today" -> search needed
        need = any(kw in prompt.lower() for kw in ("latest", "news", "price", "cost", "today", "current", "2026"))
        return cat, need

# ---------- Helper: sanitize messages (removes UNTRUSTED markers) ----------
def sanitize_messages(messages):
    cleaned = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str) and content.startswith(("UNTRUSTED", "[")):
            continue
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            full_text = " ".join(text_parts)
            if full_text.startswith(("UNTRUSTED", "[")):
                continue
        cleaned.append(m)
    return cleaned

# ---------- Context coherence check ----------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedder

def is_context_shift(messages, last_user_msg, threshold=0.4):
    """Return True if the last assistant message and the new user message are unrelated."""
    if len(messages) < 2:
        return False
    last_assistant = None
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant = m["content"]
            break
    if not last_assistant:
        return False
    embedder = get_embedder()
    emb_assistant = embedder.encode(last_assistant, convert_to_tensor=True)
    emb_user = embedder.encode(last_user_msg, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb_assistant, emb_user).item()
    print(f"Context similarity: {similarity:.3f} (threshold={threshold})")
    return similarity < threshold

# ---------- Streaming generator ----------
async def stream_model(config, messages, temperature, max_tokens):
    system_prompt = config.get("system_prompt")
    if system_prompt:
        has_system = any(m.get("role") == "system" and system_prompt in m.get("content", "") for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        stream = await config["client"].chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"num_ctx": DEFAULT_NUM_CTX},
            stream=True,
        )
        collected = False
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                collected = True
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
        if collected:
            if APPEND_DONE_MARKER:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': ' [done]'}}]})}\n\n"
            yield "data: [DONE]\n\n"
        else:
            raise ValueError("Empty stream")
    except Exception as e:
        print(f"Streaming error, falling back to non-streaming: {e}")
        try:
            resp = await config["client"].chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=0.9,
                max_tokens=max_tokens,
                extra_body={"num_ctx": DEFAULT_NUM_CTX},
                stream=False,
            )
            content = resp.choices[0].message.content or "I'm sorry, I couldn't generate a response. Please try again."
            if APPEND_DONE_MARKER:
                content += " [done]"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {e2}'}}]})}\n\n"
            yield "data: [DONE]\n\n"

# ---------- Endpoints ----------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "router",
            "object": "model",
            "context_length": 128000
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    print(f"Router received {len(messages)} messages, stream={stream}")
    if messages:
        print(f"Last message: {messages[-1].get('content', '')[:200]}")

    if not messages:
        raise HTTPException(400, "No messages")

    # Sanitize messages (removes UNTRUSTED markers)
    clean_messages = sanitize_messages(messages)

    # Extract last real user message for classification
    last_msg = ""
    for m in reversed(clean_messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                last_msg = content
                break
            elif isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                if text_parts:
                    last_msg = " ".join(text_parts)
                    break
    if not last_msg:
        last_msg = "general question"

    # --- Context coherence check ---
    if len(messages) > 1 and is_context_shift(messages, last_msg, threshold=0.4):
        print("Context shift detected – injecting coherence reminder.")
        reminder = {
            "role": "system",
            "content": (
                "The previous assistant response was about a completely different topic. "
                "Please disregard it and answer the user's new question directly. "
                "Do not repeat or refer to the earlier unrelated conversation."
            )
        }
        insert_pos = 0
        for i, msg in enumerate(clean_messages):
            if msg.get("role") == "system":
                insert_pos = i + 1
                break
        clean_messages.insert(insert_pos, reminder)

    # --- Classification + search decision ---
    category, needs_search = await classify_with_small_llm(last_msg)
    config = MODELS.get(category, MODELS["general"])
    print(f"Routing to {category} -> {config['model']}, needs_search={needs_search}")
    print(f"Query: {last_msg[:200]}")

    # Inject anti‑search instruction if the classifier said search is NOT needed
    if not needs_search:
        print("Search not required – injecting no‑search instruction.")
        no_search_msg = {
            "role": "system",
            "content": (
                "IMPORTANT: For this request, you must NOT use the web_search tool. "
                "Answer using your internal knowledge only. Do not call web_search or any other web‑related tool."
            )
        }
        # Insert after existing system prompts (but before the user message)
        insert_pos = 0
        for i, msg in enumerate(clean_messages):
            if msg.get("role") == "system":
                insert_pos = i + 1
                break
        clean_messages.insert(insert_pos, no_search_msg)

    # Use category‑specific max_tokens if defined, else default
    max_tokens = config.get("max_tokens", body.get("max_tokens", DEFAULT_MAX_TOKENS))
    temperature = config.get("temperature", body.get("temperature", 0.5))

    # --- Non-streaming path (for auto‑name etc.) ---
    if not stream:
        try:
            resp = await config["client"].chat.completions.create(
                model=config["model"],
                messages=clean_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"num_ctx": DEFAULT_NUM_CTX},
                stream=False,
            )
            content = resp.choices[0].message.content or ""
            if APPEND_DONE_MARKER:
                content += " [done]"
            return {
                "id": "chatcmpl-router",
                "object": "chat.completion",
                "created": 0,
                "model": config["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        except Exception as e:
            print(f"Non-stream model error: {e}")
            raise HTTPException(502, f"Model error: {e}")

    # --- Streaming path ---
    return StreamingResponse(
        stream_model(config, clean_messages, temperature, max_tokens),
        media_type="text/event-stream"
    )