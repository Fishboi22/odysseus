# model_router.py – Clean router with external web search
import json
import re
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

app = FastAPI()

# ---------- Configuration ----------
OLLAMA_URL = "http://127.0.0.1:11434/v1"
SEARXNG_URL = "http://localhost:8080/search"   # your SearXNG instance
CLASSIFIER_MODEL = "phi4-mini:3.8b"
MAIN_MODEL = "qwen3.6:27b"                     # change as needed
MAX_SEARCH_RESULTS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 1024

# ---------- Helper: get last user message ----------
def last_user_message(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        return part.get("text", "")
    return ""

# ---------- Helper: classify if web search is needed ----------
async def needs_web_search(query: str) -> bool:
    client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama")
    try:
        resp = await client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": "Decide if the user's question requires up‑to‑date information from the web. Answer only YES or NO."},
                {"role": "user", "content": query}
            ],
            max_tokens=5,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer == "YES"
    except Exception as e:
        print(f"Classifier error: {e}, defaulting to NO search")
        return False

# ---------- Helper: perform web search via SearXNG ----------
async def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(SEARXNG_URL, params={"q": query, "format": "json"}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])[:max_results]
            if not results:
                return "No search results found."
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:500]  # truncate
                formatted.append(f"{i}. **{title}**\n   URL: {url}\n   {content}")
            return "\n\n".join(formatted)
    except Exception as e:
        print(f"Search error: {e}")
        return f"Search failed: {e}"

# ---------- Endpoints ----------
@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "router"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    user_msg = last_user_message(messages)
    if not user_msg:
        return {"error": "No user message"}

    # Step 1: decide if web search is needed
    search_needed = await needs_web_search(user_msg)
    search_results = ""
    if search_needed:
        print(f"Web search needed for: {user_msg}")
        search_results = await web_search(user_msg)
        print(f"Got {len(search_results)} chars of search results")
    else:
        print(f"No web search needed for: {user_msg}")

    # Step 2: build clean conversation
    system_prompt = "Answer the user's question directly. Do not list URLs, do not mention tools, do not output search results. Just give the answer."
    if search_results:
        system_prompt += f"\n\nUse the following up‑to‑date information to answer:\n{search_results}"

    clean_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama")
    try:
        if stream:
            async def generate():
                resp = await client.chat.completions.create(
                    model=MAIN_MODEL,
                    messages=clean_messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
                async for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.choices[0].delta.content}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            resp = await client.chat.completions.create(
                model=MAIN_MODEL,
                messages=clean_messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            content = resp.choices[0].message.content
            return {"choices": [{"message": {"content": content}}]}
    except Exception as e:
        return {"error": str(e)}