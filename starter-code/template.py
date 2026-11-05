"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()

    respone = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    latency = time.time() - start
    response_text = respone.choices[0].message.content or ""
    usage = {
        "input_tokens": respone.usage.prompt_tokens if respone.usage else 0,
        "output_tokens": respone.usage.completion_tokens if respone.usage else 0,
    }
    
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )
        start = time.time()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        latency_seconds = time.time() - start
        response_text = response.text or ""
        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

    except ImportError:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_inst = legacy_genai.GenerativeModel(model)
        generation_config = legacy_genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        start_time = time.time()
        response = model_inst.generate_content(
            prompt,
            generation_config=generation_config,
        )
        latency_seconds = time.time() - start_time

        response_text = response.text or ""
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    return response_text,latency_seconds,usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency_seconds = time.time() - start_time

    response_text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }

    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:

    def _calc_cost(model_name: str, in_tok: int, out_tok: int) -> float:
        rates = PRICING_1M_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})
        return (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000.0

    # 1. Gọi GPT-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = _calc_cost(OPENAI_MODEL, gpt4o_usage["input_tokens"], gpt4o_usage["output_tokens"])

    # 2. Gọi GPT-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    mini_cost = _calc_cost(OPENAI_MINI_MODEL, mini_usage["input_tokens"], mini_usage["output_tokens"])

    # 3. Gọi Gemini 2.5 Flash
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_cost = _calc_cost(GEMINI_MODEL, gemini_usage["input_tokens"], gemini_usage["output_tokens"])

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:

    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict] = []

    print("=== Interactive Gemini 2.5 Chatbot Started (type 'quit' or 'exit' to quit) ===")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        # Thêm lượt của user
        history.append({"role": "user", "parts": [user_input]})

        # Giữ tối đa 3 turns gần nhất (mỗi turn gồm user + model = 6 messages)
        active_contents = history[-6:]

        print("Gemini: ", end="", flush=True)
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=active_contents,
        )

        full_reply = ""
        for chunk in response_stream:
            text = chunk.text or ""
            print(text, end="", flush=True)
            full_reply += text
        print()

        history.append({"role": "model", "parts": [full_reply]})


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        comp_dict = compare_models(prompt)
        comp_dict["prompt"] = prompt
        results.append(comp_dict)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for item in results:
        prompt_text = item.get("prompt", "").replace("\n", " ")
        if len(prompt_text) > 40:
            prompt_text = prompt_text[:37] + "..."

        models = [
            ("gpt4o", "GPT-4o"),
            ("gpt4o_mini", "GPT-4o Mini"),
            ("gemini_flash", "Gemini-Flash"),
        ]

        for key, display_name in models:
            if key not in item:
                continue
            data = item[key]
            # Truncate response tối đa 50 ký tự theo yêu cầu đề bài
            raw_resp = (data.get("response") or "").replace("\n", " ").strip()
            truncated_resp = (raw_resp[:47] + "...") if len(raw_resp) > 50 else raw_resp

            latency = f"{data.get('latency', 0.0):.2f}s"
            tokens = f"{data.get('input_tokens', 0)}/{data.get('output_tokens', 0)}"
            cost = f"${data.get('cost', 0.0):.6f}"

            lines.append(
                f"| {prompt_text} | {display_name} | {truncated_resp} | {latency} | {tokens} | {cost} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
