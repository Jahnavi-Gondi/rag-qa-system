# backend/app/llm.py
import ollama

MODEL_NAME = "llama3.2:1b"   # or "phi3:mini"


def call_llm_sync(prompt: str) -> str:
    """
    Non-streaming call, used for answer_query() in backend.
    """
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            stream=False
        )
        return response.get("response", "")

    except Exception as e:
        return f"[LLM Error] {e}"


def stream_llm_response(prompt: str):
    """
    Streaming generator for SSE.
    Must be a normal (non-async) Python generator.
    """
    try:
        stream = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            stream=True
        )

        for part in stream:
            # Ollama streaming chunks appear inside part["response"]
            chunk = part.get("response", "")
            if chunk and isinstance(chunk, str):
                yield chunk

    except Exception as e:
        # If the model crashes mid-stream, at least yield the error
        yield f"[LLM Error] {str(e)}"

