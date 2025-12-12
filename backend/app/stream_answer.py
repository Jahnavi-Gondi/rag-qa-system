# backend/app/stream_answer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.app.llm import stream_llm_response

router = APIRouter()


@router.get("/api/stream")
async def stream_answer(query: str):

    def event_generator():
        try:
            # stream_llm_response is a normal generator → use normal for-loop
            for chunk in stream_llm_response(query):
                yield f"data: {chunk}\n\n"

            # end-of-stream marker
            yield "data: [END]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [END]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
