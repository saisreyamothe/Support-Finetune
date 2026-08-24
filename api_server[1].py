"""
FastAPI server for QLoRA customer support inference.
Endpoint: POST /generate - Sub-100ms response time
"""

import time
from fastapi import FastAPI
from pydantic import BaseModel
from inference import CustomerSupportInference
import uvicorn


app = FastAPI(title="QLoRA Customer Support API", version="1.0")
engine = CustomerSupportInference()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 300
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate customer support response."""
    start = time.time()
    
    response = engine.answer(
        question=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    latency = (time.time() - start) * 1000
    tokens = len(response.split())
    
    return GenerateResponse(
        response=response,
        tokens_generated=tokens,
        latency_ms=round(latency, 1),
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "model": "qlora-llama-support"}


@app.post("/batch")
async def batch_generate(prompts: list[str], max_tokens: int = 300):
    """Generate for multiple prompts."""
    start = time.time()
    responses = engine.batch_answer(prompts, max_tokens)
    latency = (time.time() - start) * 1000
    
    return {
        "responses": responses,
        "count": len(responses),
        "latency_ms": round(latency, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
