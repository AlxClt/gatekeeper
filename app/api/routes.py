from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class VerifyRequest(BaseModel):
    prompt: str


@router.post("/verify")
async def verify(body: VerifyRequest, request: Request):
    result, preprocessed_prompt = await request.app.state.verifier.verify(body.prompt)
    return {"result": result, "preprocessed_prompt": preprocessed_prompt}


@router.post("/verify-raw")
async def verify_raw(body: VerifyRequest, request: Request):
    result = await request.app.state.verifier.verify_raw(body.prompt)
    return {"result": result}
