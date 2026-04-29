from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class VerifyRequest(BaseModel):
    prompt: str


@router.post("/verify")
async def verify(body: VerifyRequest, request: Request):
    result = await request.app.state.verifier.verify(body.prompt)
    return {"result": result}


@router.post("/verify-one-pass")
async def verify_one_pass(body: VerifyRequest, request: Request):
    result, preprocessed_prompt = await request.app.state.verifier.verify_one_pass(body.prompt)
    return {"result": result, "preprocessed_prompt": preprocessed_prompt}
