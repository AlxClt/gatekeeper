from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class VerifyRequest(BaseModel):
    prompt: str


class VerifyResponse(BaseModel):
    result: int
    preprocessed_prompt: str


class VerifyRawResponse(BaseModel):
    result: int


@router.post("/verify", response_model=VerifyResponse)
async def verify(body: VerifyRequest, request: Request):
    result, preprocessed_prompt = await request.app.state.verifier.verify(body.prompt)
    return VerifyResponse(result=result, preprocessed_prompt=preprocessed_prompt)


@router.post("/verify-raw", response_model=VerifyRawResponse)
async def verify_raw(body: VerifyRequest, request: Request):
    result = await request.app.state.verifier.verify_raw(body.prompt)
    return VerifyRawResponse(result=result)
