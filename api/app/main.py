from __future__ import annotations

from fastapi import FastAPI

from zagents.core import hello

app = FastAPI(title="zagents API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/hello")
def hello_route(name: str = "world") -> dict[str, str]:
    return {"message": hello(name)}

