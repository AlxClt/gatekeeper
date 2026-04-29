import os
from datetime import datetime, timezone

import asyncpg


class DBLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._pool: asyncpg.Pool | None = None

    async def connect(self):
        if not self.enabled:
            return
        self._pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def log(self, prompt: str, result: int):
        if not self.enabled or self._pool is None:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO logs (prompt, result, created_at) VALUES ($1, $2, $3)",
                prompt,
                result,
                datetime.now(timezone.utc),
            )
