import json
import os
import time
import urllib.request
from pathlib import Path

URL = os.getenv("GATEKEEPER_URL", "http://localhost:8000") + "/verify"
PROMPT_FILE = Path(__file__).parent / "demo_prompt.txt"

prompt = PROMPT_FILE.read_text().strip()
print(f"Prompt : {prompt}")
print(f"Sending to {URL} ...\n")

payload = json.dumps({"prompt": prompt}).encode()
req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})

t0 = time.perf_counter()
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())
elapsed = time.perf_counter() - t0

label = "THREAT DETECTED" if result["result"] == 1 else "No threat"
print(f"Result  : {result['result']}  —  {label}")
print(f"Time    : {elapsed:.2f}s")
