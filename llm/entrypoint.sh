#!/bin/bash
set -e

ollama serve &
SERVER_PID=$!
trap "kill $SERVER_PID" SIGTERM SIGINT

echo "Waiting for Ollama to start..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done

echo "Pulling model ${LLM_MODEL:-llama3.2}..."
ollama pull "${LLM_MODEL:-llama3.2}"

echo "Model ready."
wait "$SERVER_PID"
