@echo off
curl -s -N -X POST http://127.0.0.1:11434/api/pull -H "Content-Type: application/json" -d "{\"model\":\"ggml-org/gpt2\",\"file\":\"ggml-model-q4_0.gguf\"}"