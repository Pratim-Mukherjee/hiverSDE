# API Service

This folder contains the lightweight FastAPI service for the AmazonHelp support agent.

## Run

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Health check

```bash
curl http://127.0.0.1:8000/health
```

## Predict endpoint

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"customer_text":"My order has not arrived after 9 days"}'
```
