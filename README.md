# zagents
Tiny and local agents for precise tasks

## Install as a package

From the repo root:

```bash
python -m pip install -U pip
pip install .
```

Development install:

```bash
pip install -e ".[dev]"
```

## Run the optional API locally

```bash
pip install -e ".[api]"
python -m uvicorn api.app.main:app --reload
```

## Build & run as a Docker image

```bash
docker build -f api/Dockerfile -t zagents-api .
docker run --rm -p 8000:8000 zagents-api
```
