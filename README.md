# mlops-pytorch-pipeline

A production-style ML pipeline that trains and serves a ResNet-18 image classifier on CIFAR-10 using Docker and Kubernetes.

## Architecture

```
GitHub (source) --> CI (lint + test + docker build) --> Docker images
                                                              |
                                                   Kubernetes cluster
                                                   training-job (K8s Job)
                                                              |
                                                   model checkpoint (PVC)
                                                              |
                                                   serving-deployment (K8s Deployment)
                                                              |
                                                   serving-service (ClusterIP)
```

## Project Structure

```
mlops-pytorch-pipeline/
├── .github/workflows/ci.yml      # GitHub Actions: lint + test + docker build
├── configs/training_config.yaml  # Hyperparameters
├── docker/
│   ├── Dockerfile.train          # Multi-stage training image
│   └── Dockerfile.serve          # Slim serving image (non-root, HEALTHCHECK)
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── src/
│   ├── model.py                  # ResNet-18 wrapper
│   ├── dataset.py                # CIFAR-10 data loading
│   ├── train.py                  # Training loop (JSON-line logs, early stopping)
│   └── serve.py                  # FastAPI prediction server
└── tests/test_model.py
```

## Quick Start (Local)

### Train
```bash
pip install -r requirements/train.txt
python src/train.py
```

### Serve
```bash
pip install -r requirements/serve.txt
CHECKPOINT_PATH=checkpoints/classifier_v1.pt python src/serve.py
# POST http://localhost:8080/predict  (form field: image=@file.png)
# GET  http://localhost:8080/health
```

## Docker

```bash
# Training
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Serving
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# after job completes:
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# test
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Git Workflow

- `main` — stable, only PR merges
- `develop` — integration branch
- `feature/*` — all new work; merged to develop via PR
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONFIG_PATH` | `/app/configs/training_config.yaml` | Training config path |
| `CHECKPOINT_PATH` | `/app/checkpoints/classifier_v1.pt` | Checkpoint for serving |
| `PORT` | `8080` | Server port |
