# Kripos Demo: Skalerbar On-Prem ML Plattform

En demonstrasjonsplattform for datavitenskapsarbeid med GPU-tilgang, designet for å tilby både enkle og avanserte distribusjonsalternativer.

*[English version below](#kripos-demo-scalable-on-prem-ml-platform)*

## Oversikt

Dette prosjektet demonstrerer hvordan man bygger en skalerbar datavitenskapsplattform på eget utstyr (on-premises) med GPU-støtte. Plattformen er designet for å støtte dataforskere med avanserte verktøy mens den håndterer kompleksiteten under overflaten.

### Hovedfunksjoner

- **Selvbetjeningsplattform** - Intuitiv brukergrensesnitt for dataforskere
- **GPU-tilgang** - Optimalisert hardwarebruk for ML-trening
- **Sikkerhet og tilgangskontroll** - Robust autentisering og autorisering 
- **Skalerbarhet** - Vokser med behov, fra små team til større organisasjoner

## Installasjonsalternativer

Plattformen kan implementeres på to måter:

### 1. Enkel Oppsett (Docker)

Ideell for:
- Små team som kommer i gang
- Proof of concept
- Lokal utvikling

### 2. Fullstendig Oppsett (Kubernetes)

Ideell for:
- Bedriftsimplementering
- Miljøer med flere team
- Produksjonsarbeid

## Forutsetninger

Installer følgende verktøy på din Mac:

# Package manager
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Core tools
brew install docker
brew install kubectl
brew install minikube
brew install helm
brew install kind
brew install jq
brew install mkcert # for HTTPS
brew install node

# Optional for GUI-based logs and YAMLs
brew install lens
brew install k9s

# Python tools
brew install pyenv
brew install poetry

## Docker Oppsett (Enkel)

For rask oppstart med enkel konfigurasjon:

1. Opprett en `docker-compose.yml` fil med følgende tjenester:
   - JupyterHub: Flerbruker notatboksmiljø
   - MLflow: ML eksperiment og modellsporing
   - MinIO: Objektlagring for datasett og modeller

2. Start tjenestene:
   ```bash
   docker-compose up -d
   ```

3. Tilgang til tjenestene:
   - JupyterHub: http://localhost:8000
   - MLflow: http://localhost:5000
   - MinIO: http://localhost:9000 (konsoll: http://localhost:9001)

## Kubernetes Oppsett (Fullstendig)

For skalerbar bedriftsimplementering:

1. Start Minikube med GPU-støtte:
   ```bash
   minikube start --driver=docker --addons=gpu
   ```

2. Installer JupyterHub med Helm:
   ```bash
   helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
   helm repo update
   helm install jhub jupyterhub/jupyterhub --values jupyterhub-values.yaml --namespace jhub --create-namespace
   ```

3. Konfigurer persistente volumer og GPU-tilgang:
   ```bash
   kubectl apply -f k8s/pvc.yaml
   kubectl label nodes minikube gpu=true
   kubectl label nodes minikube accelerator=nvidia
   ```

4. Installer ytterligere komponenter:
   ```bash
   kubectl create namespace mlflow
   helm install mlflow ./charts/mlflow --namespace mlflow
   helm install minio ./charts/minio --namespace minio
   ```

## Sikkerhet og Tilgang

Plattformen implementerer flere sikkerhetsnivåer:

- **Autentisering**: Single Sign-On via Keycloak eller lokal autentisering
- **Autorisering**: Navneområder og RBAC-policyer i Kubernetes
- **Hemmeligheter**: Sikker håndtering med HashiCorp Vault eller Kubernetes Secrets
- **Kryptert Lagring**: Data krypteres både i hvile og under overføring

## Valgfrie Tillegg

Avhengig av behov kan du utvide plattformen med:

- **Keycloak**: For bedre SSO og identitetsstyring
- **Prometheus + Grafana**: For overvåking og varsling
- **Argo Workflows**: For ML pipeline-orkestrering
- **MLflow**: For avansert modellstyring

## Presentasjon

En komplett presentasjon finnes i `/presentation`-mappen, tilgjengelig på både engelsk og norsk.

## Anbefalte Implementeringsstrategier

- **Start Lite**: Begynn med Docker Compose for raske gevinster
- **Design for Vekst**: Planlegg arkitektur som kan utvikle seg med behov
- **Modulær Tilnærming**: Legg til komponenter gradvis ettersom adopsjonen øker

## Lisens

Dette prosjektet er distribuert under MIT-lisensen.

---

# Kripos Demo: Scalable On-Prem ML Platform

A demonstration platform for data science work with GPU access, designed to offer both simple and advanced deployment options.

## Overview

This project demonstrates how to build a scalable data science platform on-premises with GPU support. The platform is designed to support data scientists with advanced tools while handling the complexity under the hood.

### Key Features

- **Self-Service Platform** - Intuitive user interface for data scientists
- **GPU Access** - Optimized hardware utilization for ML training
- **Security & Access Control** - Robust authentication and authorization
- **Scalability** - Grows with needs, from small teams to larger organizations

## Installation Options

The platform can be implemented in two ways:

### 1. Simple Setup (Docker)

Ideal for:
- Small teams getting started
- Proof of concept
- Local development

### 2. Full Setup (Kubernetes)

Ideal for:
- Enterprise implementation
- Multi-team environments
- Production workloads

## Prerequisites

Install the following tools on your Mac:

```bash
# Package manager
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Core tools
brew install docker
brew install kubectl
brew install minikube
brew install helm
brew install kind
brew install jq
brew install mkcert # for HTTPS
brew install node

# Optional for GUI-based logs and YAMLs
brew install lens
brew install k9s

# Python tools
brew install pyenv
brew install poetry
```

## Docker Setup (Simple)

For quick start with simple configuration:

1. Create a `docker-compose.yml` file with the following services:
   - JupyterHub: Multi-user notebook environment
   - MLflow: ML experiment and model tracking
   - MinIO: Object storage for datasets and models

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Access the services:
   - JupyterHub: http://localhost:8000
   - MLflow: http://localhost:5000
   - MinIO: http://localhost:9000 (console: http://localhost:9001)

## Kubernetes Setup (Full)

For scalable enterprise implementation:

1. Start Minikube with GPU support:
   ```bash
   minikube start --driver=docker --addons=gpu
   ```

2. Install JupyterHub with Helm:
   ```bash
   helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
   helm repo update
   helm install jhub jupyterhub/jupyterhub --values jupyterhub-values.yaml --namespace jhub --create-namespace
   ```

3. Configure persistent volumes and GPU access:
   ```bash
   kubectl apply -f k8s/pvc.yaml
   kubectl label nodes minikube gpu=true
   kubectl label nodes minikube accelerator=nvidia
   ```

4. Install additional components:
   ```bash
   kubectl create namespace mlflow
   helm install mlflow ./charts/mlflow --namespace mlflow
   helm install minio ./charts/minio --namespace minio
   ```

## Security and Access

The platform implements multiple security layers:

- **Authentication**: Single Sign-On via Keycloak or local authentication
- **Authorization**: Namespaces and RBAC policies in Kubernetes
- **Secrets**: Secure handling with HashiCorp Vault or Kubernetes Secrets
- **Encrypted Storage**: Data is encrypted both at rest and in transit

## Optional Add-ons

Depending on needs, you can extend the platform with:

- **Keycloak**: For better SSO and identity management
- **Prometheus + Grafana**: For monitoring and alerting
- **Argo Workflows**: For ML pipeline orchestration
- **MLflow**: For advanced model management

## Presentation

A complete presentation is available in the `/presentation` folder, available in both English and Norwegian.

## Recommended Implementation Strategies

- **Start Small**: Begin with Docker Compose for quick wins
- **Design for Growth**: Plan architecture that can evolve with needs
- **Modular Approach**: Add components incrementally as adoption grows

## License

This project is distributed under the MIT license.

