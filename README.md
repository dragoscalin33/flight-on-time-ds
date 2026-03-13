# FlightOnTime — Intelligent Flight Delay Prediction System

**Project Status:** Production (v6.0 — Production Pipeline)
**Live Demo:** [Launch App (Oracle Cloud)](http://flight-on-time.vm3.arbly.com/)
**Architecture:** Monorepo (Frontend + Backend + Data Science + ML Pipeline) | Oracle Cloud (OCI)

**FlightOnTime** is a full-stack solution for predicting commercial flight delays in Brazil. The system combines advanced Machine Learning, real-time weather data, and a microservices architecture to help passengers plan safer trips.

---

## Repository Structure

```
flight-on-time/
├── data-science/        # Original ML research (CatBoost v5.0, notebooks, prototyping)
├── data-science-prod/   # Production ML pipeline (v6.0 — Hydra + MLflow + K-Fold CV)
├── back-end/            # API Gateway (Java 21, Spring Boot)
├── front-end/           # Web Dashboard (React, Vite, Tailwind)
└── infrastructure/      # Oracle Cloud deployment configs
```

---

## System Architecture

### 1. Production ML Pipeline (NEW — v6.0)

**Directory:** [`/data-science-prod`](./data-science-prod)
[Full Technical Documentation](./data-science-prod/README.md)

The evolution from notebook-based research to a **production-grade ML pipeline**. This is the architecture you'd deploy in a real MLOps environment.

| Aspect | v5.0 (Notebook) | v6.0 (Production) |
|:-------|:-----------------|:--------------------|
| Configuration | Hardcoded in cells | YAML via **Hydra** |
| Experiment Tracking | `print()` to stdout | **MLflow** (params, metrics, artifacts) |
| Data Validation | None | **Pandera** schemas |
| Cross-Validation | Single holdout | **Stratified K-Fold** (5-fold) |
| Code Organization | Monolithic notebook | Modular `src/` packages |
| Model Artifact | Manual `.joblib` | **MLflow Registry** + `.joblib` |
| API | Coupled to artifact | Health check + dual model loading |

Key features: 10-step training pipeline, F2-optimized threshold tuning with business safety override (0.35), native CatBoost categorical support, Haversine distance computation, Brazilian holiday detection, live weather integration (Open-Meteo API).

### 2. ML Research & Prototyping (v5.0)

**Directory:** [`/data-science`](./data-science)
[Technical Documentation](./data-science/README.md)

The original research environment where the CatBoost model was developed. Includes Jupyter notebooks with exploratory analysis, modeling strategy iterations (v1–v5), and deep learning experiments with embeddings. CatBoost outperformed the neural approaches by ~10% ROC-AUC and was selected for production.

*Modelo / Model:* CatBoost Classifier with live weather integration (OpenMeteo)
*API:* FastAPI (Python)

### 3. Backend API

**Directory:** [`/back-end`](./back-end)
[Technical Documentation](./back-end/README.md)

The system orchestrator. Routes requests, connects to the ML engine, and applies business rules.

*Stack:* Java 21 + Spring Boot 3.5.4 + MySQL (Flyway migrations)

### 4. Frontend Dashboard

**Directory:** [`/front-end`](./front-end)
[Technical Documentation](./front-end/README.md)

The passenger-facing interface with intelligent airport autocomplete and IATA code validation.

*Stack:* React + Vite + Tailwind CSS

---

## Quick Start (Local)

You'll need 3 terminal windows to run the full stack locally.

### Prerequisites

- Python 3.10+
- Node.js 18+
- MySQL 8.0+
- Maven 3.8+

### Step 1: Start the ML Engine

```bash
# Using the production pipeline (v6.0):
cd data-science-prod
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make train                    # Train with Hydra + MLflow
make serve                    # Start FastAPI on :8000

# Or using the original engine (v5.0):
cd data-science
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.app:app --reload --port 8000
```

### Step 2: Start the Backend (Java)

```bash
cd back-end
./mvnw clean install
./mvnw spring-boot:run
```

### Step 3: Start the Frontend (React)

```bash
cd front-end
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Risk Semaphore (Business Rule)

The system translates the ML probability into a visual risk indicator:

| Risk Level | Probability | Action |
|:-----------|:------------|:-------|
| ON_TIME (green) | < 35% | Good conditions — no action needed |
| PREVENTIVE_ALERT (yellow) | 35% – 70% | Instability detected — monitor flight status |
| LIKELY_DELAYED (red) | > 70% | Critical conditions — consider contingency plan |

---

## Technology Stack

| Layer | Technologies |
|:------|:-------------|
| **Frontend** | React, Vite, Tailwind CSS |
| **Backend** | Java 21, Spring Boot 3.5.4, MySQL, Flyway |
| **ML Research** | Python, CatBoost, Jupyter, FastAPI |
| **ML Production** | Hydra, MLflow, Pandera, Stratified K-Fold CV |
| **External Data** | Open-Meteo API (real-time weather forecasts) |
| **Infrastructure** | Oracle Cloud Infrastructure (OCI), Docker |

---

## Dataset

- **Source:** Brazilian domestic flights (2015–2017) — Kaggle
- **Weather Enrichment:** Open-Meteo Historical API
- **Size:** 2.5M raw records → 2.2M after cleaning
- **Target:** Binary (1 = delay > 15 min, 0 = on-time)
- **Class Balance:** 88.4% on-time / 11.6% delayed

---

<details>
<summary><strong>Versao em Portugues / Portuguese Version</strong></summary>

O **FlightOnTime** e uma solucao completa para prever atrasos em voos comerciais no Brasil. O sistema combina Inteligencia Artificial avancada, dados meteorologicos em tempo real e uma arquitetura robusta de microservicos.

**Estrutura:** Monorepo com `data-science/` (pesquisa original), `data-science-prod/` (pipeline de producao com Hydra + MLflow), `back-end/` (Java/Spring Boot), `front-end/` (React/Vite) e `infrastructure/` (Oracle Cloud).

**Semaforo de Risco:** PONTUAL (< 35%), ALERTA PREVENTIVO (35–70%), ATRASO PROVAVEL (> 70%).

Para detalhes completos, consulte a documentacao tecnica em cada diretorio.

</details>
