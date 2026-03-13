# Flight On Time — Deployment on Oracle Cloud Infrastructure (OCI)

This repository documents the architecture and implementation of the **Flight On Time** project deployed on **Oracle Cloud Infrastructure (OCI)**, using **Docker containers** to isolate and orchestrate the application services.

The solution is designed to be **modular, scalable, and easily reproducible**, following infrastructure and DevOps best practices.

---

## Architecture Overview

All application components run in **Docker containers**, hosted on a **VM on OCI**.
External access is centralized through a **Caddy reverse proxy**, which manages routing and, optionally, TLS certificates.

```
                         ┌───────────────┐
                         │   Internet    │
                         └───────┬───────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │     Caddy       │
                        │  Reverse Proxy  │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼
        │                        │
        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐
│                 │     │                 │       │                 │
│    Frontend     │     │    Backend      │       │   Datascience   │
│     (React)     │─────│ (Java / Spring) ┼───────│    (Python)     │
│                 │     │                 │       │                 │
└─────────────────┘     └───────┬─────────┘       └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │      MySQL       │
                       │    Database      │
                       └──────────────────┘
```

---

## Solution Components

### Caddy (Reverse Proxy)
- Runs in a Docker container
- Acts as the single entry point for the application
- Responsible for:
    - HTTP/HTTPS routing
    - Reverse proxy to internal services
    - Automatic TLS certificate management
- Enables service exposure without direct coupling to containers

---

### Frontend (React)
- Frontend application built with **React**
- Runs in a Docker container
- Build generated in a controlled environment (`npm run build`)
- Served via a secondary Caddy container
- Responsible for:
    - User interface
    - Consuming APIs exposed by the Java Backend
- Does not directly access databases or internal services

---

### Backend (Java)
- Java application (Spring Boot)
- Runs in a Docker container based on **Temurin**
- Responsible for:
    - REST API exposure
    - Business rules
    - MySQL database integration
    - Communication with the Data Science service
- Build via **Maven**

---

### Data Science (Python)
- Python service running in a Docker container
- Responsible for:
    - Loading and executing predictive models
    - Data processing
    - Endpoint exposure (FastAPI)
- Consumed by the Java backend via HTTP
- Fully decoupled from the backend, allowing independent evolution

---

### MySQL
- Relational database running in a Docker container
- Responsible for storing:
    - Operational data
    - Historical data used by the model
- Persistence ensured via **Docker volumes**
- Not directly exposed to the internet (internal access only)

---

## OCI Infrastructure

- **Oracle Cloud Infrastructure (OCI)**
- **Compute Instance (Linux VM)**
- Docker and Docker Compose installed on the VM
- Containers running on the same instance
- Inter-service communication via **internal Docker network**
- OCI firewall allowing access only to necessary ports (e.g., 80/443)

---

## Containers & Orchestration

- All services are defined via **Docker Compose**
- Benefits:
    - Environment standardization
    - Easy deploy and rollback
    - Service isolation
    - Local and production reproducibility

---

## Communication Flow

1. User accesses the application via browser
2. Request reaches **Caddy**
3. Caddy serves the **React Frontend**
4. Frontend consumes **Java Backend** APIs
5. Backend accesses:
    - MySQL for persistent data
    - Data Science for model inference
6. Response returns to the user via Caddy

---

## Architecture Benefits

- Clear separation of responsibilities
- Frontend decoupled from backend
- Easy maintenance and service evolution
- Ability to scale components individually
- Simple and low-cost infrastructure on OCI
- Adherence to modern cloud and container practices

---

## Notes

- No internal service (MySQL, Data Science) is directly exposed
- All external communication goes through **Caddy**
- The architecture allows future migration to Kubernetes without major refactoring

---

## License

This project is for educational use. Developed for the NoCountry Hackathon in partnership with Alura/Oracle ONE.

---

<details>
<summary><strong>Versao em Portugues / Portuguese Version</strong></summary>

Este repositorio documenta a arquitetura e implementacao do projeto **Flight On Time** implantado na **Oracle Cloud Infrastructure (OCI)**, utilizando containers Docker para isolar e orquestrar os servicos.

**Arquitetura:** Todos os componentes rodam em containers Docker numa VM OCI, com Caddy como proxy reverso. Frontend (React), Backend (Java/Spring Boot), Data Science (Python/FastAPI) e MySQL, orquestrados via Docker Compose.

**Fluxo:** Usuario → Caddy → Frontend/Backend → MySQL + Data Science → Resposta.

Para detalhes completos da arquitetura e componentes, consulte a versao em ingles acima.

</details>
