# Flight On Time API

The Flight On Time API is a Back-End (REST) application built with Java and the Spring Boot framework. Its main goal is to provide predictions about flight status (delayed or on-time) using the integrated Data Science model via microservice.

## Prediction Flow (Data → Model → Prediction)

The application follows three main stages:

1. **Data Input**: The Java API receives flight details via JSON (airline, airports, and departure date).
2. **DS Integration**: The service (`FlightPredictionService`) communicates via `RestClient` with the Data Science microservice.
3. **Response**: The API standardizes the response with the prediction, decimal probability, risk semaphore color, and the details used.

## Tools & Dependencies

- **Language:** Java 21
- **Framework:** Spring Boot 3.5.4
- **Database:** MySQL with Flyway migrations
- **Documentation:** SpringDoc OpenAPI (Swagger)
- **Resilience:** Resilience4j (Circuit Breaker)

## Running Locally

**Prerequisites**

- Java 21 and Maven (or use the included `./mvnw`)
- MySQL running locally
- The Data Science microservice running

**Steps**

1. **Configure the Database**: Run the migrations in `src/main/resources/db/migration` to create the users, profiles, airports, and airlines tables.

2. **Set environment variables**: Define the database credentials and service URLs:

| Variable | Description |
|----------|-------------|
| `FLIGHTONTIME_DATASOURCE_DEV` | MySQL URL (e.g., `jdbc:mysql://localhost:3306/flightontime`) |
| `FLIGHTONTIME_USERNAME_DEV` | Database username |
| `FLIGHTONTIME_PASSWORD_DEV` | Database password |
| `FLIGHTONTIME_DATASCIENCE_BASEURL` | AI engine URL (e.g., `http://localhost:8000`) |
| `FLIGHTONTIME_JWT_SECRET_DEV` | Secret for JWT token generation |
| `FLIGHTONTIME_PATH_DEV` | Application context path (optional) |

3. **Run the API**:

```bash
./mvnw spring-boot:run
```

4. **Access:** Interactive documentation is available at `/swagger-ui.html`.

## Usage Examples (Endpoint `/predict`)

The service exposes a `POST` endpoint that validates the presence of all required fields before processing the query.

### 1. On-Time Flight Example (Low Risk)

**Request:**

```json
{
  "companhia": "GOL",
  "origem": "GIG",
  "destino": "GRU",
  "data_partida": "2025-11-10T14:30:00Z"
}
```

**Response (Probability < 0.35):**

```json
{
  "previsao": "ON_TIME",
  "probabilidade": 0.15,
  "cor": "green",
  "detalhes": {
    "distancia": 350.0,
    "chuva": 0.0,
    "vento": 5.2,
    "fonte_clima": "LIVE (OpenMeteo)"
  }
}
```

### 2. Delayed Flight Example (High Risk)

**Request (Christmas holiday with bad weather):**

```json
{
  "companhia": "GOL",
  "origem": "GRU",
  "destino": "REC",
  "data_partida": "2025-12-25T14:30:00Z"
}
```

**Response (Probability > 0.70):**

```json
{
  "previsao": "LIKELY_DELAYED",
  "probabilidade": 0.72,
  "cor": "red",
  "detalhes": {
    "distancia": 2689.0,
    "chuva": 12.5,
    "vento": 18.3,
    "fonte_clima": "LIVE (OpenMeteo)"
  }
}
```

### 3. Validation Error Example

If a required field such as `data_partida` is omitted, the API returns a standardized error:

**Response (400 Bad Request):**

```json
[
  {
    "campo": "data_partida",
    "mensagem": "data_partida must not be null"
  }
]
```

---

<details>
<summary><strong>Versao em Portugues / Portuguese Version</strong></summary>

A **Flight On Time API** e uma aplicacao Back-End (REST) desenvolvida em Java com Spring Boot. Fornece previsoes sobre o status de voos (atrasado ou pontual) utilizando o modelo de Data Science integrado via microservico.

**Fluxo:** A API recebe dados do voo via JSON, comunica-se com o microservico de Data Science via RestClient, e retorna a previsao com probabilidade, cor do semaforo de risco e detalhes utilizados.

**Stack:** Java 21, Spring Boot 3.5.4, MySQL (Flyway), SpringDoc OpenAPI, Resilience4j (Circuit Breaker).

Para detalhes completos de configuracao e exemplos, consulte a versao em ingles acima.

</details>
