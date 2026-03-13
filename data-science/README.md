# FlightOnTime — AI Engine (Research & Prototyping)

> **Status:** Production (v5.0.0-LiveWeather) | **Safety Recall:** 90.8%

This repository contains the **Data Science Core** of the FlightOnTime project. Our mission is to predict delays in Brazilian commercial flights using advanced Machine Learning enriched with real-time weather data, focusing on passenger safety and planning.

---

## Model Evolution (From MVP to Live-Weather)

Our biggest challenge was handling the **severe class imbalance** (only 11% of flights are delayed) and the complexity of external factors.

We evolved from a purely historical model to an autonomous architecture that queries weather APIs in real-time.

| Version | Model | Technology | Recall (Detection) | Status |
|:--------|:------|:-----------|:-------------------|:-------|
| v1.0 | Random Forest | Bagging Ensemble | 87.0% | Discontinued |
| v2.0 | XGBoost | Gradient Boosting | 87.2% | Tested |
| v3.0 | CatBoost | Pure Historical | 89.4% | Legacy (MVP) |
| v4.0 | CatBoost + OpenMeteo | Weather-Aware Pipeline | 86.0% | Tested |
| v4.1 | CatBoost Native | Weather-Aware + Native Features | 90.8% | Stable |
| v4.2 | CatBoost + GeoMaps | Smart Distance Calculation | 90.7% | Stable |
| **v5.0** | **CatBoost + Live API** | **Real-Time Weather Integration** | **90.7%** | **In Production** |
| **vEXP** | **Deep Learning** | **Entity Embeddings + DNN** | **77.5%** | **Experimental Research** |

*Note: With CatBoost Native implementation and Live integration, we surpassed previous models by combining historical precision with real-world data.*

---

## Strategic Business Decisions

### 1. Decision Threshold Optimization

We performed a mathematical analysis using the **F2-Score** (which prioritizes Recall).

- **Algorithm Suggestion:** Cutoff at **0.43**.
- **Business Decision (Override):** We fixed the cutoff at **0.35**.
- **Reason:** We chose to sacrifice statistical precision to ensure **Safety**. We prefer the risk of a "False Preventive Alert" over letting a passenger miss their flight due to an unannounced incoming storm.

### 2. Weather & Holiday Strategy (Pareto)

- **Holidays:** We apply the `holidays.Brazil()` calendar only to the departure date, covering 94% of demand peaks.
- **Weather:** The model queries the **OpenMeteo** API in real-time. Adverse conditions (rain > 10mm, wind > 30km/h) drastically increase the calculated risk.

---

## Architecture & Feature Engineering

The v5.0 model is an autonomous system that crosses historical data with live data:

1. **Weather Integration (NEW):** Ingestion of `precipitation` (mm) and `wind_speed` (km/h) data to understand the physical impact on the aircraft.
2. **Holiday Detector:** Real-time cross-referencing of the flight date with the official calendar.
3. **Georeferencing:** Geodesic distance calculation (`distance_km`) via the Haversine Formula.
4. **CatBoost Native Support:** Native category handling, increasing accuracy on complex routes.
5. **Smart Distance (v4.2):** The model "knows" airport coordinates and calculates distance automatically.
6. **Live Weather Integration (v5.0):** Real-time connection with the `OpenMeteo` API. If the user doesn't provide weather data, the system automatically fetches the weather forecast for the flight's time and location.

---

## Research Lab: Deep Learning & Entity Embeddings

As part of our pursuit of excellence and innovation, we conducted an advanced experiment exploring **Deep Neural Networks (Deep Learning)** as an alternative to Gradient Boosting. The goal was to understand whether an **Entity Embeddings**-based architecture could capture latent patterns between airports and routes that escape tree-based models.

### Experiment Journey (Rescue Pipeline)

Unlike CatBoost, which natively handles categories, developing the Neural Network required complex data engineering to avoid model collapse and hardware issues:

* **Abandoning One-Hot Encoding:** Initially, we tested One-Hot for airports and airlines. However, high cardinality generated sparse vectors that consumed all available RAM and diluted the model's predictive power.
* **Implementing Entity Embeddings:** We replaced One-Hot with Embedding layers. This allowed the model to learn dense numerical representations (embeddings) for each airport, logically "grouping" terminals with similar operational behaviors.
* **Severe Imbalance Treatment:**
    * **Class Weights:** Instead of resampling techniques, we applied differential weights in the loss function to force the model to give greater importance to delays (minority class).
    * **Stable Binary Crossentropy:** After testing *Focal Loss*, we stabilized training with *Binary Crossentropy* and a reduced *Learning Rate* (10⁻⁴) to prevent gradient explosion.
* **Threshold Optimization:** Instead of the default 0.5, we used the **F2-Score** to find the optimal decision point at **0.425**, prioritizing **Recall** (passenger safety) over precision.

### Comparative Results

| Metric | CatBoost (Production) | Deep Learning (Stable) |
|:-------|:---------------------|:----------------------|
| **ROC-AUC** | **0.794** | 0.697 |
| **Recall (Detection)** | **90.8%** | 77.5% |
| **Accuracy** | **73.1%** | 50.5% |
| **F1-Score** | **0.725** | 0.267 |

### Diagnosis & Engineering Decision

After rigorous analysis, we decided to **keep CatBoost in production**. The main reasons were:

1. **Efficiency on Tabular Data:** Gradient Boosting models proved superior for this structured dataset with 11 features. Neural Networks generally require higher feature dimensionality to outperform tree models.
2. **Precision/Recall Tradeoff:** The DL model, while achieving solid Recall (77%), presented a significantly higher false positive rate than CatBoost, which could compromise user experience with unnecessary alerts.
3. **Operational Complexity:** The computational cost and maintenance of a Deep Learning architecture did not justify the inferior performance compared to CatBoost's native solution.

> **Portfolio Note:** This experiment demonstrates our debugging capability and our discipline in following a scientific approach: testing complex hypotheses, but choosing the most effective tool for the real business problem. The notebook is preserved in the notebooks folder.

---

### Tech Stack

- **Language:** Python 3.10+
- **ML Core:** CatBoost (Gradient Boosting)
- **External Data:** Open-Meteo API (Weather Data)
- **API:** FastAPI + Uvicorn
- **Dependencies:** `requests` library for HTTP calls
- **Deployment:** Docker / Oracle Cloud Infrastructure (OCI)

---

## Business Rule: The Risk Semaphore

We translate the mathematical probability into a visual experience for the user:

| Risk Level | Probability | Description |
|:-----------|:------------|:------------|
| ON_TIME (green) | < 35% | Good flight conditions and stable weather |
| PREVENTIVE_ALERT (yellow) | 35% – 70% | Model detected instability (e.g., light rain or congested airport). Monitor the dashboard |
| LIKELY_DELAYED (red) | > 70% | Critical conditions detected (e.g., Storm + Holiday). High chance of problems |

---

## Installation & Running

### 1. Set Up the Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Train Model v5.0 (Optional)

The repository already includes the updated `flight_classifier_v4.joblib` file with the coordinates map. To retrain:
```bash
python data-science/src/train.py
```

### 3. Start the API

Start the prediction server locally (from the project root):
```bash
python -m uvicorn data-science.src.app:app --reload
```

Access the automatic documentation at: http://127.0.0.1:8000/docs

---

## API Documentation

The API accepts flight data and automatically fetches weather if needed.

**Endpoint:** `POST /predict`

**Input Payload (Minimalist - v5.0):** The system is now autonomous. Just provide the flight and date.
```json
{
  "companhia": "GOL",
  "origem": "Congonhas",
  "destino": "Santos Dumont",
  "data_partida": "2025-12-24T14:00:00"
}
```

*Note: `distancia_km`, `precipitation`, and `wind_speed` are optional. If omitted, the API calculates the geodesic distance and fetches weather in real-time via OpenMeteo.*

**API Response (Example with Automatic Weather):**
```json
{
  "previsao": "PREVENTIVE_ALERT",
  "probabilidade": 0.654,
  "cor": "yellow",
  "dados_utilizados": {
    "distancia": 366.0,
    "chuva": 5.2,
    "vento": 12.0,
    "fonte_clima": "LIVE (OpenMeteo)"
  }
}
```

---

## Strategic Roadmap (Phase 3)

With the delivery of v5.0 (Live Weather), the system is complete in terms of physical prediction. The next step is air traffic.

### 1. Air Network Monitoring (Domino Effect)

**The Challenge:** Aviation delays work in cascades. A delay in Brasilia affects Guarulhos hours later.

**The Solution:** Integrate with traffic APIs (FlightRadar24) to calculate the "average airport delay" in the last 60 minutes.

**Planned New Features:**

- `takeoff_queue_current`: Number of aircraft waiting for the runway.
- `airport_delay_index`: Current average delay at the hub.

---

## Dataset

**Official Source:** Flights in Brazil (2015-2017) — Kaggle
**Weather Data:** Enrichment via Open-Meteo Historical API.

**How to use:**

1. Run Notebook `1_data_engineering_weather.ipynb` in `data-science/notebooks/` to generate the dataset.
2. Run Notebook `2_modeling_strategy_v4.ipynb` for exploratory analysis.

---

<details>
<summary><strong>Versao em Portugues / Portuguese Version</strong></summary>

Este repositorio contem o **Core de Data Science** do projeto FlightOnTime. Nossa missao e prever atrasos em voos comerciais no Brasil utilizando Machine Learning avancado enriquecido com dados meteorologicos em tempo real.

**Evolucao:** Do Random Forest (v1.0) ao CatBoost com integracao Live Weather (v5.0), alcancando 90.8% de Recall. Tambem experimentamos Deep Learning com Entity Embeddings (vEXP), mas o CatBoost manteve superioridade de ~10% no ROC-AUC.

**Decisoes de Negocio:** Limiar de decisao fixado em 0.35 (override do F2-optimal 0.43) para priorizar seguranca. Semaforo de risco: PONTUAL (< 35%), ALERTA PREVENTIVO (35-70%), ATRASO PROVAVEL (> 70%).

**Stack:** Python 3.10+, CatBoost, Open-Meteo API, FastAPI, Uvicorn, Docker/OCI.

Para detalhes completos, consulte a versao em ingles acima.

</details>
