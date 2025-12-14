#  FlightOnTime - Motor de Inteligência Artificial

> **Status:** 🚀 Em Produção (v3.0.0-CAT) | **Recall de Segurança:** 89.4%

Este repositório contém o **Core de Data Science** do projeto FlightOnTime. Nossa missão é prever atrasos em voos comerciais no Brasil utilizando Machine Learning avançado, focando na segurança e planejamento do passageiro.

---

##  A Evolução do Modelo (Do MVP ao State-of-the-Art)

Nosso maior desafio foi lidar com o **desbalanceamento severo** dos dados (apenas 11% dos voos atrasam). Um modelo comum teria 89% de acurácia apenas dizendo "Nenhum voo vai atrasar", o que seria inútil.

Testamos diversos algoritmos de Boosting, priorizando a métrica de **Recall** (capacidade de detectar o perigo real).

| Versão | Modelo | Tecnologia | Recall (Detecção) | Decisão |
| :--- | :--- | :--- | :--- | :--- |
| v1.0 | **Random Forest** | Bagging Ensemble | 87.0% | Descontinuado |
| v2.0 | **XGBoost** | Gradient Boosting | 87.2% | Testado |
| **v3.0** | **CatBoost** | **Categorical Boosting** | **89.4% ** | **Em Produção** |

**Por que CatBoost?**
O algoritmo da Yandex demonstrou superioridade ao lidar com as variáveis categóricas complexas (rotas e companhias aéreas), permitindo atingir quase **90% de detecção de atrasos** sem sacrificar a performance da API.

---

##  Decisões Estratégicas de Negócio

### 1. Otimização do Limiar de Decisão (Threshold)
Realizamos uma análise matemática utilizando o **F2-Score** (que prioriza o Recall).
* **Sugestão do Algoritmo:** Corte em **0.43** (Recall 84%).
* **Decisão de Negócio (Override):** Fixamos o corte em **0.40**.
* **Motivo:** Decidimos sacrificar precisão estatística para ganhar **+5% de Segurança (Recall sobe para ~89.4%)**. Preferimos o risco de um "Falso Alerta" (Amarelo) do que deixar um passageiro perder o voo.

### 2. Engenharia de Feriados (Análise de Pareto)
Para otimizar o tempo de resposta da API, analisamos a origem de **2.5 milhões de voos**:
* **Partidas do Brasil:** 93.72%
* **Partidas do Exterior:** 6.28%

**Decisão:** Aplicamos apenas o calendário `holidays.Brazil()`. Como o atraso na decolagem é causado primariamente pelo aeroporto de origem, cobrimos **94% dos cenários de risco** de calendário com custo computacional mínimo.

---

##  Arquitetura e Engenharia de Features

O modelo não olha apenas para o passado. Enriquecemos os dados brutos com:

1.  **Detector de Feriados Dinâmico:** Cruzamento em tempo real da data do voo com o calendário oficial.
2.  **Georreferenciamento:** Cálculo da distância geodésica (`distancia_km`) entre coordenadas de aeroportos.
3.  **Decomposição Temporal:** Análise granular de Hora, Dia da Semana e Sazonalidade.

### Stack Tecnológico
* **Linguagem:** Python 3.10+
* **ML Core:** CatBoost, Scikit-Learn
* **Data Processing:** Pandas, Numpy, Holidays
* **API:** FastAPI (Uvicorn)
* **Deploy:** Docker / Oracle Cloud Infrastructure (OCI)

---

##  Regra de Negócio: O Semáforo de Risco

Traduzimos a probabilidade matemática (0.0 a 1.0) em uma experiência útil para o usuário:

* 🟢 **PONTUAL (Risco < 40%):**
    * Condições operacionais normais.
* 🟡 **ALERTA (Risco 40% - 60%):**
    * O modelo detectou instabilidade. Recomendamos monitorar o painel.
* 🔴 **ATRASADO (Risco > 60%):**
    * Alta probabilidade de problemas. O usuário deve se planejar para contingências.

---

##  Instalação e Execução

### 1. Preparar o Ambiente
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar o "Cérebro" (Opcional)
O repositório já inclui o modelo treinado. Mas se desejar retreinar com novos dados:

```bash
python src/train.py
```

*Isso gerará um novo arquivo `flight_classifier_mvp.joblib` com a lógica mais recente.*

### 3. Subir a API
Inicie o servidor de predição localmente:

```bash
python -m uvicorn src.app:app --reload
```

Acesse a documentação automática em: `http://127.0.0.1:8000/docs`

---

##  Documentação da API

A API foi desenhada para ser consumida por qualquer Front-End ou Back-End.

**Endpoint:** `POST /predict`

**Payload de Entrada (Exemplo):**

```json
{
  "companhia": "TAM",
  "origem": "Guarulhos - Governador Andre Franco Montoro",
  "destino": "Eduardo Gomes",
  "data_partida": "2025-12-25T20:00:00",
  "distancia_km": 2689.0
}
```

**Resposta da API (Exemplo Real):**

```json
{
  "previsao": "🔴 ATRASADO",
  "nivel_risco": "ALTO",
  "probabilidade": 0.8942,
  "is_feriado": true,
  "mensagem": "Alta probabilidade de atraso (89.4%). Planeje-se."
}
```

---

##  Deploy em Produção (Oracle Cloud)

Graças à infraestrutura configurada na OCI, a API já está disponível publicamente para integração via internet.

* **Base URL:** `http://flight-on-time.ds.vm3.arbly.com`
* **Endpoint de Predição:** `POST /predict`
* **Documentação Interativa (Swagger):** [Acessar Docs](http://flight-on-time.ds.vm3.arbly.com/docs)

**Teste rápido via Terminal (cURL):**

```bash
curl -X POST "http://flight-on-time.ds.vm3.arbly.com/predict" \
-H "Content-Type: application/json" \
-d '{"companhia": "AZUL", "origem": "Guarulhos", "destino": "Recife", "data_partida": "2025-12-25T14:30:00", "distancia_km": 2100.5}'
```

---

## Próximos Passos (Roadmap)

Embora o modelo atual seja robusto (89% Recall), identificamos oportunidades para a versão 2.0:

1. **Integração Meteorológica em Tempo Real:** Conectar com APIs de clima (OpenWeather) para considerar chuvas/tempestades no momento da predição.
2. **Monitoramento de Tráfego Aéreo:** Incluir variáveis sobre congestionamento de pistas em tempo real.

---

## Dataset (Origem dos Dados)

O modelo foi treinado com dados históricos reais de voos brasileiros.
Devido ao tamanho do arquivo, ele não está versionado neste repositório.

**Fonte Oficial (Kaggle):**
[Flights in Brazil (2015-2017) - Ramiro Bentes](https://www.kaggle.com/datasets/ramirobentes/flights-in-brazil)

**Como usar:**
1. Baixe o arquivo `BrFlights2.csv`.
2. Salve o arquivo na pasta: `data-science/data/BrFlights2.csv`.
