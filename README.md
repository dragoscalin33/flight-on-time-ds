#  FlightOnTime: Previsor de Atrasos Aéreos

##  Descrição
Este projeto é um MVP desenvolvido durante o Hackathon **No Country**. Seu objetivo é prever a probabilidade de atraso de um voo comercial no Brasil utilizando Machine Learning.

##  Arquitetura
* **Data Science:** Python, Pandas, Scikit-Learn (Random Forest).
* **API:** FastAPI (para servir o modelo).
* **Back-End:** Java Spring Boot (Consumidor da API).

##  Estrutura do Repositório
* `/data`: `BrFlights2.csv`.
* `/notebooks`: Análise exploratória e treinamento (Jupyter).
* `/src`: Código-fonte da API e scripts de produção.
* `/models`: Arquivos serializados (.joblib).

##  Como executar (Data Science)
1. Instalar dependências:
   ```bash
   pip install -r requirements.txt
