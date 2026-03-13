# FlightOnTime — Production Commands
# ====================================

.PHONY: install train train-tuned serve mlflow-ui test clean

# ── Setup ──
install:
	pip install -r requirements.txt

# ── Training ──
train:
	python -m src.models.train

train-tuned:
	python -m src.models.train model=catboost_tuned

train-custom:
	python -m src.models.train $(ARGS)

# ── Serving ──
serve:
	MODEL_SOURCE=joblib uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

serve-mlflow:
	MODEL_SOURCE=mlflow uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# ── MLflow ──
mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Cleanup ──
clean:
	rm -rf outputs/ mlruns/ artifacts/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
