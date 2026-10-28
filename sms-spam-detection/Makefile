.PHONY: help install train api ui test lint smoke load docker-build docker-up docker-down clean

PYTHON ?= python
PORT ?= 8000

help:
	@echo "install      install runtime + dev dependencies"
	@echo "train        download data and train the model"
	@echo "api          run the FastAPI service (port $(PORT))"
	@echo "ui           run the Streamlit dashboard (port 8501)"
	@echo "test         run the test suite"
	@echo "lint         run ruff"
	@echo "smoke        black-box smoke test against a running API"
	@echo "load         concurrent load test"
	@echo "docker-build build the production image"
	@echo "docker-up    start the service with docker compose"
	@echo "clean        remove caches and generated artifacts"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

train:
	$(PYTHON) main.py train

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --reload

ui:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

smoke:
	$(PYTHON) scripts/smoke_test.py --url http://localhost:$(PORT)

load:
	$(PYTHON) scripts/load_test.py --url http://localhost:$(PORT) --requests 500 --concurrency 25

docker-build:
	docker build -t sms-spam-detector:latest .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ models feedback
