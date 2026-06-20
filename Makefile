# GLASSBOX — single entry point for the dev workflow.
# Backend (FastAPI) lives in backend/ and is driven by uv; frontend (React/Vite) in frontend/.

.DEFAULT_GOAL := help
.PHONY: help setup api web dev test test-fast lint format fmt-check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install backend + frontend dependencies
	cd backend && uv sync --extra dev
	cd frontend && npm ci

api: ## Run the FastAPI backend with reload (http://localhost:8000)
	cd backend && uv run uvicorn app.main:app --reload

web: ## Run the Vite dev server (http://localhost:5173)
	cd frontend && npm run dev

dev: ## Run backend and frontend together
	$(MAKE) -j2 api web

test: ## Run the full backend test suite (downloads GPT-2 on first run)
	cd backend && uv run pytest

test-fast: ## Run only model-free tests
	cd backend && uv run pytest -m "not slow"

lint: ## Lint backend (ruff) and frontend (eslint)
	cd backend && uv run ruff check .
	cd frontend && npm run lint

format: ## Auto-format the backend with ruff
	cd backend && uv run ruff format . && uv run ruff check --fix .

fmt-check: ## Check backend formatting without writing (used in CI)
	cd backend && uv run ruff format --check .
