.PHONY: db api worker web test lint typecheck

db:
	docker compose up -d db

api:
	cd services/backend && conda run -n PaperScope python -m uvicorn paperscope.main:app --reload --host 127.0.0.1 --port 8000

worker:
	cd services/backend && conda run -n PaperScope python -m paperscope.worker

web:
	pnpm --filter @paperscope/web dev

test:
	cd services/backend && conda run -n PaperScope python -m pytest tests

lint:
	cd services/backend && conda run -n PaperScope python -m ruff check paperscope tests
	pnpm lint

typecheck:
	pnpm typecheck
