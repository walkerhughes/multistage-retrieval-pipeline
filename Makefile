
api: # restarts the api, preserves postgres
	docker compose down
	docker compose build api
	docker compose up

service: # restarts both api and postgres
	docker compose down
	docker volume rm retrieval-evals_postgres_data || true
	docker compose up -d