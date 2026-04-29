.PHONY: build up down logs clean

build:
	docker compose build

up:
	docker compose up -d
	@echo "Rede SDV rodando. Logs: make logs"

down:
	docker compose down

logs:
	docker compose logs -f

logs-gps:
	docker compose logs -f ecu-gps

logs-consumer:
	docker compose logs -f ecu-consumer

captures:
	@ls -lh ids_monitor/captures/ 2>/dev/null || echo "Nenhuma captura ainda."

clean:
	docker compose down -v --rmi local
	rm -f ids_monitor/captures/*.pcap
