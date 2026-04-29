# SDV SOME/IP Network Simulation

Reprodução da rede veicular simulada descrita em **Kim et al. (2026)**,
Seção 4.1 — *"Simulation Data Collection"*.

> Kim, T. et al. *XGBoost-Based Anomaly Detection Framework for SOME/IP
> in In-Vehicle Networks*. Systems 2026, 14, 196.
> https://doi.org/10.3390/systems14020196

---

## Arquitetura (Figura 2 do artigo)

```
  PUBLISHERS                     SUBSCRIBERS
  ──────────                     ───────────

  ┌─────────┐  GPS (0x1234)  ──► NAV  (172.30.0.22)
  │   GPS   │  pos 100ms     ──► CLU  (172.30.0.21)
  │172.30.0.10│              ──► TEL  (172.30.0.25)
  └─────────┘

  ┌─────────┐  IMU (0x1235)  ──► STE  (172.30.0.24)
  │   IMU   │  accel 50ms    ──► TEL  (172.30.0.25)
  │172.30.0.11│
  └─────────┘

  ┌─────────┐  VDE (0x1236)  ──► ADAS (172.30.0.20)
  │   VDE   │  dynamics 50ms ──► PT   (172.30.0.23)
  │172.30.0.12│
  └─────────┘

  ┌─────────────────────────────────────────────────┐
  │  IDS Monitor (172.30.0.99)                      │
  │  tcpdump passivo → /captures/*.pcap (rot. 60s)  │
  └─────────────────────────────────────────────────┘

  Rede Docker: vehicle-bus  172.30.0.0/24
  Protocolo  : SOME/IP sobre UDP  |  Service Discovery: 224.224.224.245:30490
```

---

## Fluxo de Dados

```
  [GPS/IMU/VDE]          [SOME/IP-SD]           [Subscriber]
       │                      │                      │
       │  offer_service() ───►│                      │
       │                      │◄─── request_service()│
       │                      │──── offer (multicast)│
       │                      │                      │
       │◄─────────────────────┼──── subscribe()      │
       │                      │                      │
       │  notify() ───────────┼──────────────────────►│
       │  (100ms / 50ms)      │                      │  on_message()
       │                      │                      │  → imprime dado
       │                      │                      │
       └──────────────────────┴──[IDS Monitor]───────┘
                                   tcpdump captura
                                   todos os frames
                                   UDP 30490-30510
```

---

## Payloads

| Serviço | ID | Event | Frequência | Payload |
|---------|----|-------|-----------|---------|
| GPS | 0x1234 | 0x8001 | 100 ms | `float lat, lon` (8 bytes) |
| IMU | 0x1235 | 0x8002 | 50 ms | `float ax,ay,az,gx,gy,gz` (24 bytes) |
| VDE | 0x1236 | 0x8003 | 50 ms | `float speed,steer,throttle,brake` (16 bytes) |

---

## Estrutura

```
someip-sdv-network/
├── docker-compose.yml       # 10 containers, rede 172.30.0.0/24
├── Makefile                 # build / up / down / logs
├── config/                  # JSON vSomeIP por ECU
│   ├── vsomeip_gps.json
│   ├── vsomeip_imu.json
│   ├── vsomeip_vde.json
│   ├── vsomeip_adas.json
│   ├── vsomeip_clu.json
│   ├── vsomeip_nav.json
│   ├── vsomeip_pt.json
│   ├── vsomeip_ste.json
│   └── vsomeip_tel.json
├── ecu_gps/                 # Publisher: GPS (C++ + vSomeIP)
├── ecu_imu/                 # Publisher: IMU
├── ecu_vde/                 # Publisher: Vehicle Dynamics
├── ecu_subscriber/          # Subscriber genérico (ADAS/CLU/NAV/PT/STE/TEL)
│   └── src/subscriber.cpp   # comportamento via ECU_NAME env var
└── ids_monitor/             # Captura passiva → PCAPs
    ├── Dockerfile
    ├── monitor.sh
    └── captures/            # *.pcap gerados em runtime
```

---

## Como Rodar

```bash
# 1. Build (compila vSomeIP + aplicações — ~10 min na primeira vez)
make build

# 2. Sobe a rede
make up

# 3. Acompanha logs em tempo real
make logs

# 4. Ver apenas um ECU
make logs-gps

# 5. Listar capturas do IDS
make captures

# 6. Derrubar tudo
make down
```

---

## Integração com o IDS (próximo passo)

Os PCAPs gerados em `ids_monitor/captures/` são a entrada direta do pipeline:

```
captures/*.pcap
      │
      ▼
01_parse_pcap.py       → parsed_packets.csv
      │
      ▼
02_extract_features.py → train/test_features.csv
      │
      ▼
XGBoost IDS model      → predição normal / ataque
```

Para injetar ataques (Fuzzy, DoS, MitM) e gerar tráfego malicioso,
adicionar containers de ataque apontando para os mesmos serviços.
