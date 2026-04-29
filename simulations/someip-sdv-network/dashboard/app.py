"""
SDV SOME/IP — Telemetry Dashboard
Lê logs dos containers Docker em tempo real e emite métricas via Socket.IO.
Acesse: http://localhost:5000
"""

import re
import time
import threading
from collections import defaultdict, deque
from datetime import datetime

import docker
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Configuração dos ECUs monitorados ─────────────────────────────────────────
ECUS = {
    "ecu-gps": {"label": "GPS",  "color": "#2196F3", "role": "publisher"},
    "ecu-imu": {"label": "IMU",  "color": "#4CAF50", "role": "publisher"},
    "ecu-vde": {"label": "VDE",  "color": "#FF9800", "role": "publisher"},
    "ecu-adas":{"label": "ADAS", "color": "#9C27B0", "role": "subscriber"},
    "ecu-clu": {"label": "CLU",  "color": "#E91E63", "role": "subscriber"},
    "ecu-nav": {"label": "NAV",  "color": "#00BCD4", "role": "subscriber"},
    "ecu-pt":  {"label": "PT",   "color": "#FF5722", "role": "subscriber"},
    "ecu-ste": {"label": "STE",  "color": "#607D8B", "role": "subscriber"},
    "ecu-tel": {"label": "TEL",  "color": "#795548", "role": "subscriber"},
}

# Padrões para detectar pacote enviado/recebido nos logs
SEND_PATTERNS = {
    "ecu-gps":  re.compile(r"\[GPS\].*lat="),
    "ecu-imu":  re.compile(r"\[IMU\].*accel="),
    "ecu-vde":  re.compile(r"\[VDE\].*speed="),
}
RECV_PATTERN = re.compile(r"lat=|accel=|speed=")

# ── Estado global ─────────────────────────────────────────────────────────────
stats = {
    name: {
        "total":    0,
        "per_sec":  0,
        "last_payload": "",
        "history":  deque(maxlen=60),   # últimos 60s
        "window":   deque(maxlen=10),   # janela de 1s para calcular /seg
    }
    for name in ECUS
}

lock = threading.Lock()


# ── Leitor de logs por container ──────────────────────────────────────────────
def watch_container(client, container_name):
    pattern = SEND_PATTERNS.get(container_name, RECV_PATTERN)
    while True:
        try:
            container = client.containers.get(container_name)
            for line in container.logs(stream=True, follow=True, tail=0):
                text = line.decode("utf-8", errors="ignore").strip()
                if pattern.search(text):
                    ts = time.time()
                    with lock:
                        s = stats[container_name]
                        s["total"] += 1
                        s["window"].append(ts)
                        s["last_payload"] = text[text.find("]")+2:][:60]
        except docker.errors.NotFound:
            time.sleep(3)
        except Exception as e:
            time.sleep(2)


# ── Cálculo de pacotes/seg e emissão via Socket.IO ───────────────────────────
def emit_loop():
    while True:
        time.sleep(1)
        now = time.time()
        payload = {}
        with lock:
            for name, s in stats.items():
                # conta eventos na janela do último segundo
                per_sec = sum(1 for t in s["window"] if now - t <= 1.0)
                s["per_sec"] = per_sec
                s["history"].append(per_sec)
                payload[name] = {
                    "label":       ECUS[name]["label"],
                    "color":       ECUS[name]["color"],
                    "role":        ECUS[name]["role"],
                    "total":       s["total"],
                    "per_sec":     per_sec,
                    "last_payload":s["last_payload"],
                    "history":     list(s["history"]),
                }
        socketio.emit("metrics", payload)


# ── Inicia threads ao arrancar ────────────────────────────────────────────────
def start_watchers():
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
    except Exception:
        client = docker.from_env()

    for name in ECUS:
        t = threading.Thread(target=watch_container, args=(client, name), daemon=True)
        t.start()

    threading.Thread(target=emit_loop, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html", ecus=ECUS)


if __name__ == "__main__":
    start_watchers()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
