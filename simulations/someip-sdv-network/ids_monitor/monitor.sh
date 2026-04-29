#!/bin/bash
# IDS Monitor — captura tráfego SOME/IP passivamente
# Salva PCAPs rotativos de 60s em /captures/
# Futura integração: pipe para o modelo XGBoost

IFACE=${CAPTURE_IFACE:-eth0}
OUTDIR=/captures
ROTATE_SECONDS=60

echo "[IDS] Iniciando captura em $IFACE (porta 30490-30510)"
echo "[IDS] Salvando em $OUTDIR — rotação a cada ${ROTATE_SECONDS}s"

mkdir -p "$OUTDIR"

exec tcpdump -i "$IFACE" \
    -w "$OUTDIR/capture_%Y%m%d_%H%M%S.pcap" \
    -G "$ROTATE_SECONDS" \
    -Z root \
    "udp portrange 30490-30510 or tcp portrange 30490-30510"
