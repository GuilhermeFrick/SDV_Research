# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libboost-system-dev libboost-thread-dev libboost-filesystem-dev \
    libboost-log-dev libboost-program-options-dev \
    && rm -rf /var/lib/apt/lists/*

# Clona e compila vSomeIP
RUN git clone --depth 1 --branch 3.4.10 \
    https://github.com/COVESA/vsomeip.git /vsomeip

RUN cmake -S /vsomeip -B /vsomeip/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_SIGNAL_HANDLING=1 \
    && cmake --build /vsomeip/build --parallel $(nproc) \
    && cmake --install /vsomeip/build

# Compila a aplicação GPS
COPY CMakeLists.txt /app/
COPY src/ /app/src/
RUN cmake -S /app -B /app/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/build --parallel $(nproc)

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 libboost-thread1.74.0 \
    libboost-filesystem1.74.0 libboost-log1.74.0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/libvsomeip3* /usr/local/lib/
COPY --from=builder /app/build/gps_service /usr/local/bin/gps_service

RUN ldconfig

CMD ["gps_service"]
