FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tcpdump tshark python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install scapy --quiet

RUN mkdir -p /captures

COPY monitor.sh /usr/local/bin/monitor.sh
RUN chmod +x /usr/local/bin/monitor.sh

CMD ["monitor.sh"]
