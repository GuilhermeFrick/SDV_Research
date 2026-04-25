# Etapa 1 — Parsing de PCAPs (Seção 5.1)

**Script:** `experiments/files/01_parse_pcap.py`
**Referência:** Kim et al. (2026), Seção 5.1 — *Layered Packet Extraction*
**Status:** Em execução

---

## Objetivo

Transformar os 7 arquivos PCAP brutos em um CSV estruturado com registros por pacote, separando as camadas IP, TCP/UDP e SOME/IP (incluindo SOME/IP-SD). Cada registro é rotulado com o tipo de ataque correspondente ao PCAP de origem.

---

## Dataset de Entrada

| Arquivo | Rótulo | Tamanho |
|---------|--------|---------|
| `benign_traffic.pcap` | normal | 213 MB |
| `dos_noti_flood.pcap` | dos | 186 MB |
| `fuzzy_sd_offer_rand_noti(1).pcap` | fuzzy | 219 MB |
| `fuzzy_sd_offer_rand_noti(2).pcap` | fuzzy | 127 MB |
| `fuzzy_sd_offer_rand_noti(3).pcap` | fuzzy | 216 MB |
| `mitm_multi_attacker.pcap` | mitm | 239 MB |
| `mitm_single_attacker.pcap` | mitm | 205 MB |

**Total:** ~1.4 GB | **Esperado:** ~14,23 milhões de pacotes SOME/IP

---

## O que o Script Faz

### Parsing em Camadas (Seção 5.1 do Artigo)

Para cada frame Ethernet capturado:

```
Ethernet
  └─ IP  (src, dst, proto, ttl, len, id, flags)
      └─ TCP ou UDP  (sport, dport, len/flags)
          └─ SOME/IP header (16 bytes fixos, big-endian)
              ├─ Service ID  [0-1]  → 0xFFFF = SOME/IP-SD
              ├─ Method ID   [2-3]
              ├─ Length      [4-7]
              ├─ Client ID   [8-9]
              ├─ Session ID  [10-11]
              ├─ Proto Ver.  [12]
              ├─ Iface Ver.  [13]
              ├─ Msg Type    [14]  → REQUEST/NOTIFICATION/RESPONSE/ERROR
              └─ Return Code [15]
```

### Identificação de Pacotes SOME/IP

O script detecta pacotes SOME/IP pelo intervalo de portas:
- `30490` — SOME/IP-SD (Service Discovery, multicast UDP)
- `30501–30503` — SOME/IP regular (serviços da simulação vSomeIP: GPS, IMU, VDE)

> **Nota:** O dataset usa portas 30501–30503 para os serviços, não 30491–30500 como seria mais comum. Isso é específico da configuração vSomeIP usada pelos autores.

### Distinção SOME/IP vs SOME/IP-SD

Um pacote é classificado como **SOME/IP-SD** quando `Service ID == 0xFFFF`. O campo `is_sd` no CSV marca essa distinção, que é essencial para a Etapa 2 (features calculadas separadamente por tipo).

---

## Saída

**Arquivo:** `data/parsed_packets.csv`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `timestamp` | float | Timestamp Unix do pacote |
| `src_ip` | str | IP de origem |
| `dst_ip` | str | IP de destino |
| `ip_proto` | int | Protocolo (6=TCP, 17=UDP) |
| `ip_ttl` | int | Time-to-Live |
| `ip_len` | int | Comprimento total IP |
| `transport` | str | "TCP" ou "UDP" |
| `src_port` | int | Porta de origem |
| `dst_port` | int | Porta de destino |
| `transport_len` | int | Comprimento da camada de transporte |
| `service_id` | int | Service ID SOME/IP |
| `method_id` | int | Method/Event ID |
| `someip_len` | int | Comprimento declarado SOME/IP |
| `client_id` | int | Client ID |
| `session_id` | int | Session ID |
| `msg_type` | int | Tipo de mensagem (0x00/0x02/0x80…) |
| `is_sd` | bool | True se SOME/IP-SD (service_id=0xFFFF) |
| `someip_payload_len` | int | Bytes de payload após o header de 16 bytes |
| `payload_hex` | str | Primeiros 32 bytes do payload (hex) |
| `label` | str | "normal", "dos", "fuzzy" ou "mitm" |
| `pcap_file` | str | Nome do PCAP de origem |

---

## Como Executar

```bash
# A partir da raiz do projeto
pip install scapy

python experiments/files/01_parse_pcap.py \
  --pcap-dir "data/dataset_ism_xgboost" \
  --output   "data/parsed_packets.csv"
```

Ou a partir da pasta do script:
```bash
cd experiments/files
python 01_parse_pcap.py \
  --pcap-dir "..\..\data\dataset_ism_xgboost" \
  --output   "..\..\data\parsed_packets.csv"
```

**Tempo estimado:** ~9 minutos (7 PCAPs × ~80s cada)

O script imprime progresso a cada 100.000 pacotes.

---

## Problemas Encontrados e Soluções

### 1. Encoding cp1252 no Windows
O script original usava o caractere `→` (U+2192) em mensagens de print, incompatível com o encoding padrão do terminal Windows (cp1252).

**Solução:** Substituído por `>>` (ASCII puro).

### 2. Portas SOME/IP incompletas
O script original checava apenas portas `{30490, 30491, 30500}`, perdendo todos os pacotes SOME/IP regulares (que usam 30501–30503 neste dataset).

**Diagnóstico:** Inspeção dos primeiros 5.000 pacotes do `benign_traffic.pcap` mostrou 0% de pacotes regulares com o filtro original; 95% com o filtro corrigido.

**Solução:** Alterado para intervalo `30490 <= porta <= 30510`.

### 3. `rdpcap` carrega PCAP inteiro na RAM
Para PCAPs de 200+ MB, `rdpcap()` carrega tudo na memória de uma vez.

**Solução:** Substituído por `PcapReader` (leitura streaming, pacote por pacote).

---

## Próxima Etapa

Com o CSV gerado, seguir para:
[Etapa 2 — Extração de Features](step02_feature_extraction.md)
