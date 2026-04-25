# Reprodução de Kim et al. (2026) — Documentação do Pipeline

**Artigo:** *XGBoost-Based Anomaly Detection Framework for SOME/IP in In-Vehicle Networks*
**Publicação:** Systems 2026, 14, 196 — DOI: 10.3390/systems14020196
**Dataset:** Figshare — https://doi.org/10.6084/m9.figshare.30970450

---

## Objetivo

Reproduzir integralmente o framework proposto por Kim et al. (2026): parsing de tráfego SOME/IP capturado em simulação vSomeIP, extração de features comportamentais em camadas, treinamento de um classificador XGBoost com augmentação CTGAN e avaliação nos dois cenários do artigo (imbalanceado e balanceado).

---

## Estrutura do Pipeline

```
PCAPs (7 arquivos, 1.4 GB)
    │
    ▼  Etapa 1 — Parsing
data/parsed_packets.csv  (~14M linhas)
    │
    ▼  Etapa 2 — Extração de Features
data/features/train_features.csv
data/features/test_features.csv     (12 features normalizadas, split 50/50)
    │
    ▼  Etapa 3 — Treinamento e Avaliação
models/xgboost_someip_ids.json
results/metrics_summary.json
results/figures/
```

---

## Etapas

| # | Script | Seção do Artigo | Documento |
|---|--------|-----------------|-----------|
| 1 | `01_parse_pcap.py` | 5.1 — Layered Packet Extraction | [step01_pcap_parsing.md](step01_pcap_parsing.md) |
| 2 | `02_extract_features.py` | 5.2–5.3 — Feature Extraction & Vector Generation | [step02_feature_extraction.md](step02_feature_extraction.md) |
| 3 | `03_train_evaluate.py` | 5.4 & 6 — XGBoost + Avaliação | [step03_training_evaluation.md](step03_training_evaluation.md) |

---

## Achados e Decisões Importantes

### Mapeamento das 12 colunas do dataset dos autores

Os arquivos `.npy` disponibilizados pelos autores têm shape `(7.116.674, 12)`. A Tabela 1 do artigo lista 9 *tipos* de features, mas o campo `SOME/IP(or -SD)` indica que 3 delas são calculadas separadamente para SOME/IP regular e SOME/IP-SD, gerando 12 colunas no total:

| Col | Feature | Zeros |
|-----|---------|-------|
| 0 | SOME/IP likelihood (regular) | 0% |
| 1 | SOME/IP entropy (regular) | 48% |
| 2 | SOME/IP-SD likelihood | 0% |
| 3 | TCP/UDP likelihood | 0% |
| 4 | SOME/IP-SD entropy | 61% |
| 5 | TCP/UDP entropy | 95% |
| 6 | SOME/IP payload changes | 91% |
| 7 | SOME/IP-SD payload changes (ou IP time interval) | 0% |
| 8 | TCP/UDP payload changes | 96% |
| 9 | IP time interval (ou SOME/IP-SD payload changes) | 0% |
| 10 | IP length changes | 9% |
| 11 | TCP/UDP length changes | 34% |

**Prova matemática:** `col0 + col1 = 1,0` exato para cada amostra (idem cols 2+4 e 3+5). Isso resulta da relação `normalized_logL + normalized_H = 1` quando ambos são derivados dos mesmos termos `log P_i(x_i)`.

### Correção de portas SOME/IP

O dataset vSomeIP usa:
- Porta `30490` — SOME/IP-SD (multicast UDP)
- Portas `30501–30503` — SOME/IP regular (serviços GPS, IMU, VDE)

O script original checava apenas `{30490, 30491, 30500}`, perdendo todos os pacotes regulares. Corrigido para o intervalo `30490–30510`.

---

## Ambiente

```
Python 3.x (conda base)
scapy >= 2.5
pandas, numpy, xgboost, scikit-learn, lightgbm, ctgan, matplotlib
```

Instalação:
```bash
pip install -r experiments/files/requirements.txt
```

---

## Resultados Esperados (do Artigo)

| Cenário | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---------|-----------|--------|----|--------|---------|
| Imbalanceado | 0.97 | 0.97 | 0.97 | 0.93 | 0.99 |
| Balanceado | 0.97 | 0.97 | 0.90 | 0.97 | 0.97 |

Threshold ótimo (maximiza F1 no treino): **0.36**
Latência end-to-end (feature + inferência): **0.556 ms/pacote**
