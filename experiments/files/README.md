# Reprodução: Kim et al. (2026) — XGBoost-Based Anomaly Detection for SOME/IP

Reprodução completa do pipeline de detecção de intrusão SOME/IP descrito em:

> Kim, T. et al. *"XGBoost-Based Anomaly Detection Framework for SOME/IP
> in In-Vehicle Networks"*, Systems 2026, 14, 196.
> https://doi.org/10.3390/systems14020196

---

## Estrutura dos Scripts

```
someip_ids/
├── 01_parse_pcap.py        # Etapa 1: Parsing de PCAPs em camadas (Seção 5.1)
├── 02_extract_features.py  # Etapa 2: 9 features comportamentais (Seção 5.2-5.3)
├── 03_train_evaluate.py    # Etapa 3: XGBoost + CTGAN + Avaliação (Seção 5.4, 6.x)
├── requirements.txt        # Dependências Python
└── README.md               # Este arquivo
```

---

## Instalação

```bash
# Cria e ativa ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Instala dependências
pip install -r requirements.txt
```

---

## Download do Dataset

1. Acesse: https://figshare.com/articles/dataset/30970450
2. Baixe os 8 arquivos PCAP:
   - `benign_traffic.pcap`
   - `dos_noti_flood.pcap`
   - `fuzzy_sd_offer_rand_noti(1).pcap`
   - `fuzzy_sd_offer_rand_noti(2).pcap`
   - `fuzzy_sd_offer_rand_noti(3).pcap`
   - `mitm_multi_attacker.pcap`
   - `mitm_single_attacker.pcap`
3. Coloque todos na pasta `data/pcap/`

> **Nota:** O dataset também inclui `tr_te_sets.tar` com os conjuntos de treino/teste
> pré-processados pelos autores. Você pode usá-los diretamente na Etapa 3 para
> comparar com os resultados do artigo de forma mais precisa.

---

## Execução Passo a Passo

### Etapa 1 — Parsing dos PCAPs

```bash
python 01_parse_pcap.py --pcap-dir data/pcap --output data/parsed_packets.csv
```

**Saída:** `data/parsed_packets.csv` (~14M linhas com campos IP/TCP/UDP/SOME/IP)

**Tempo estimado:** 10-30 min dependendo do hardware

---

### Etapa 2 — Extração de Features

```bash
python 02_extract_features.py \
    --parsed-csv data/parsed_packets.csv \
    --output-dir data/
```

**Saída:**
- `data/train_features.csv` — 50% treino, normalizado (Min-Max)
- `data/test_features.csv`  — 50% teste, normalizado
- `data/all_features_raw.csv` — todas as features antes da normalização

**Tempo estimado:** 5-20 min

As 9 features calculadas (Tabela 1 do artigo):

| Coluna | Feature | Equação |
|--------|---------|---------|
| `f01_ip_time_interval` | Δt entre pacotes IP consecutivos | Eq. 1 |
| `f02_someip_likelihood` | Log-likelihood payload SOME/IP | Eq. 5 |
| `f03_tcpudp_likelihood` | Log-likelihood payload TCP/UDP | Eq. 5 |
| `f04_someip_entropy` | Cross-entropy SOME/IP | Eq. 6 |
| `f05_tcpudp_entropy` | Cross-entropy TCP/UDP | Eq. 6 |
| `f06_someip_payload_changes` | Hamming distance SOME/IP consecutivos | Eq. 7 |
| `f07_tcpudp_payload_changes` | Hamming distance TCP/UDP consecutivos | Eq. 7 |
| `f08_ip_length_changes` | Δ comprimento pacotes IP | Eq. 1 |
| `f09_tcpudp_length_changes` | Δ comprimento TCP/UDP | Eq. 1 |

---

### Etapa 3 — Treinamento e Avaliação

```bash
# Completo (com CTGAN e comparação de baselines) — mais lento
python 03_train_evaluate.py \
    --train-csv data/train_features.csv \
    --test-csv  data/test_features.csv \
    --output-dir results/

# Rápido (sem CTGAN, sem baselines) — para teste inicial
python 03_train_evaluate.py \
    --train-csv data/train_features.csv \
    --test-csv  data/test_features.csv \
    --output-dir results/ \
    --no-ctgan \
    --no-baselines
```

**Saída:**
- `results/xgboost_someip_ids.json` — modelo treinado
- `results/figure10_performance_curves.png` — ROC, PR, F1×threshold, DET
- `results/figures11_12_baseline_comparison.png` — comparação de modelos
- `results/baseline_comparison.csv` — tabela de métricas
- `results/metrics_summary.json` — todas as métricas em JSON

---

## Resultados Esperados (Kim et al. 2026)

| Métrica | Imbalanceado | Balanceado |
|---------|-------------|------------|
| Precision | 0.97 | 0.97 |
| Recall | 0.97 | 0.97 |
| F1-Score | 0.97 | 0.90 |
| PR-AUC | 0.93 | 0.97 |
| ROC-AUC | 0.99 | 0.97 |
| Threshold ótimo | 0.36 | — |
| Latência total | 0.556 ms/pacote | — |

---

## Usando os Dados Pré-processados dos Autores

O arquivo `tr_te_sets.tar` do Figshare contém os conjuntos de treino/teste
já extraídos pelos autores originais. Para usá-los diretamente:

```bash
# Extrai o arquivo
tar -xf tr_te_sets.tar -C data/

# Adapta as colunas ao formato esperado e pula as etapas 1 e 2
# (os CSVs dos autores já têm as features numéricas)
python 03_train_evaluate.py \
    --train-csv data/tr_te_sets/train.csv \
    --test-csv  data/tr_te_sets/test.csv \
    --output-dir results/
```

> **Nota:** Pode ser necessário renomear as colunas para coincidir com
> os nomes `f01_..._norm` usados neste pipeline.

---

## Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| scapy | ≥2.5 | Parsing de PCAPs |
| pandas | ≥1.5 | Manipulação de dados |
| numpy | ≥1.23 | Cálculo numérico |
| xgboost | ≥1.7 | Modelo de classificação |
| scikit-learn | ≥1.2 | Métricas e baselines |
| ctgan | ≥0.7 | Aumentação de dados |
| matplotlib | ≥3.6 | Visualizações |
| lightgbm | ≥3.3 | Baseline LGB (opcional) |

---

## Notas de Reprodução

1. **Modelo de bytes separado por fluxo:** O artigo treina modelos de distribuição
   de bytes distintos para SOME/IP e SOME/IP-SD. Este script usa um modelo único
   mas distingue via flag `is_sd` — pode ser separado facilmente.

2. **CTGAN:** A geração demora proporcionalmente ao tamanho do dataset de ataque.
   Use `--no-ctgan` para uma primeira execução rápida.

3. **Limiar de decisão:** O artigo encontra threshold=0.36 no conjunto de treino.
   Este script otimiza o threshold automaticamente e reporta o valor encontrado.

4. **Dados pré-processados:** Para comparação mais fiel, use o `tr_te_sets.tar`
   do Figshare, que contém exatamente os dados usados pelos autores.
