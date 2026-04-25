# Etapa 2 — Extração de Features Comportamentais (Seção 5.2–5.3)

**Script:** `experiments/files/02_extract_features.py`
**Referência:** Kim et al. (2026), Seções 5.2–5.3 — *Feature Extraction & Feature Vector Generation*
**Entrada:** `data/parsed_packets.csv`
**Saída:** `data/train_features.csv`, `data/test_features.csv`

---

## Objetivo

Transformar cada pacote SOME/IP do CSV em um vetor de 9 features comportamentais (Tabela 1 do artigo), normalizá-las via Min-Max e dividir em conjuntos de treino e teste (split 50/50 estratificado).

---

## As 9 Features da Tabela 1

| # | Feature | Equação | Descrição |
|---|---------|---------|-----------|
| f01 | IP time interval | Δt | Intervalo entre pacotes consecutivos no mesmo fluxo |
| f02 | SOME/IP likelihood | Eq. 5 | log L(x) = Σ log Pᵢ(xᵢ) do payload SOME/IP |
| f03 | TCP/UDP likelihood | Eq. 5 | Mesmo cálculo, modelo treinado em contexto TCP/UDP |
| f04 | SOME/IP entropy | Eq. 6 | H(x;P) = -(1/L) Σ log Pᵢ(xᵢ) — SOME/IP |
| f05 | TCP/UDP entropy | Eq. 6 | Cross-entropy — TCP/UDP |
| f06 | SOME/IP payload changes | Eq. 7 | Distância de Hamming entre payloads consecutivos |
| f07 | TCP/UDP payload changes | Eq. 7 | Hamming distance — TCP/UDP |
| f08 | IP length changes | Δ | Variação de comprimento IP entre pacotes consecutivos |
| f09 | TCP/UDP length changes | Δ | Variação de comprimento TCP/UDP entre pacotes consecutivos |

> **Nota sobre SOME/IP-SD:** O artigo mantém as features de SOME/IP regular e SOME/IP-SD **separadas** (por isso 12 colunas nos .npy dos autores, não 9). A distinção é feita pela flag `is_sd` do CSV. O script atual usa um único modelo compartilhado — pode ser separado facilmente para reprodução exata.

---

## Modelo de Distribuição de Bytes

### Treinamento (apenas tráfego benigno)

A distribuição de probabilidade de bytes é aprendida **somente** no tráfego benigno para servir como referência do comportamento "normal". Pacotes anômalos terão baixo log-likelihood e alta entropia em relação a este modelo.

```
Para cada posição i e valor de byte b:
  cᵢ(b)  = contagem de vezes que byte b aparece na posição i
  Pᵢ(b)  = (cᵢ(b) + α) / (Nᵢ + 256·α)   ← Laplace smoothing (Eq. 3)
```

**Parâmetros:**
- `alpha = 1.0` (Laplace smoothing)
- `max_positions = 256` (primeiros 256 bytes do payload)
- Treinado em amostra de 50.000 pacotes benignos (por velocidade)

### Log-Likelihood (Equação 5)

```
log L(x) = Σᵢ₌₁ᴸ log Pᵢ(xᵢ)
```

Valor negativo (mais negativo = mais improvável = mais anômalo).

### Cross-Entropy (Equação 6)

```
H(x;P) = -(1/L) Σᵢ₌₁ᴸ log Pᵢ(xᵢ)
```

Normalizado pelo comprimento L → independente do tamanho do payload.

> **Relação matemática provada:** `normalized_logL + normalized_H = 1` exato para cada amostra. Isso explica por que os pares de colunas (f02,f04), (f03,f05) somam 1.0 nos .npy dos autores — são derivados dos mesmos termos `log Pᵢ(xᵢ)`.

### Payload Changes (Equação 7)

Distância de Hamming bit-a-bit entre payloads consecutivos no mesmo fluxo:

```
dH(b₁, b₂) = Σᵢ wH(b₁ᵢ XOR b₂ᵢ)
```

---

## Agrupamento por Fluxo

Features dependentes de tempo (time interval, payload changes, length changes) são calculadas **por fluxo**, onde fluxo = five-tuple:

```
(src_ip, dst_ip, src_port, dst_port, transport)
```

O primeiro pacote de cada fluxo recebe valor 0 para features de delta (sem pacote anterior).

---

## Normalização Min-Max (Equação 8)

```
x' = (x - x_min) / (x_max - x_min)
```

- Parâmetros calculados **exclusivamente no conjunto de treino**
- Aplicados no teste (sem data leakage)
- Valores fora do range de treino são clipados para [0, 1]

---

## Split Treino/Teste

- **Método:** Estratificado 50/50 (preserva proporção de classes)
- **Referência:** Artigo Seção 5.3: "We randomly split our dataset in half for training and testing"
- `random_state = 42` para reprodutibilidade

---

## Como Executar

```bash
# A partir da raiz do projeto
python experiments/files/02_extract_features.py \
  --parsed-csv data/parsed_packets.csv \
  --output-dir data/
```

**Tempo estimado:** ~20–40 minutos para ~14M registros (loop Python por pacote)

Para acelerar, pode-se usar o subset amostral:
```bash
# Apenas 10% dos dados para teste rápido
python -c "
import pandas as pd
df = pd.read_csv('data/parsed_packets.csv')
df.sample(frac=0.1, random_state=42).to_csv('data/parsed_packets_sample.csv', index=False)
"
python experiments/files/02_extract_features.py \
  --parsed-csv data/parsed_packets_sample.csv \
  --output-dir data/
```

---

## Arquivos de Saída

| Arquivo | Descrição |
|---------|-----------|
| `data/train_features.csv` | Treino (50%) com 9 features normalizadas + label |
| `data/test_features.csv` | Teste (50%) com 9 features normalizadas + label |
| `data/all_features_raw.csv` | Todas as amostras com features brutas (pré-normalização) |

---

## Mapeamento para os 12 .npy dos Autores

Os autores disponibilizam `X_train.npy` e `X_test.npy` com shape `(7.116.674, 12)`. As 12 colunas mapeiam para:

| Col .npy | Feature gerada por este script |
|----------|-------------------------------|
| 0 | f02 (SOME/IP likelihood — regular) |
| 1 | f04 (SOME/IP entropy — regular) |
| 2 | f03 (SOME/IP-SD likelihood) |
| 3 | f03 (TCP/UDP likelihood) |
| 4 | f05 (SOME/IP-SD entropy) |
| 5 | f05 (TCP/UDP entropy) |
| 6 | f06 (SOME/IP payload changes) |
| 7 | f01 ou f06 para SD (ambíguo — ver README.md) |
| 8 | f07 (TCP/UDP payload changes) |
| 9 | f01 (IP time interval) |
| 10 | f08 (IP length changes) |
| 11 | f09 (TCP/UDP length changes) |

---

## Próxima Etapa

Com os CSVs de features gerados, seguir para:
[Etapa 3 — Treinamento e Avaliação](step03_training_evaluation.md)
