# Etapa 3 — Treinamento XGBoost + CTGAN + Avaliação (Seções 5.4 e 6)

**Script:** `experiments/files/03_train_evaluate.py`
**Referência:** Kim et al. (2026), Seções 5.4 & 6 — *XGBoost Training + Evaluation*
**Entrada:** `data/train_features.csv`, `data/test_features.csv`
**Saída:** `results/` (modelo, métricas JSON, gráficos)

---

## Objetivo

Treinar o classificador XGBoost com aumentação CTGAN, otimizar o limiar de decisão por F1, e avaliar em dois cenários: dataset imbalanceado (realista) e dataset balanceado (downsampling). Reproduzir os gráficos da Figura 10 e a comparação com 10 baselines (Seção 6.3).

---

## Aumentação com CTGAN (Seção 5.4.1)

O dataset real é desbalanceado — tráfego benigno é muito mais frequente que ataques. A aumentação CTGAN gera amostras sintéticas da classe minoritária (ataques) para balancear o treino.

**Configuração do artigo:**

| Parâmetro | Valor |
|-----------|-------|
| embedding_dim | 128 |
| generator_dim | (256, 256) |
| discriminator_dim | (256, 256) |
| batch_size | 500 |
| epochs | 100 |

O CTGAN é treinado **apenas nos dados de ataque** do conjunto de treino e gera amostras até atingir proporção 1:1 com o tráfego benigno.

---

## Hiperparâmetros XGBoost (Tabela 2)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| objective | binary:logistic | Classificação binária |
| n_estimators | 1000 | Número de árvores |
| learning_rate | 0.05 | Taxa de aprendizado |
| max_depth | 6 | Profundidade máxima |
| subsample | 0.8 | Fração de amostras por árvore |
| colsample_bytree | 0.8 | Fração de features por árvore |
| min_child_weight | 1 | Peso mínimo de instâncias na folha |
| reg_lambda | 1.0 | Regularização L2 |
| min_split_loss (gamma) | 0.0 | Redução mínima de perda para split |

---

## Otimização do Limiar de Decisão (Seção 5.4.2)

Em vez de usar o limiar padrão de 0.5, o artigo otimiza o limiar que maximiza o F1-score **no conjunto de treino**:

```python
for t in np.arange(0.01, 0.99, 0.01):
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1, best_t = f1, t
```

**Resultado reportado pelo artigo:** threshold ótimo = **0.36**, F1 = 0.97

---

## Cenários de Avaliação (Seção 6.2)

### Cenário 1: Imbalanceado (Realista)

- Conjunto de teste original com distribuição real das classes
- Representa condições de produção em veículo

### Cenário 2: Balanceado (Downsampling)

- Downsampling aleatório da classe majoritária (normal) para igualar à classe de ataque
- Permite comparação justa de métricas

---

## Resultados Esperados (do Artigo)

| Cenário | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---------|-----------|--------|----|--------|---------|
| Imbalanceado | 0.97 | 0.97 | 0.97 | 0.93 | 0.99 |
| Balanceado | 0.97 | 0.97 | 0.90 | 0.97 | 0.97 |

**Latência end-to-end** (feature extraction + inferência): **0.556 ms/pacote**

---

## Comparação com Baselines (Seção 6.3)

O script compara XGBoost com 6 outros modelos usando o mesmo conjunto de features:

| Modelo | Biblioteca |
|--------|-----------|
| XGB | xgboost |
| RF | sklearn RandomForest |
| DT | sklearn DecisionTree |
| LR | sklearn LogisticRegression |
| KNN | sklearn KNeighborsClassifier |
| NB | sklearn GaussianNB |
| LGB | lightgbm (opcional) |

Todos avaliados com threshold=0.36 fixo para comparação justa.

---

## Gráficos Gerados (Figura 10)

| Arquivo | Gráfico |
|---------|---------|
| `results/figure10_performance_curves.png` | (a) ROC, (b) PR curve, (c) F1 vs Threshold, (d) DET |
| `results/figures11_12_baseline_comparison.png` | Comparação de modelos (barras) |
| `results/baseline_comparison.csv` | Tabela numérica dos baselines |
| `results/metrics_summary.json` | Métricas finais em JSON |
| `results/xgboost_someip_ids.json` | Modelo XGBoost treinado |

---

## Como Executar

```bash
# Execução completa (com CTGAN e baselines — ~30-60 min)
python experiments/files/03_train_evaluate.py \
  --train-csv data/train_features.csv \
  --test-csv  data/test_features.csv \
  --output-dir results/

# Execução rápida (sem CTGAN, sem baselines — ~5 min)
python experiments/files/03_train_evaluate.py \
  --train-csv data/train_features.csv \
  --test-csv  data/test_features.csv \
  --output-dir results/ \
  --no-ctgan \
  --no-baselines
```

---

## Estrutura da Saída

```
results/
├── xgboost_someip_ids.json           # Modelo treinado
├── metrics_summary.json              # Métricas finais (imbalanceado + balanceado)
├── baseline_comparison.csv           # Comparação com outros modelos
├── figure10_performance_curves.png   # ROC + PR + F1×threshold + DET
└── figures11_12_baseline_comparison.png  # Barras de comparação
```

---

## Notas de Reprodução

- O script usa `use_label_encoder=False` para compatibilidade com versões modernas do XGBoost
- `eval_metric="logloss"` alinha com o critério de parada do artigo
- `random_state=42` em todos os splits e modelos para reprodutibilidade
- A CTGAN pode gerar resultados ligeiramente diferentes entre execuções (não determinística por default)
