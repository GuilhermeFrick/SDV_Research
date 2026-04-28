# Capítulo 6 — Outros Algoritmos (Baselines)

O notebook compara o XGBoost com 6 outros modelos. Entender cada um ajuda a interpretar por que o XGBoost vence — e quando os outros seriam mais adequados.

---

## Regressão Logística (LR)

### O que é

Apesar do nome "regressão", é um **classificador**. Calcula uma combinação linear das features e passa por uma função logística para obter uma probabilidade.

```
score = w₁×f₁ + w₂×f₂ + ... + w₁₂×f₁₂ + b
P(ataque) = 1 / (1 + e^(-score))
```

O modelo aprende os **pesos** `w` que melhor separam as classes.

### Intuição visual

Traça uma **linha reta** (ou hiperplano em dimensões maiores) que separa as classes.

```
f_entropy
    |  . . . . . 
    |  . . .  /  × × ×
    |  . .   /  × × × ×
    |  .    /  × × × × ×
    +-------/-------------- f_likelihood
           ↑
      Fronteira de decisão (linha reta)
. = normal    × = ataque
```

### Quando usar
- Dados **linearmente separáveis**
- Precisa de um modelo muito **rápido** de treinar e interpretar
- Quer saber **qual feature** mais influencia a decisão (via coeficientes)

### Limitação
Se a fronteira entre classes não for uma linha reta, a LR vai falhar.
No nosso dataset, a fronteira é complexa → XGBoost supera LR.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)

# Coeficientes — quais features mais influenciam
import pandas as pd
coef = pd.Series(lr.coef_[0], index=feature_names).sort_values()
print(coef)
```

---

## Árvore de Decisão (DT)

Já detalhada no Capítulo 2. Como baseline, usa-se uma árvore **sem ensemble** — serve para mostrar o ganho do XGBoost.

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
```

**No resultado do artigo:** F1=0.91 vs XGBoost F1=0.97 — 6% de melhora só por usar ensemble.

---

## Random Forest (RF)

Já detalhado no Capítulo 3. Ensemble de árvores independentes com votação.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
```

**No resultado do artigo:** F1=0.91 — empata com DT isolado.
**Por que RF não superou DT?** Com dados muito desbalanceados, o bootstrap pode piorar a diversidade.

---

## KNN — K Vizinhos Mais Próximos

### O que é

Não constrói um modelo durante o treino. Na predição, encontra os K exemplos mais próximos no espaço de features e faz votação entre eles.

```
Novo pacote → encontra os 5 mais parecidos no treino → 3 são ataques → prediz ataque
```

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)  # apenas memoriza os dados
```

### Intuição visual

```
f_entropy
    |  . . × . × . 
    |  . . .  ← novo pacote
    |  . . . (vizinhos: 4 normais, 1 ataque → prediz normal)
    +------------------------ f_likelihood
```

### Vantagens
- Simples de entender
- Funciona bem com padrões locais complexos

### Desvantagens graves para nosso problema
- **Treino "instantâneo"** mas **inferência lenta** — para cada predição, precisa calcular distâncias para todos os 7M exemplos do treino
- No notebook: **inferência = 1575s** vs XGBoost = 84s
- Inviável para IDS embarcado que precisa de latência < 1ms

### Quando usar
Datasets pequenos (< 50k amostras) onde a relação de vizinhança é significativa.

---

## Naive Bayes (NB)

### O que é

Aplica o **Teorema de Bayes** assumindo que todas as features são **independentes entre si** (a parte "Naive").

```
P(ataque | features) ∝ P(ataque) × P(f₁|ataque) × P(f₂|ataque) × ... × P(f₁₂|ataque)
```

O `GaussianNB` assume que cada feature segue uma distribuição gaussiana (normal) dentro de cada classe.

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
```

### Por que falhou no nosso caso?

**Precision=0.94 mas Recall=0.07** — detectou quase nenhum ataque.

As features **não são independentes** entre si:
- `f_someip_likelihood + f_someip_entropy = 1` sempre
- Ou seja, são 100% dependentes — viola a premissa "Naive"

Quando a independência é violada, as probabilidades calculadas ficam erradas.

### Vantagens
- Treino e inferência **extremamente rápidos**
- Funciona bem com texto (spam, sentimentos)
- Ótimo com features genuinamente independentes

---

## LightGBM (LGB)

### O que é

Competidor direto do XGBoost. Implementa Gradient Boosting com duas otimizações diferentes:

1. **GOSS** (Gradient-based One-Side Sampling): em vez de usar todas as amostras, foca nas que têm maior gradiente (maior erro)
2. **EFB** (Exclusive Feature Bundling): agrupa features exclusivas para reduzir dimensionalidade

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
```

### LGB vs. XGBoost

| Aspecto | XGBoost | LightGBM |
|---------|---------|----------|
| Velocidade de treino | Bom | **Mais rápido** (geralmente 3-10×) |
| Memória | Moderada | **Menor** |
| Crescimento de árvores | Level-wise (por nível) | Leaf-wise (por folha) |
| Risco de overfit | Menor | Maior (leaf-wise é mais agressivo) |
| Precisão | Excelente | Excelente |
| Popularidade | Alta | Alta |

**No resultado do artigo:** F1 e ROC-AUC praticamente iguais ao XGBoost.
LGB é mais rápido para treinar — escolher entre eles é questão de preferência.

---

## Tabela comparativa dos baselines

| Modelo | Como decide | Treino | Inferência | Interpretável? |
|--------|------------|--------|-----------|---------------|
| LR | Linha reta | Rápido | Rápido | Sim (coeficientes) |
| DT | Perguntas em árvore | Médio | Rápido | Sim (visualizável) |
| RF | Votação de árvores | Lento (paralelo) | Rápido | Moderado |
| **XGB** | **Boosting sequencial** | **Lento** | **Rápido** | **Moderado (SHAP)** |
| KNN | Vizinhos mais próximos | Instantâneo | Muito lento | Sim |
| NB | Probabilidades bayesianas | Muito rápido | Muito rápido | Sim |
| LGB | Boosting leaf-wise | Moderado | Rápido | Moderado |

---

## Resultado do artigo revisitado

| Modelo | F1 | ROC-AUC | Treino | Inferência |
|--------|-----|---------|--------|-----------|
| XGB | **0.97** | **0.99** | 333s | 84s |
| LGB | 0.97 | 0.99 | 261s | 172s |
| RF | 0.91 | 0.99 | 546s | 27s |
| LR | 0.97 | 0.99 | 10s | 0.2s |
| DT | 0.91 | 0.99 | 41s | 1s |
| KNN | 0.69 | 0.99 | 34s | **1575s** |
| NB | 0.14 | 0.99 | 3s | 3s |

**Observação interessante:** todos têm ROC-AUC=0.99 — o **rankeamento** está correto em todos os modelos. O que diferencia é a habilidade de encontrar o threshold certo, o que o CTGAN facilita.

→ [Capítulo 7 — Desbalanceamento de Classes](07_desbalanceamento.md)
