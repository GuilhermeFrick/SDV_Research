# Capítulo 4 — XGBoost em Profundidade

XGBoost (eXtreme Gradient Boosting) é o algoritmo central do artigo de Kim et al.
Este capítulo detalha cada parâmetro da **Tabela 2** e explica o porquê de cada escolha.

---

## O que o "eXtreme" acrescenta ao Gradient Boosting

O XGBoost adicionou 4 inovações sobre o Gradient Boosting clássico:

1. **Regularização L1 e L2** — penaliza árvores complexas (evita overfitting)
2. **Paralelismo** — constrói as árvores mais rápido usando múltiplos núcleos
3. **Tratamento de valores ausentes** — aprende a direção padrão para NaN
4. **Poda de árvores** — remove ramos que não contribuem (via `gamma`)

---

## A Tabela 2 do Artigo — explicada linha a linha

```python
import xgboost as xgb

modelo = xgb.XGBClassifier(
    objective        = "binary:logistic",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    reg_lambda       = 1.0,
    gamma            = 0.0,
    eval_metric      = "logloss",
    random_state     = 42,
)
```

---

### `objective = "binary:logistic"`

Define o **tipo de problema** e a função de perda.

- `binary:logistic` → classificação binária (0 ou 1), saída é uma probabilidade entre 0 e 1
- `multi:softmax` → classificação com mais de 2 classes
- `reg:squarederror` → regressão

A função `logistic` transforma qualquer número real numa probabilidade:
```
P(ataque) = 1 / (1 + e^(-score))

score = -3  →  P = 0.05  (muito provavelmente normal)
score =  0  →  P = 0.50  (incerto)
score = +3  →  P = 0.95  (muito provavelmente ataque)
```

---

### `n_estimators = 1000`

Número de árvores na sequência de boosting. Cada árvore corrige os erros das anteriores.

- Mais árvores = mais capacidade = mais tempo de treino
- Com learning_rate=0.05, 1000 árvores são necessárias para convergir

```python
# Para monitorar quando parar (early stopping)
modelo.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,  # para se o teste não melhorar em 50 rounds
    verbose=100
)
print(f"Melhor iteração: {modelo.best_iteration}")
```

---

### `learning_rate = 0.05`

Controla o tamanho do passo de correção a cada árvore.

```
Predição final = Σᵢ lr × árvore_i(x)

lr = 0.05: cada árvore contribui com 5% do que calculou
```

**Trade-off:**
```
lr alto (0.3)  → converge rápido → pode overfittar
lr baixo (0.01) → precisa de mais árvores → mais lento mas mais preciso
lr = 0.05 com 1000 árvores → configuração clássica para máxima precisão
```

---

### `max_depth = 6`

Profundidade máxima de cada árvore individual no ensemble.

No Boosting, árvores rasas (depth 3-8) funcionam melhor do que árvores profundas:
- Cada árvore captura um "pedaço" do padrão
- A profundidade total é construída iterativamente pelo ensemble

```
depth=3: 2³ = 8 folhas possíveis
depth=6: 2⁶ = 64 folhas possíveis  ← escolha do artigo
depth=9: 2⁹ = 512 folhas possíveis (risco de overfitting)
```

---

### `subsample = 0.8`

Fração aleatória das **amostras** usadas para construir cada árvore.

```
0.8 → cada árvore vê 80% dos dados de treino (escolhidos aleatoriamente)
```

- Evita overfitting (nenhuma árvore memoriza todos os dados)
- Adiciona diversidade ao ensemble (similar ao bagging)
- Também acelera o treino

Valores típicos: 0.6 a 0.9.

---

### `colsample_bytree = 0.8`

Fração aleatória das **features** usadas para construir cada árvore.

```
0.8 com 12 features → cada árvore usa 9-10 features aleatórias
```

- Mesma ideia do Random Forest: força diversidade entre árvores
- Previne que uma feature muito forte domine todas as árvores

---

### `min_child_weight = 1`

Peso mínimo de instâncias que um nó folha deve ter.

- Valor baixo (1) = folhas podem ser pequeninhas = mais complexidade
- Valor alto (10) = folhas precisam de pelo menos 10 amostras = mais generalização

Com 7 milhões de amostras, `min_child_weight=1` não é um risco real — folhas com poucas amostras são improváveis nesse volume de dados.

---

### `reg_lambda = 1.0` — Regularização L2

Penaliza o modelo por ter pesos grandes nas folhas.

```
Função objetivo = Perda nos dados + λ × Σ(pesos das folhas)²
```

- `lambda=0` → sem regularização → overfitting
- `lambda=1` → regularização leve (padrão do XGBoost)
- `lambda=10` → regularização forte → modelo mais simples

Analogia: é como cobrar "imposto" sobre a complexidade do modelo.

---

### `gamma = 0.0` — Poda de árvores (min_split_loss)

Redução mínima na função de perda necessária para fazer um split.

```
Se o split reduz a perda em menos que gamma → não faz o split → folha
```

- `gamma=0` → qualquer split que reduza a perda é aceito
- `gamma=1` → só aceita splits que reduzam a perda em ≥ 1 unidade

Com dados de alta qualidade (como os .npy dos autores), `gamma=0` é adequado.

---

## O que é `eval_metric="logloss"`?

Métrica monitorada durante o treino para detectar overfitting.

**Log loss (entropia cruzada binária):**
```
logloss = -(1/N) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

- Penaliza mais quando o modelo está **confiante e errado**
- Diminui à medida que o modelo melhora
- Bom para monitorar com `early_stopping_rounds`

---

## Otimização do Threshold — o passo que o artigo adiciona

O XGBoost retorna uma **probabilidade** (0 a 1). A decisão final depende do threshold:

```python
y_prob = modelo.predict_proba(X_test)[:, 1]

# Threshold padrão
y_pred_05 = (y_prob >= 0.5).astype(int)

# Threshold otimizado no treino
y_pred_36 = (y_prob >= 0.36).astype(int)  # valor do artigo
```

**Por que o threshold ótimo ≠ 0.5?**

Com dados desbalanceados ou após CTGAN, a distribuição das probabilidades muda.
O threshold 0.5 pode não ser o ponto que maximiza o F1.

```python
import numpy as np
from sklearn.metrics import f1_score

y_prob_train = modelo.predict_proba(X_train)[:, 1]

best_f1, best_t = 0, 0.5
for t in np.arange(0.01, 0.99, 0.01):
    f1 = f1_score(y_train, (y_prob_train >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Threshold ótimo: {best_t:.2f}")
print(f"F1 no treino:    {best_f1:.4f}")
```

**Por que usar o treino para otimizar o threshold?**

Porque usamos o teste apenas para a **avaliação final**. Otimizar no teste seria trapaça (data leakage).

---

## scale_pos_weight — parâmetro para desbalanceamento

Não está na Tabela 2 do artigo porque o CTGAN já balanceia as classes.
Mas é essencial sem CTGAN:

```python
n_normal = (y_train == 0).sum()
n_attack = (y_train == 1).sum()

modelo = xgb.XGBClassifier(
    ...
    scale_pos_weight = n_normal / n_attack
    # Ex: 88%/12% → scale_pos_weight ≈ 7.3
    # Diz ao modelo: cada amostra de ataque "vale" 7.3 amostras normais
)
```

---

## Exemplo completo com os parâmetros do artigo

```python
import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Parâmetros exatos da Tabela 2
modelo = xgb.XGBClassifier(
    objective        = "binary:logistic",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    reg_lambda       = 1.0,
    gamma            = 0.0,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1,
)

modelo.fit(X_train, y_train, verbose=False)

# Threshold otimizado
y_prob_train = modelo.predict_proba(X_train)[:, 1]
best_f1, best_t = 0, 0.5
for t in np.arange(0.01, 0.99, 0.01):
    f1 = f1_score(y_train, (y_prob_train >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

# Avaliação final no teste
y_prob_test = modelo.predict_proba(X_test)[:, 1]
y_pred = (y_prob_test >= best_t).astype(int)

print(f"Threshold ótimo: {best_t:.2f}")
print(f"F1:      {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}")
```

---

## Resumo dos parâmetros

| Parâmetro | O que controla | Aumentar → |
|-----------|---------------|-----------|
| `n_estimators` | Nº de árvores | Mais preciso, mais lento |
| `learning_rate` | Tamanho do passo | Mais agressivo, mais rápido |
| `max_depth` | Complexidade por árvore | Mais expressivo, risco de overfit |
| `subsample` | % amostras por árvore | Mais ruído (boa diversidade) |
| `colsample_bytree` | % features por árvore | Mais diversidade |
| `reg_lambda` | Penalidade L2 | Mais simples, mais generaliza |
| `gamma` | Poda de ramos | Mais conservador |
| `scale_pos_weight` | Peso da classe positiva | Mais foco em ataques |

→ [Capítulo 5 — Métricas de Avaliação](05_metricas_avaliacao.md)
