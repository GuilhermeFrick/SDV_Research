# Capítulo 3 — Ensemble e Boosting

Uma única árvore é fraca e instável. A ideia dos métodos ensemble é: **combinar muitas árvores fracas para criar um modelo forte**.

Existem duas estratégias principais: **Bagging** (Random Forest) e **Boosting** (XGBoost, LightGBM).

---

## Bagging — Random Forest

### A ideia

Treina **N árvores independentes em paralelo**, cada uma em uma amostra aleatória dos dados.
Na predição, faz uma **votação** entre todas as árvores.

```
Dados originais (1000 amostras)
    ↓
Bootstrap 1 (800 amostras aleatórias c/ reposição) → Árvore 1 → prediz "Ataque"
Bootstrap 2 (800 amostras aleatórias c/ reposição) → Árvore 2 → prediz "Normal"
Bootstrap 3 (800 amostras aleatórias c/ reposição) → Árvore 3 → prediz "Ataque"
...
Bootstrap N → Árvore N → prediz "Ataque"

Votação: 3 "Ataque" vs 1 "Normal" → predição final: "ATAQUE"
```

O "Random" do nome vem de outro truque: cada árvore vê apenas uma **subconjunto aleatório das features** em cada corte. Isso força diversidade entre as árvores.

### Por que funciona?

Cada árvore comete erros **diferentes**. Quando você tira a média dos erros, eles se cancelam parcialmente.

Analogia: se você pede a 100 pessoas para estimar o peso de uma caixa, a **média** é quase sempre mais precisa do que qualquer estimativa individual.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,    # número de árvores
    max_depth=None,      # cada árvore pode crescer livremente
    max_features="sqrt", # raiz quadrada das features em cada corte
    random_state=42,
    n_jobs=-1            # usa todos os núcleos da CPU
)
rf.fit(X_train, y_train)
```

### Vantagens do Random Forest
- Muito robusto ao overfitting
- Funciona bem sem normalização das features
- Fornece importância de features
- Paralelizável (árvores independentes)

### Desvantagem
- Menos preciso que o Boosting em problemas complexos
- Modelo mais pesado (muitas árvores independentes)

---

## Boosting — A estratégia do XGBoost

### A ideia fundamental

Em vez de árvores independentes, o Boosting treina árvores **sequencialmente**:
cada nova árvore foca nos exemplos que a árvore anterior **errou**.

```
Dados originais
    ↓
Árvore 1 → faz predições → calcula erros
    ↓
Árvore 2 → treinada nos ERROS da Árvore 1 → faz predições → calcula erros
    ↓
Árvore 3 → treinada nos ERROS das Árvores 1+2 → ...
    ↓
...
Predição final = soma ponderada de todas as árvores
```

Cada árvore "corrige" um pouco as falhas das anteriores. Após 1000 iterações, o modelo é muito preciso.

### Comparação visual

```
BAGGING (Random Forest):
  Árvore 1 → ●●●● (treina em dados aleatórios, independente)
  Árvore 2 → ●●●● (treina em dados aleatórios, independente)
  Árvore 3 → ●●●● (treina em dados aleatórios, independente)
  Resultado: votação das três

BOOSTING (XGBoost):
  Árvore 1 → ●●●● (treina em todos os dados)
  Árvore 2 → ○●●● (foca nos erros da Árvore 1 — ● = erro, ○ = acerto)
  Árvore 3 → ○○●● (foca nos erros das Árvores 1+2)
  Resultado: soma das três com pesos
```

---

## Gradient Boosting — A matemática simplificada

"Gradient" Boosting porque cada nova árvore é ajustada na direção do **gradiente** da função de perda.

Intuição sem cálculo:
1. O modelo atual prevê `ŷ` para cada amostra
2. O "erro" é `y - ŷ` (resíduo)
3. A próxima árvore aprende a prever esse resíduo
4. A predição final é: `ŷ_final = ŷ₁ + lr × ŷ₂ + lr × ŷ₃ + ...`

Onde `lr` é o **learning rate** — quão grande é cada passo de correção.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,  # tamanho do passo de correção
    max_depth=3,        # árvores rasas são suficientes no Boosting
    random_state=42
)
gb.fit(X_train, y_train)
```

---

## Learning Rate — o parâmetro mais crítico do Boosting

```
learning_rate grande (ex: 0.3):
  Correções agressivas → aprende rápido → risco de overfitting

learning_rate pequeno (ex: 0.05):
  Correções conservadoras → precisa de mais árvores → mais preciso

Regra prática: learning_rate × n_estimators ≈ constante
  Se você dobra n_estimators, divida learning_rate por 2
```

O artigo usa `learning_rate=0.05` com `n_estimators=1000` — configuração conservadora que maximiza precisão ao custo de tempo de treino.

---

## Bagging vs. Boosting — quando usar cada um?

| Critério | Bagging (RF) | Boosting (XGB) |
|----------|-------------|----------------|
| Dataset ruidoso | Melhor | Pode superajustar ao ruído |
| Máxima precisão | Bom | Geralmente melhor |
| Velocidade de treino | Mais rápido (paralelo) | Mais lento (sequencial) |
| Velocidade de inferência | Similar | Similar |
| Interpretabilidade | Boa | Boa (com SHAP) |
| Sensível a hiperparâmetros | Menos | Mais |

---

## Exemplo comparando os três

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
import time

modelos = {
    "Árvore única":      DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost":           xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss"),
}

for nome, clf in modelos.items():
    t0 = time.time()
    clf.fit(X_train, y_train)
    t_treino = time.time() - t0

    f1_tr = f1_score(y_train, clf.predict(X_train))
    f1_te = f1_score(y_test,  clf.predict(X_test))
    print(f"{nome:20s}  F1-treino={f1_tr:.3f}  F1-teste={f1_te:.3f}  [{t_treino:.1f}s]")
```

---

## Resumo

- **Ensemble** = combinar várias árvores fracas
- **Bagging** (Random Forest) = árvores independentes em paralelo + votação
- **Boosting** (XGBoost) = árvores sequenciais, cada uma corrige os erros da anterior
- Learning rate controla o tamanho do passo de correção — menor = mais preciso, mais lento
- XGBoost geralmente supera RF em precisão; RF é mais simples de usar

→ [Capítulo 4 — XGBoost em Profundidade](04_xgboost.md)
