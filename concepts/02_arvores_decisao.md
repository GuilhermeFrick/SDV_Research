# Capítulo 2 — Árvores de Decisão

A árvore de decisão é o **bloco de construção fundamental** do XGBoost e do Random Forest. Entendê-la bem é entender 80% de como esses modelos funcionam.

---

## A ideia central

Uma árvore faz perguntas binárias sobre as features até chegar a uma conclusão.

```
                    f_ip_time_interval < 0.01?
                         /           \
                       SIM            NÃO
                        |               |
            f_someip_entropy > 0.5?   NORMAL
                  /        \
                SIM         NÃO
                 |            |
              ATAQUE        NORMAL
```

Cada nó interno = uma pergunta sobre uma feature.
Cada folha = uma decisão final (classe prevista).

---

## Como a árvore decide onde fazer o corte?

O objetivo é separar as classes da forma mais "pura" possível.
Para isso, usa-se uma medida de **impureza**.

### Índice de Gini (o mais comum)

```
Gini(nó) = 1 - Σ pᵢ²
```

Onde `pᵢ` é a proporção da classe `i` no nó.

- Nó puro (só ataques): `Gini = 1 - (1² + 0²) = 0`
- Nó misto (50/50): `Gini = 1 - (0.5² + 0.5²) = 0.5`

A árvore testa todos os cortes possíveis e escolhe o que **mais reduz o Gini**.

```python
from sklearn.tree import DecisionTreeClassifier

# Gini é o padrão — mais eficiente computacionalmente
arvore_gini = DecisionTreeClassifier(criterion="gini", max_depth=3)

# Alternativa: entropia (Information Gain)
arvore_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)
```

### Exemplo numérico

Você tem 10 pacotes: 7 normais, 3 ataques.

```
Antes do corte:
  Gini = 1 - (7/10)² - (3/10)² = 1 - 0.49 - 0.09 = 0.42

Corte A: f_time < 0.01 → Esq: 6 normais, 0 ataques | Dir: 1 normal, 3 ataques
  Gini(Esq) = 1 - 1² - 0²     = 0.00
  Gini(Dir) = 1 - (1/4)² - (3/4)² = 1 - 0.0625 - 0.5625 = 0.375
  Gini(A) = (6/10)*0.00 + (4/10)*0.375 = 0.15  ← melhor!

Corte B: f_entropy > 0.3 → Esq: 5 normais, 2 ataques | Dir: 2 normais, 1 ataque
  Gini(B) = ...  ← pior

A árvore escolhe o Corte A.
```

---

## Profundidade da árvore (max_depth)

É o parâmetro mais importante de uma árvore.

```
max_depth=1:  1 pergunta  → underfitting (simples demais)
max_depth=3:  7 nós max   → geralmente bom
max_depth=6:  63 nós max  → usado no XGBoost do artigo
max_depth=∞:  memoriza tudo → overfitting
```

### Overfitting vs. Underfitting

```
         Erro
          |
Treino →  |_______________
          |               \
          |                \______ Teste
          |                        ↑ Pior com profundidade grande
          |
          +-------------------------→ Profundidade
               ↑                ↑
          Underfitting     Overfitting
```

**Underfitting**: modelo muito simples, não captura os padrões.
**Overfitting**: modelo memoriza o treino mas falha em dados novos.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

for depth in [1, 3, 5, 10, None]:  # None = sem limite
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    f1_tr = f1_score(y_train, clf.predict(X_train))
    f1_te = f1_score(y_test,  clf.predict(X_test))
    print(f"depth={str(depth):4s}  F1-treino={f1_tr:.3f}  F1-teste={f1_te:.3f}")
```

Saída típica:
```
depth=1     F1-treino=0.612  F1-teste=0.608
depth=3     F1-treino=0.841  F1-teste=0.835
depth=5     F1-treino=0.921  F1-teste=0.903
depth=10    F1-treino=0.998  F1-teste=0.871  ← começa a overfittar
depth=None  F1-treino=1.000  F1-teste=0.842  ← memorizado
```

---

## Importância de Features

A árvore naturalmente diz quais features mais ajudaram nas decisões:

```python
import pandas as pd

importances = clf.feature_importances_
df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(df)
```

Uma feature com importância alta → o modelo a usa frequentemente perto da raiz → tem grande poder discriminativo.

---

## Limitações da árvore sozinha

| Problema | Consequência |
|----------|-------------|
| Alta variância | Mudar poucas amostras no treino muda muito a árvore |
| Overfitting fácil | Precisa de `max_depth` controlado |
| Fronteiras de decisão | Só faz cortes horizontais/verticais — não captura diagonais |

**Solução**: combinar muitas árvores. É exatamente o que o Random Forest e o XGBoost fazem.

---

## Visualizando uma árvore (pequena)

```python
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Texto
print(export_text(clf, feature_names=feature_names))

# Gráfico
fig, ax = plt.subplots(figsize=(15, 6))
plot_tree(clf, feature_names=feature_names,
          class_names=["Normal", "Ataque"],
          filled=True, ax=ax)
plt.show()
```

---

## Resumo

- Árvore = sequência de perguntas binárias sobre features
- Aprende qual feature e qual corte minimizam o Gini
- `max_depth` controla complexidade e overfitting
- Sozinha é instável — o poder vem ao combinar muitas delas

→ [Capítulo 3 — Ensemble e Boosting](03_ensemble_boosting.md)
