# Capítulo 1 — Fundamentos de Machine Learning

## O que é Machine Learning?

Programação tradicional: você escreve as regras.
Machine Learning: você mostra exemplos e o algoritmo **aprende as regras sozinho**.

```
Tradicional:  dados + regras  →  resultado
ML:           dados + resultado  →  regras (o modelo)
```

No nosso caso: mostramos ao XGBoost milhões de pacotes de rede rotulados como "normal" ou "ataque", e ele aprende quais padrões distinguem um do outro.

---

## Aprendizado Supervisionado

É o tipo de ML que usamos. "Supervisionado" porque cada exemplo de treino tem um **rótulo** (a resposta certa).

```
Exemplo de entrada:
  f_ip_time_interval   = 0.003
  f_someip_likelihood  = 0.92
  f_someip_entropy     = 0.08
  ...
  
Rótulo:
  label = 0  (normal)
  label = 1  (ataque)
```

O modelo aprende a mapear entradas → rótulo. Depois, dado um pacote **novo** (sem rótulo), ele prevê se é normal ou ataque.

---

## Terminologia essencial

### Feature (variável de entrada)
Uma medida do exemplo que o modelo usa para decidir.
No nosso projeto: `f_ip_time_interval`, `f_someip_entropy`, etc.
Também chamada de: atributo, variável independente, coluna X.

### Label (rótulo / variável alvo)
A resposta que o modelo deve prever.
No nosso projeto: `0 = normal`, `1 = ataque`.
Também chamada de: target, classe, variável dependente, coluna y.

### Amostra (sample / instância)
Um único exemplo — no nosso caso, um pacote de rede.
O dataset tem ~14 milhões de amostras.

### Treino e Teste
- **Treino**: o modelo vê os dados e aprende os padrões (50% dos dados)
- **Teste**: avaliamos o modelo em dados que ele **nunca viu** (50% dos dados)

Por que separar? Para garantir que o modelo generalizou, não apenas memorizou.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,       # 50% para teste
    stratify=y,          # mantém proporção de classes
    random_state=42      # reprodutibilidade
)
```

### Modelo
A "caixa" que recebe features e devolve uma predição.
Matematicamente: `ŷ = f(X)`, onde `f` é aprendida a partir dos dados.

---

## Classificação vs. Regressão

| Tipo | Saída | Exemplo |
|------|-------|---------|
| **Classificação** | Classe discreta | "normal" ou "ataque" |
| **Regressão** | Número contínuo | "temperatura em 3h = 23.4°C" |

Nosso problema é **classificação binária** (duas classes).

---

## O ciclo de ML

```
1. Coletar dados
      ↓
2. Extrair features (Etapa 2 do pipeline)
      ↓
3. Dividir em treino/teste
      ↓
4. Treinar o modelo (Etapa 3)
      ↓
5. Avaliar métricas no teste
      ↓
6. Ajustar (hiperparâmetros, features, dados)
      ↓
   Repetir até satisfatório
```

---

## Normalização

Algoritmos como KNN e Regressão Logística são sensíveis à **escala** das features.
Se uma feature vai de 0 a 10.000 e outra de 0 a 1, a primeira domina.

**Min-Max (usada no artigo — Equação 8):**
```
x' = (x - x_min) / (x_max - x_min)
```
Resultado: tudo entre 0 e 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)   # aprende min/max no treino
X_test_norm  = scaler.transform(X_test)        # aplica os mesmos parâmetros no teste
```

**Regra de ouro:** `fit` apenas no treino. Nunca no teste — senão você "vaza" informação do futuro.

---

## Exemplo prático completo

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gera dataset sintético: 1000 amostras, 5 features, 2 classes
X, y = make_classification(
    n_samples=1000, n_features=5,
    n_informative=3, n_redundant=2,
    random_state=42
)

# Divide
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Treina
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# Avalia
y_pred = modelo.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
```

---

## Por que acurácia não é suficiente?

Se 95% dos pacotes são normais e o modelo sempre prevê "normal":
- **Acurácia = 95%** — parece ótimo!
- **Detecta ataques = 0%** — é inútil como IDS

É por isso que usamos Precision, Recall e F1. Veja o Capítulo 5.

---

## Próximo capítulo

Com esses fundamentos, você está pronto para entender como as **Árvores de Decisão** tomam decisões — a base de tudo que vem depois, incluindo o XGBoost.

→ [Capítulo 2 — Árvores de Decisão](02_arvores_decisao.md)
