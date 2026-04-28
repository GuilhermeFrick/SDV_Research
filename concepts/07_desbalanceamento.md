# Capítulo 7 — Desbalanceamento de Classes

Este é um dos problemas centrais do nosso projeto. Entender as técnicas de tratamento é crucial para reproduzir o artigo e para qualquer problema de detecção de anomalias.

---

## O problema

Em tráfego de rede veicular real, ataques são raros. O dataset dos autores reflete isso:

```
Normal: 88.3%  (6.282.383 pacotes)
Ataque: 11.7%  (834.291 pacotes)
```

Se um modelo aprende com esses dados sem tratamento, ele tende a:
- Prever "normal" para tudo (alta acurácia, zero recall)
- Prever "ataque" para tudo (alto recall, baixa precision)

---

## Por que o desbalanceamento confunde o modelo?

Durante o treino com Gradient Boosting, a função de perda é:

```
Loss = Σ [yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ)]
```

Com 88% de normais, os **8x mais** exemplos normais dominam o gradiente.
O modelo aprende a minimizar o erro nos normais — e ignora os ataques.

Analogia: você está estudando para uma prova com 88 questões de matemática e 12 de física. Se estudar proporcionalmente, vai muito bem em matemática mas vai passar raspando em física.

---

## Técnica 1 — scale_pos_weight (XGBoost)

A solução mais simples: diz ao XGBoost que cada amostra de ataque "vale" N amostras normais.

```python
n_normal = (y_train == 0).sum()  # 6.282.383
n_attack = (y_train == 1).sum()  #   834.291

peso = n_normal / n_attack  # ≈ 7.53

modelo = xgb.XGBClassifier(
    ...,
    scale_pos_weight = peso,  # cada ataque pesa como 7.53 normais
)
```

**Efeito:** o gradiente dos ataques é amplificado. O modelo "presta mais atenção" neles.

**Vantagens:** simples, rápido, não precisa de dados extras.
**Desvantagem:** não gera diversidade — apenas repondera o que já existe.

---

## Técnica 2 — Oversampling com SMOTE

SMOTE (Synthetic Minority Over-sampling TEchnique) cria amostras **sintéticas** da classe minoritária interpolando entre amostras reais.

```
Ataque A: [0.1, 0.9, 0.3, ...]
Ataque B: [0.2, 0.8, 0.4, ...]

Novo sintético: [0.15, 0.85, 0.35, ...]  ← ponto entre A e B
```

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Antes:  {(y_train==0).sum()} normais, {(y_train==1).sum()} ataques")
print(f"Depois: {(y_train_res==0).sum()} normais, {(y_train_res==1).sum()} ataques")
# Depois: 6.282.383 normais, 6.282.383 ataques
```

**Vantagem:** cria dados novos, aumenta diversidade.
**Desvantagem:** interpola linearmente — pode criar amostras "irreais" se as classes se sobrepõem.

---

## Técnica 3 — CTGAN (usada no artigo)

CTGAN (Conditional Tabular GAN) é uma **rede neural generativa** treinada para aprender a distribuição dos dados de ataque e gerar amostras sintéticas **realistas**.

### Como funciona

```
Dados reais de ataque
        ↓
  Generator ←→ Discriminator
   (gera)       (critica)
        ↓
  Gera amostras sintéticas
  indistinguíveis dos reais
```

O Generator e o Discriminator treinam juntos num jogo adversarial:
- Generator: tenta criar amostras que pareçam reais
- Discriminator: tenta distinguir reais de sintéticas

Após o treino, o Generator cria amostras de qualidade alta.

### Configuração do artigo

```python
from ctgan import CTGAN
import pandas as pd

# Treina CTGAN apenas nos dados de ataque
X_attack = X_train[y_train == 1]
df_attack = pd.DataFrame(X_attack, columns=feature_names)

ctgan = CTGAN(
    embedding_dim      = 128,
    generator_dim      = (256, 256),
    discriminator_dim  = (256, 256),
    batch_size         = 500,
    epochs             = 100,
)
ctgan.fit(df_attack)

# Gera até balancear (1:1 com normais)
n_synthetic = n_normal - n_attack
X_synthetic = ctgan.sample(n_synthetic).values
y_synthetic = np.ones(n_synthetic)

# Dataset augmentado
X_train_aug = np.vstack([X_train, X_synthetic])
y_train_aug = np.concatenate([y_train, y_synthetic])
```

**Por que CTGAN supera SMOTE?**

SMOTE só interpola linearmente. CTGAN aprende a distribuição multivariada complexa dos ataques — incluindo correlações não lineares entre features.

**Desvantagem:** demora muito para treinar (~30-60 minutos com 100 epochs).

---

## Técnica 4 — Downsampling

Remove exemplos da **classe majoritária** para igualar ao número da minoritária.

```python
import numpy as np

idx_attack = np.where(y_train == 1)[0]
idx_normal = np.where(y_train == 0)[0]

rng = np.random.default_rng(42)
idx_normal_sub = rng.choice(idx_normal, size=len(idx_attack), replace=False)

idx_balanced = np.concatenate([idx_attack, idx_normal_sub])
rng.shuffle(idx_balanced)

X_train_bal = X_train[idx_balanced]
y_train_bal = y_train[idx_balanced]
```

**Vantagem:** simples, rápido.
**Desvantagem:** descarta dados válidos — pode perder informação.

O artigo usa downsampling apenas no **conjunto de teste** (Cenário 2), não no treino.

---

## Técnica 5 — Threshold Adjustment

Sem mudar os dados, apenas ajuste onde você traça a linha de decisão.

```python
# Sem tratamento:
# Modelo prevê probabilidade baixa para ataques (ex: 0.3)
# Threshold padrão 0.5 → todos previtos como normais

# Com threshold 0.3:
# Os mesmos ataques com probabilidade 0.3 já são detectados

best_threshold = encontrar_melhor_threshold(y_train, y_prob_train)
y_pred = (y_prob_test >= best_threshold).astype(int)
```

**Mais simples de todas** — mas depende que o modelo já esteja rankeando corretamente.
Se o ROC-AUC é alto (≥ 0.9), threshold adjustment é suficiente.

---

## Comparação das técnicas

| Técnica | Dados novos? | Tempo | Eficácia | Quando usar |
|---------|-------------|-------|----------|-------------|
| `scale_pos_weight` | Não | Instantâneo | Boa | Sempre como primeiro passo |
| Threshold adjustment | Não | Instantâneo | Boa (se ROC-AUC ≥ 0.9) | Quando o rankeamento já é bom |
| Downsampling | Não (remove) | Instantâneo | Moderada | Dados abundantes |
| SMOTE | Sim (linear) | Rápido | Boa | Dados tabulares simples |
| CTGAN | Sim (realista) | Lento | Excelente | Máxima qualidade, tempo disponível |

---

## O nosso problema específico

Nosso pipeline (Etapa 1) **labela incorretamente**:
- PCAPs de ataque → todos os pacotes rotulados como ataque
- Mas 88% do tráfego nesses PCAPs é na verdade normal (background)

Resultado: 84% do dataset é "ataque" — o inverso do artigo.

A solução correta é fixar o **labeling** na Etapa 1.
O `scale_pos_weight` seria um paliativo que não corrige o problema raiz.

```python
# O que aconteceu no nosso caso:
# n_normal = 1.095.390  (15.8%)
# n_attack = 5.848.823  (84.2%)

# Modelo que prevê "ataque" sempre:
# Precision = 5.848.823 / 6.944.213 = 0.8423  ✓ (nosso resultado)
# Recall    = 1.0000                            ✓ (nosso resultado)
# F1        = 0.9144                            ✓ (nosso resultado)
```

→ [Capítulo 8 — Como Escolher o Algoritmo](08_como_escolher.md)
