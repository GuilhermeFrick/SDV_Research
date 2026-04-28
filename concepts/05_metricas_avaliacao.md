# Capítulo 5 — Métricas de Avaliação

Este é o capítulo mais importante para interpretar os resultados do notebook.
Cada métrica mede um aspecto diferente do desempenho — nenhuma sozinha é suficiente.

---

## A Matriz de Confusão — o ponto de partida

Toda métrica de classificação deriva da matriz de confusão.

```
                    Predito
                  Normal  Ataque
Real   Normal  [   TN   |   FP  ]
       Ataque  [   FN   |   TP  ]
```

| Sigla | Nome | Significado |
|-------|------|-------------|
| **TP** | True Positive | Era ataque, previu ataque ✓ |
| **TN** | True Negative | Era normal, previu normal ✓ |
| **FP** | False Positive | Era normal, previu ataque ✗ — **falso alarme** |
| **FN** | False Negative | Era ataque, previu normal ✗ — **ataque não detectado** |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Ataque"])
fig, ax = plt.subplots(figsize=(4, 4))
disp.plot(ax=ax, colorbar=False)
plt.show()
```

---

## Precision — "quando alarmou, acertou?"

```
Precision = TP / (TP + FP)
```

- Alta precision → poucos falsos alarmes
- No IDS automotivo: sistema não vai alertar o motorista sem necessidade

**Exemplo:** o modelo alarmou 100 vezes. 97 eram ataques reais, 3 eram normais.
`Precision = 97/100 = 0.97`

```python
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
```

---

## Recall — "dos ataques reais, quantos detectou?"

```
Recall = TP / (TP + FN)
```

- Alto recall → poucos ataques escapam
- No IDS automotivo: crítico — um ataque não detectado pode ser catastrófico

**Exemplo:** havia 1000 ataques reais. O modelo detectou 970, deixou passar 30.
`Recall = 970/1000 = 0.97`

Também chamado de: **Sensibilidade**, **True Positive Rate (TPR)**.

```python
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)
```

---

## O Trade-off Precision × Recall

Eles competem entre si. Você não pode maximizar ambos ao mesmo tempo com o mesmo modelo.

```
threshold alto (ex: 0.9):
  Só alerta quando muito confiante
  ↑ Precision (poucos alarmes falsos)
  ↓ Recall (deixa passar ataques incertos)

threshold baixo (ex: 0.1):
  Alerta mesmo quando pouco confiante
  ↓ Precision (muitos alarmes falsos)
  ↑ Recall (detecta quase tudo)
```

Visualizando:
```
Precision
  1.0 |●
      | ●
      |  ●
      |    ●●
      |       ●●●
      |           ●●●●●●
  0.0 +─────────────────── Recall
      0.0                1.0
```

---

## F1-Score — a média harmônica

Combina Precision e Recall numa única métrica.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- Valoriza **equilíbrio**: um modelo com Precision=1.0 e Recall=0.0 tem F1=0
- É a métrica central do artigo de Kim et al.

**Por que média harmônica e não média aritmética?**

A média harmônica penaliza valores extremos:
```
Precision=1.0, Recall=0.1:
  Média aritmética = (1.0 + 0.1)/2 = 0.55  (parece razoável)
  F1 (harmônica)   = 2×(1.0×0.1)/(1.0+0.1) = 0.18  (revela que o modelo é ruim)
```

```python
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
```

---

## Acurácia — quando ela engana

```
Acurácia = (TP + TN) / Total
```

Com 88% de pacotes normais e 12% de ataques:
- Modelo que prevê "normal" para tudo: **Acurácia = 88%** — parece ótimo!
- Precision = 0 (nunca alarmou)
- Recall = 0 (nunca detectou nada)
- F1 = 0

**Conclusão:** nunca use acurácia sozinha em problemas desbalanceados.

---

## ROC-AUC — independente do threshold

### Curva ROC

Plota **Recall (TPR)** no eixo Y vs. **Taxa de Falso Positivo (FPR)** no eixo X,
para **todos os thresholds possíveis**.

```
TPR (Recall)
  1.0 |         ●●●●●●●●
      |       ●●
      |      ●
      |    ●●
      |   ●
      |  ●
  0.0 |●
      +─────────────────── FPR
      0.0               1.0
```

- Curva na diagonal = modelo aleatório (AUC = 0.5)
- Curva próxima do canto superior esquerdo = modelo perfeito (AUC = 1.0)

### AUC — Área Sob a Curva

```
AUC = 0.5 → modelo aleatório (inútil)
AUC = 0.7 → razoável
AUC = 0.9 → bom
AUC = 0.99 → excelente (resultado do artigo)
```

**Interpretação probabilística:** "AUC = probabilidade de que, pegando um ataque e um normal aleatoriamente, o modelo rankeie o ataque com score mais alto".

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
plt.plot([0,1],[0,1], "k:")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(); plt.show()
```

---

## PR-AUC — melhor para dados desbalanceados

### Curva Precision-Recall

Plota **Precision** no eixo Y vs. **Recall** no eixo X para todos os thresholds.

- Mais informativa que ROC quando a classe positiva (ataque) é rara
- Não é afetada pelo grande número de TN (normais)

```
Precision
  1.0 |●●●●
      |     ●●●
      |        ●●
      |          ●●●●
      |              ●●
  0.0 +─────────────────── Recall
      0.0               1.0
```

PR-AUC perfeito = 1.0.
PR-AUC de um classificador aleatório = proporção da classe positiva (ex: 0.12 para 12% de ataques).

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

pr_auc = average_precision_score(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
```

---

## Curva DET — Detection Error Tradeoff

Variante da ROC usada em sistemas de detecção.

```
Eixo X: FMR (False Match Rate) = FP / (FP + TN)
Eixo Y: FNMR (False Non-Match Rate) = FN / (FN + TP)
```

- Ponto ideal: canto inferior esquerdo (zero falsos de ambos os tipos)
- Escala logarítmica — permite ver diferenças em zonas de baixo erro
- Comum em biometria e sistemas de detecção de intrusão

```python
from sklearn.metrics import det_curve
fmr, fnmr, _ = det_curve(y_test, y_prob)

plt.plot(fmr, fnmr)
plt.xlabel("FMR"); plt.ylabel("FNMR")
plt.title("Curva DET")
plt.show()
```

---

## Como as métricas se complementam

| Métrica | Pergunta respondida | Quando priorizar |
|---------|--------------------|--------------------|
| **F1** | Equilíbrio geral | Comparação de modelos |
| **Precision** | Confiabilidade dos alertas | Custo alto de falso alarme |
| **Recall** | Cobertura de ataques | Custo alto de ataque não detectado |
| **ROC-AUC** | Qualidade do rankeamento | Dataset balanceado |
| **PR-AUC** | Qualidade com desbalanceamento | Dataset desbalanceado (nosso caso) |
| **DET** | Trade-off de erros | Comparação com sistemas biométricos |

---

## Resultados do artigo em contexto

| Cenário | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---------|-----------|--------|----|--------|---------|
| Imbalanceado | 0.97 | 0.97 | 0.97 | 0.93 | 0.99 |
| Balanceado | 0.97 | 0.97 | 0.90 | 0.97 | 0.97 |

**Lendo:** "Com threshold otimizado (0.36), o modelo acerta 97% dos alertas e detecta 97% dos ataques. A área sob a curva ROC é 0.99 — quase perfeita."

**Por que F1 cai de 0.97 para 0.90 no balanceado?**
No cenário balanceado há muito mais normais sendo comparados — o threshold 0.36 que era ótimo para a distribuição real não é mais ideal para 50/50. O PR-AUC sobe de 0.93 para 0.97 porque há mais normais "difíceis" para separar.

---

## Exemplo prático: calculando tudo

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

def relatorio_completo(y_true, y_prob, threshold, nome="Modelo"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*40}")
    print(f"Relatório: {nome} (threshold={threshold:.2f})")
    print(f"{'='*40}")
    print(f"  TP={tp:>8,}  FP={fp:>8,}")
    print(f"  FN={fn:>8,}  TN={tn:>8,}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_true, y_pred):.4f}")
    print(f"  PR-AUC    : {average_precision_score(y_true, y_prob):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")

relatorio_completo(y_test, y_prob_test, threshold=0.36, nome="XGBoost (artigo)")
relatorio_completo(y_test, y_prob_test, threshold=0.5,  nome="XGBoost (padrão)")
```

→ [Capítulo 6 — Outros Algoritmos](06_outros_algoritmos.md)
