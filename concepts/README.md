# Guia de Estudo — Machine Learning para IDS Automotivo

Material de apoio para a reprodução de Kim et al. (2026).
Cada capítulo é autocontido mas segue uma progressão de dificuldade.

---

## Roteiro de Leitura

| # | Capítulo | O que você vai saber ao final |
|---|----------|-------------------------------|
| 1 | [Fundamentos de ML](01_fundamentos_ml.md) | O que é aprendizado supervisionado, features, labels, treino/teste |
| 2 | [Árvores de Decisão](02_arvores_decisao.md) | Como uma árvore decide, overfitting, profundidade, impureza |
| 3 | [Ensemble e Boosting](03_ensemble_boosting.md) | Random Forest, Gradient Boosting, por que ensembles são melhores |
| 4 | [XGBoost em Profundidade](04_xgboost.md) | Cada hiperparâmetro da Tabela 2, regularização, early stopping |
| 5 | [Métricas de Avaliação](05_metricas_avaliacao.md) | Precision, Recall, F1, ROC-AUC, PR-AUC, DET, threshold |
| 6 | [Outros Algoritmos](06_outros_algoritmos.md) | LR, KNN, NB, LightGBM — quando cada um brilha ou falha |
| 7 | [Desbalanceamento de Classes](07_desbalanceamento.md) | SMOTE, CTGAN, scale_pos_weight, downsampling |
| 8 | [Como Escolher o Algoritmo](08_como_escolher.md) | Cheat sheet, árvore de decisão de escolha, trade-offs |

---

## Como usar este material

1. Leia na ordem — cada capítulo usa conceitos do anterior.
2. Execute os blocos de código no Jupyter ou num script `.py`.
3. Tente modificar os exemplos (mude parâmetros, observe o efeito).
4. Quando voltar ao notebook principal, releia a seção correspondente.

---

## Pré-requisito mínimo

```python
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm
```
