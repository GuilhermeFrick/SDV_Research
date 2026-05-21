# Reprodução do Experimento — Kim et al. (2026)

**Artigo:** XGBoost-Based Anomaly Detection Framework for SOME/IP in In-Vehicle Networks  
**Data:** Abril de 2026  
**Ambiente:** Google Colab (GPU T4), Python 3.12

---

## 1. Objetivo

Reproduzir o pipeline de detecção de intrusão proposto por Kim et al. (2026) para redes veiculares SOME/IP, validando as métricas reportadas no artigo e analisando o impacto do balanceamento sintético via CTGAN.

---

## 2. Dataset

O dataset foi obtido publicamente via Figshare, disponibilizado pelos próprios autores.

| Conjunto | Amostras | Normal | Ataque |
|---|---|---|---|
| Treino | 7.116.674 | 6.285.515 (88,32%) | 831.159 (11,68%) |
| Teste | 7.116.673 | 6.285.514 (88,32%) | 831.159 (11,68%) |

O dataset contém 12 features comportamentais pre-extraidas pelos autores, organizadas em 5 categorias:

| Categoria | Features |
|---|---|
| Time interval | IP time interval |
| Payload likelihood | SOME/IP likelihood, SOME/IP-SD likelihood, TCP/UDP likelihood |
| Payload entropy | SOME/IP entropy, SOME/IP-SD entropy, TCP/UDP entropy |
| Payload changes | SOME/IP payload changes, SOME/IP-SD payload changes, TCP/UDP payload changes |
| Length changes | IP length changes, TCP/UDP length changes |

---

## 3. Pipeline Implementado

### 3.1 Pre-processamento

- Normalização Min-Max aplicada conforme descrito na Secao 5.3 do artigo (*"continuous variables... normalized using min-max normalization"*). O artigo nao especifica se o fit foi feito apenas no treino; na reproducao, o scaler foi ajustado exclusivamente sobre o treino e aplicado ao teste, seguindo boas praticas para evitar vazamento de informacao.
- Para selecao do threshold, o artigo indica apenas *"training-side evaluation"* e *"validation data"* (Secao 6, Figura 10c), sem detalhar o percentual de split. Na reproducao foi adotado um split estratificado de 10% do treino como conjunto de validacao. Essa diferenca explica a discrepancia de threshold: o artigo reporta 0.36, enquanto a reproducao obteve 0.43 (baseline) — ainda assim, as metricas finais no conjunto de teste convergem.

### 3.2 Modelo XGBoost

Hiperparâmetros idênticos ao artigo:

```
n_estimators    = 1000
learning_rate   = 0.05
max_depth       = 6
subsample       = 0.8
colsample_bytree= 0.8
min_child_weight= 1
reg_lambda      = 1.0
gamma           = 0.0
objective       = binary:logistic
```

### 3.3 CTGAN

Configuração conforme o artigo:

```
embedding_dim      = 128
generator_dim      = [256, 256]
discriminator_dim  = [256, 256]
batch_size         = 500
epochs             = 100
pac                = 10
```

Estratégia: treinar o CTGAN exclusivamente sobre os ataques reais do treino e gerar amostras sintéticas para balancear o dataset. O treino aumentado resultou em aproximadamente 12 milhões de amostras.

### 3.4 Seleção de Threshold

O artigo descreve (Secao 5.4.2): *"By evaluating multiple candidate thresholds on validation data and selecting the one that maximizes the target metric (e.g., F1)"*. O threshold reportado no artigo e 0.36, obtido sobre dados de validacao do lado do treino. Na reproducao, com split de 10% separado, o threshold otimizado foi 0.43 (baseline) — valor diferente, mas que resulta nas mesmas metricas arredondadas no conjunto de teste.

### 3.5 Cenários de Avaliação

- **Cenário A (realista):** teste desbalanceado (6,2M normal vs 831K ataque)
- **Cenário B (controlado):** teste balanceado por downsampling (831K vs 831K)

---

## 4. Dificuldades Encontradas

### 4.1 Codigo nao disponivel

O artigo nao disponibiliza codigo publico. Todo o pipeline foi reconstruido a partir da descricao metodologica das secoes 4 e 5 do artigo.

### 4.2 Interpretacao das metricas

O artigo reporta Precision, Recall e F1 = 0.97 sem especificar explicitamente que sao metricas **weighted average**. A analise inicial comparava essas metricas com as metricas da **classe ataque** (F1 = 0.84), gerando aparente discrepancia. Apos identificar que o artigo usa weighted average, os resultados convergiram.

### 4.3 Custo computacional do CTGAN

O treinamento do CTGAN com 748.043 amostras de ataque por 100 epochs levou aproximadamente 22 minutos em GPU T4 no Google Colab. A geracao dos 5,6 milhoes de amostras sinteticas levou menos de 1 minuto.

### 4.4 Gerenciamento de memoria

O dataset completo materializado em memoria ocupa aproximadamente 3,2 GB (float32). O dataset aumentado pelo CTGAN ocupa cerca de 4,6 GB adicionais. Foi necessario uso de `mmap_mode` no carregamento inicial e coleta de lixo explicita entre etapas.


---

## 5. Resultados

### 5.1 Metricas globais — comparacao com o artigo

| Metrica | Artigo | Obtido | Status |
|---|---|---|---|
| PR-AUC | 0.93 | 0.9310 | Reproduzido |
| ROC-AUC | 0.99 | 0.9873 | Reproduzido |
| F1 weighted | 0.97 | 0.9660 | Reproduzido |
| Accuracy | ~0.97 | 0.9681 | Reproduzido |

### 5.2 Metricas da classe ataque (cenario A — desbalanceado)

| Modelo | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (sem CTGAN) | 0.431 | 0.988 | 0.737 | 0.845 |
| CTGAN + XGBoost | 0.399 | 0.987 | 0.737 | 0.844 |

### 5.3 Cenario com recall alto (threshold reduzido para ~0.97 de recall)

| Modelo | Threshold | Precision | Recall | Falsos Positivos |
|---|---|---|---|---|
| Baseline | 0.137 | 0.593 | 0.970 | 554.346 |
| CTGAN + XGBoost | 0.143 | 0.592 | 0.971 | 556.335 |

### 5.4 Cenario B — teste balanceado (831K vs 831K)

| Modelo | F1 | Precision | Recall |
|---|---|---|---|
| Baseline | 0.9429 | 0.9169 | 0.9703 |
| CTGAN + XGBoost | 0.9429 | 0.9167 | 0.9706 |

### 5.5 Comparacao com outros algoritmos (Secao 6.3 do artigo)

Reproducao da avaliacao comparativa do artigo. Threshold otimizado por F1 no conjunto de validconsu(10% do treino). Metricas weighted sobre o conjunto de teste desbalanceado (7,1M amostras).

| Modelo | Threshold | F1 (w) | Precision (w) | Recall (w) | PR-AUC | F1 Ataque | Recall Ataque |
|---|---|---|---|---|---|---|---|
| XGB | 0.431 | 0.9663 | 0.9690 | 0.9683 | 0.9315 | 0.8445 | 0.7372 |
| LGB | 0.372 | 0.9660 | 0.9687 | 0.9681 | 0.9300 | 0.8434 | 0.7364 |
| RF† | 0.710 | 0.9654 | 0.9680 | 0.9675 | 0.9260 | 0.8404 | 0.7336 |
| DT‡ | 1.000 | 0.9569 | 0.9567 | 0.9572 | 0.7083 | 0.8139 | **0.8017** |
| LR | 0.319 | 0.9297 | 0.9287 | 0.9312 | 0.7031 | 0.6911 | 0.6592 |
| NB‡ | 1.000 | 0.9078 | 0.9064 | 0.9095 | 0.4901 | 0.5961 | 0.5719 |
| KNN*† | 0.800 | 0.9644 | 0.9672 | 0.9666 | 0.8674 | 0.8354 | 0.7261 |
| AE | ~0.000 | 0.9317 | 0.9310 | 0.9325 | 0.7027 | 0.7032 | 0.6848 |
| IF*† | 0.047 | 0.8892 | 0.8934 | 0.8858 | 0.4888 | 0.5451 | 0.5860 |
| LOF*† | 1.002 | 0.5579 | 0.8259 | 0.4752 | 0.1230 | 0.2355 | 0.6920 |

*\* Avaliado em subamostra (dataset completo inviavel no Colab — ver Secao 4)*  
*† Limitacoes computacionais: RF e KNN usam 1M e 100k amostras respectivamente; IF usa 500k; LOF usa 50k*  
*‡ Score discreto: `predict_proba` retorna 0 ou 1; threshold=1.0 equivale a predicao hard*

---

## 6. Analise Critica

### 6.1 Sobre o F1 = 0.97 reportado

O artigo reporta Precision, Recall e F1 = 0.97, porém essas metricas sao **weighted averages** calculadas sobre ambas as classes. Como o dataset tem 88% de trafego normal, a metrica ponderada e dominada pela classe normal e resulta em valores elevados mesmo com deteccao de ataque deficiente.

As metricas reais da **classe ataque** sao:
- Recall = 0.737 (26% dos ataques nao sao detectados com o threshold otimo de F1)
- Para IDS de segurança veicular, recall de 0.737 e operacionalmente critico

### 6.2 Sobre o CTGAN

O CTGAN nao trouxe melhora mensuravel em nenhum dos cenarios avaliados:

| Metrica | Delta (CTGAN vs Baseline) |
|---|---|
| Weighted F1 | -0.0002 |
| Attack F1 | -0.0009 |
| Attack Recall | -0.0004 |

Custo computacional: 22 minutos de GPU. Beneficio: nulo.

Possiveis explicacoes: as 12 features comportamentais ja sao suficientemente discriminativas para o XGBoost; o dataset original ja contem representacao adequada dos padroes de ataque; o CTGAN nao consegue adicionar variabilidade relevante ao espaco de features.

### 6.3 Sobre a avaliacao comparativa

**DT com maior attack recall (0.802):** O Decision Tree produz probabilidades discretas (0 ou 1 por folha pura). Com threshold=1.0, o resultado e identico a predicao hard do DT — que por construcao maximiza pureza de classe. O recall alto nao reflete melhor discriminacao probabilistica, como evidenciado pelo PR-AUC baixo (0.708). Esse artefato deve ser considerado ao interpretar a tabela: DT nao e superior ao XGB em deteccao, apenas em recall bruto com predicao binaria.

**LOF inviavel em Colab:** O LOF avaliado em subamostra de 50k amostras nao e comparavel com o dataset completo. O resultado (F1=0.558) reflete a limitacao de subamostragem, nao o desempenho real do algoritmo. O artigo reporta tempo de execucao de 5.295s para LOF no dataset completo, o que e inviavel no Colab sem GPU e com restricoes de memoria.

**AE threshold proximo de zero:** O threshold de ~0.000 do Autoencoder indica que a distribuicao de erros de reconstrucao entre classes normal e ataque tem sobreposicao proxima do limite inferior. O AE nao consegue separar adequadamente os dois tipos de trafego, o que resulta em attack recall (0.685) abaixo dos modelos supervisionados.

**Modelos ensemble dominantes:** XGB, LGB, RF e KNN* (subamostra) agrupam-se no topo com weighted F1 entre 0.965 e 0.966 e attack recall entre 0.726 e 0.737. A diferenca entre eles e marginal — o ganho de cada modelo adicional sobre o XGB e insignificante dado o custo computacional.

### 6.4 Trade-off operacional

Para atingir recall de ataque proximo a 0.97, o threshold precisa ser reduzido para ~0.14, o que gera aproximadamente 554.000 falsos positivos no conjunto de teste. Em operacao real (1.800 pacotes/segundo), isso corresponde a alertas falsos continuos e inviabiliza o uso pratico como IDS em tempo real sem pos-filtragem.

---

## 7. Conclusao

O pipeline foi reproduzido com sucesso. As metricas globais obtidas convergem com os valores reportados no artigo quando arredondadas para 2 casas decimais (PR-AUC = 0.93, ROC-AUC = 0.99, weighted F1 = 0.97).

A reproducao revelou tres pontos nao destacados no artigo:

1. O F1 = 0.97 e uma metrica weighted, nao a metrica da classe ataque, o que pode induzir interpretacao otimista da capacidade de deteccao.
2. O CTGAN, componente central do pipeline proposto, nao contribui para a melhora do desempenho neste dataset, questionando sua necessidade pratica.
3. A avaliacao comparativa (Secao 5.5) mostra que XGB, LGB, RF e KNN produzem resultados virtualmente identicos (weighted F1 ~0.966, attack recall ~0.73). O DT apresenta o maior attack recall (0.802) por um artefato de predicao discreta, nao por melhor discriminacao. Modelos nao-supervisionados (AE, IF, LOF) ficam significativamente abaixo dos supervisionados, indicando que as 12 features comportamentais fornecem sinal discriminativo adequado para abordagens supervisionadas simples.
