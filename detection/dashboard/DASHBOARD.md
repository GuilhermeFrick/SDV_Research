# Dashboard — SOME/IP IDS Pipeline

Interface Streamlit para análise, mistura de datasets e avaliação do classificador multiclasse de intrusão em redes veiculares SOME/IP.

**Como iniciar:**
```bash
streamlit run detection/dashboard/app.py
```

---

## Páginas

### 📊 Visão Geral dos Datasets

**O que mostra:**  
Metadados de cada arquivo CSV já parseado e labelado pelo pipeline Kim 2026.

**Tabela de arquivos:**  
Para cada CSV exibe o total de pacotes, quantos estão rotulados como benigno (`label=0`) e quantos como anomalia (`label=1`), o percentual de ataque e a duração da captura em segundos.

| Coluna | Significado |
|---|---|
| Total | Todos os pacotes do arquivo |
| Benigno (label=0) | Pacotes classificados como tráfego normal |
| Anomalia (label=1) | Pacotes classificados como ataque |
| % Anomalia | Fração do arquivo que é ataque |
| Duração (s) | Intervalo entre o primeiro e o último pacote |

**Gráfico de composição (barras empilhadas):**  
Cada barra representa um arquivo. A parte verde é benigno, a vermelha é anomalia. O `benign_traffic.csv` é 100% verde por definição — é a captura de referência sem ataque.

**Gráfico de range temporal:**  
Mostra onde no tempo (eixo X = Unix timestamp) cada arquivo foi capturado. Cada barra começa no primeiro pacote e termina no último.

> **Interpretação chave:** as barras não se sobrepõem — cada cenário de ataque foi uma captura isolada, realizada em momentos diferentes. Isso significa que os datasets não compartilham contexto temporal real, o que motiva a criação do **dataset misto** (ver página Mixer).

---

### 🔀 Mixer de Datasets

**O que faz:**  
Permite configurar e visualizar como os arquivos de ataque serão combinados com o tráfego benigno para gerar um dataset misto que simula um cenário realista de rede veicular.

**Por que é necessário:**  
Em operação real, um veículo pode sofrer um ataque DoS enquanto tráfego SOME/IP legítimo continua circulando — e possivelmente um MITM em paralelo. Os datasets individuais capturam cada ataque em isolamento. O mixer sintetiza um fluxo onde múltiplos eventos coexistem no tempo.

**Configurações disponíveis:**

| Parâmetro | Descrição |
|---|---|
| Ataques a incluir | Quais arquivos de ataque participam do mix (DoS, Fuzzy, MITM Multi, MITM Single) |
| % de pacotes benigno | Controla a proporção de tráfego normal no dataset final |
| Estratégia de mistura | Como os pacotes são ordenados após a combinação |

**Estratégias de mistura:**

- **Temporal (por timestamp):** normaliza os timestamps de cada arquivo para começar em t=0 e intercala os pacotes em ordem cronológica relativa. É a estratégia mais próxima de um cenário real — o ataque "acontece" durante o tráfego benigno, não antes nem depois.

- **Aleatório (embaralhado):** descarta a ordem temporal e embaralha todos os pacotes. Garante que o modelo não aprenda padrões de sequência, apenas de feature por pacote.

- **Proporcional:** mantém ordenação temporal mas ajusta a quantidade de cada classe para o ratio configurado via slider.

**Gráfico de densidade temporal (preview):**  
Após clicar em "Gerar Preview", exibe um gráfico de área onde cada cor representa uma classe. A sobreposição vertical entre cores indica que naquele intervalo de tempo existem pacotes de diferentes classes — é exatamente o cenário que o modelo precisará classificar em produção.

> **Como explicar para a banca:** *"O gráfico mostra que, no dataset misto, um pacote que chega no segundo 50 pode ser benigno ou DoS. O modelo não pode usar o contexto 'agora estou no meio de um ataque DoS' — precisa classificar baseado apenas nas features comportamentais daquele pacote e de sua janela deslizante."*

---

### ⚙️ Pipeline

**O que faz:**  
Interface para executar os scripts de extração de features e treino diretamente pelo browser, sem precisar de terminal.

**Etapas disponíveis:**

| Etapa | Script | O que faz |
|---|---|---|
| 1 | `multiclass/01_features.py` | Lê todos os CSVs Kim, extrai 12 features stateful, gera `features.csv` com 5 classes |
| 2 | `multiclass/01b_merge_fakeclientid.py` | Adiciona classe FakeClientID (Alkhatib 2021), recalcula normalização, gera arrays `.npy` |
| 3 | `multiclass/02_train.py` | Treina XGBoost `multi:softprob` (6 classes, 200 estimadores), salva modelo e métricas |

**Tabela de status:**  
Lista todos os arquivos intermediários e finais do pipeline com indicação de existência e tamanho em MB. Permite verificar de um relance se o pipeline foi executado completamente.

---

### 📈 Resultados

**O que mostra:**  
Avaliação completa do classificador multiclasse a partir do `results.json` gerado pelo treino.

**Métricas globais:**

| Métrica | Valor | Significado |
|---|---|---|
| F1 Macro | 99,91% | Média do F1 por classe, sem peso por volume |
| F1 Weighted | 99,91% | Média do F1 ponderada pelo número de amostras por classe |
| Acurácia | 99,91% | Fração de pacotes classificados corretamente |
| Latência/pkt | 0,79 ms | Tempo para classificar um único pacote (benchmark em loop de 1000) |
| Throughput | 225.813 pkt/s | Capacidade em processamento em lote |

**Matriz de confusão:**  
Heatmap 6×6 onde linhas = classe real e colunas = classe prevista. Células na diagonal = acertos. Fora da diagonal = erros.

- **Modo absoluto:** mostra o número bruto de pacotes em cada célula.
- **Modo por linha (recall):** normaliza por classe real — mostra a fração de cada classe que foi detectada corretamente. Útil para comparar classes com volumes muito diferentes.

**Principais erros (fora da diagonal):**
- Benigno → MITM Multi (841 pacotes): pacotes benignos que passam pelo relay service `0x100B` são confundidos com ataque MITM Multi, pois compartilham a feature `f21=1`.
- MITM Multi → Benigno (304 pacotes): pacotes do ataque que não usam o relay (f21=0) perdem a feature discriminante.

**Gráfico F1 por classe:**  
Compara o F1 do classificador multiclasse (barras) com o F1 dos classificadores binários especializados (diamantes). Permite verificar se o modelo unificado manteve a qualidade dos modelos individuais.

**Importância das features:**  
Ranking das 13 features por impureza de Gini no XGBoost. Features com maior importância são as que mais separam as classes.

---

### 🔍 Inferência

**O que faz:**  
Classifica pacotes individuais usando o modelo treinado e exibe os resultados visualmente.

**Fontes de dados:**
- `features.csv` existente (amostragem configurável de 1k a 100k linhas)
- Upload de qualquer CSV com as 13 colunas de features

**Saídas:**
- Distribuição das predições (pizza + tabela)
- Distribuição de confiança por classe (box plot) — mostra se o modelo está "seguro" ou "em dúvida" nas suas predições
- Timeline das predições (se o CSV tiver coluna `timestamp`) — scatter plot mostrando onde no tempo cada classe foi detectada
- Download dos resultados em CSV

---

## Arquitetura dos arquivos

```
detection/dashboard/
├── app.py              # Entrada principal — sidebar + roteamento
└── pages/
    ├── p1_overview.py  # Visão geral dos datasets
    ├── p2_mixer.py     # Mixer de datasets
    ├── p3_pipeline.py  # Execução do pipeline
    ├── p4_results.py   # Resultados do multiclasse
    └── p5_inference.py # Inferência por pacote
```

## Dependências

```bash
pip install streamlit plotly pandas numpy xgboost scikit-learn
```
