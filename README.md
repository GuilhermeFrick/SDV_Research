# Software-Defined Vehicle (SDV) Security Research

Este repositório centraliza os materiais utilizados na pesquisa sobre **detecção de anomalias em redes automotivas baseadas em SOME/IP no contexto de Software-Defined Vehicles (SDV)**.

---

## Objetivo

Organizar e consolidar os artefatos que suportam o estudo e evolução da pesquisa, incluindo:

- Artigos científicos (papers)
- Experimentos e testes
- Protótipos e códigos
- Análises e extração de dados
- Documentações técnicas
- Notas de estudo e insights

Este repositório serve como base prática complementar à wiki do projeto, onde o conteúdo teórico e estruturado é desenvolvido.

---

## Configuração do Ambiente

### Pré-requisitos

- Python 3.x (recomendado via [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Git + Git LFS (necessário para baixar o dataset)

### 1. Instalar Git e Git LFS

**Linux (Mint/Ubuntu):**
```bash
sudo apt update
sudo apt install git git-lfs -y
git lfs install
```

**Windows:**
```powershell
# Git LFS já incluso no instalador do Git para Windows
git lfs install
```

### 2. Clonar o repositório

```bash
git clone https://github.com/GuilhermeFrick/SDV_Research.git
cd SDV_Research
```

> O dataset (`data/dataset_ism_xgboost.rar`, 356 MB) é baixado automaticamente via Git LFS durante o clone.

### 3. Instalar dependências Python

```bash
pip install scapy pandas numpy xgboost scikit-learn lightgbm matplotlib tqdm ctgan
```

Ou via requirements:
```bash
pip install -r experiments/files/requirements.txt
```

### 4. Extrair o dataset

**Linux:**
```bash
sudo apt install unrar -y
unrar x data/dataset_ism_xgboost.rar data/
```

**Windows:**
```powershell
# Com WinRAR ou 7-Zip instalado
# Extrair data/dataset_ism_xgboost.rar para data/
```

### 5. Verificar instalação

```bash
ls data/dataset_ism_xgboost/           # 7 arquivos .pcap
ls data/dataset_ism_xgboost/tr_te_sets/ # X_train.npy, X_test.npy, y_train.npy, y_test.npy
python -c "import scapy, xgboost, pandas, numpy; print('ok')"
```

---

## Pipeline de Reprodução (Kim et al. 2026)

Documentação completa em [`experiments/docs/README.md`](experiments/docs/README.md).

```bash
# Etapa 1 — Parsing dos PCAPs (~9 min)
python experiments/files/01_parse_pcap.py \
  --pcap-dir data/dataset_ism_xgboost \
  --output   data/parsed_packets.csv

# Etapa 2 — Extração de features
python experiments/files/02_extract_features.py

# Etapa 3 — Treinamento e avaliação
python experiments/files/03_train_evaluate.py
```

---

## Estrutura do Repositório

```bash
.
├── papers/           # Artigos científicos utilizados na pesquisa
├── experiments/      # Scripts e cenários de teste (SOME/IP, ataques, etc)
│   ├── files/        # Pipeline de reprodução Kim et al. (2026)
│   └── docs/         # Documentação detalhada de cada etapa
├── data/             # Dataset comprimido (via Git LFS) e outputs do pipeline
├── models/           # Modelos de Machine Learning (ex: XGBoost)
├── docs/             # Documentação técnica complementar