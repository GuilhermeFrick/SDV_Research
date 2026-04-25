01_parse_pcap.py — Seção 5.1 do artigo

Lê os 7 PCAPs do Figshare com Scapy
Faz parsing em camadas (Ethernet → IP → TCP/UDP → SOME/IP)
Distingue SOME/IP normal de SOME/IP-SD (Service Discovery)
Rotula cada pacote automaticamente pelo nome do arquivo
Saída: CSV com ~14M linhas


02_extract_features.py — Seções 5.2 e 5.3

Implementa as 9 features da Tabela 1 com as equações exatas do artigo (Eq. 1–8)
Modelo de distribuição de bytes com Laplace smoothing (Eq. 2–3)
Log-likelihood (Eq. 5), cross-entropy (Eq. 6), Hamming distance (Eq. 7)
Features calculadas por fluxo (five-tuple), como descrito na Seção 5.2
Normalização Min-Max (Eq. 8) com parâmetros calculados só no treino
Split estratificado 50/50 fiel ao artigo

03_train_evaluate.py — Seções 5.4 e 6

XGBoost com os exatos hiperparâmetros da Tabela 2 (lr=0.05, 1000 árvores, depth=6, λ=1)
CTGAN com embedding=128, hidden=256, batch=500, 100 épocas
Otimização de threshold por F1 (artigo encontra 0.36)
Avaliação nos dois cenários: imbalanceado (realista) e balanceado (downsampling)
Reproduz a Figura 10 (ROC, PR, F1×threshold, DET)
Comparação com 10 baselines (RF, DT, LGB, KNN, LR, NB...)

Para começar:
bashpip install -r requirements.txt

# Baixe os PCAPs do Figshare → coloque em data/pcap/
python 01_parse_pcap.py
python 02_extract_features.py
python 03_train_evaluate.py --no-ctgan --no-baselines  # teste rápido

