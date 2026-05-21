# Sistema de Detecção de Intrusão para SOME/IP — Relatório Completo

**Contexto:** Mestrado em Segurança de Veículos Definidos por Software (SDV)  
**Objetivo:** Avaliar a detectabilidade de ataques em redes SOME/IP automotivas via aprendizado de máquina  

---

## Visão Geral

O trabalho é composto por dois eixos complementares, cada um com dataset, pipeline e modelo próprios:

| | **Eixo 1 — Kim et al.** | **Eixo 2 — Multiclasse** |
|---|---|---|
| Dataset | Kim et al. (2026) — 7 PCAPs | Alkhatib et al. / someip_traces (2021) |
| Tarefa | Binário: normal vs. ataque | Multiclasse: 6 tipos |
| Features | 18 (f01–f18) | 13 (f01, f08, f11–f22) |
| Modelo | XGBoost binário | XGBoost multi:softprob |
| Resultado | F1=0,9979 | F1 macro=0,9992 |

---

# PARTE 1 — Reprodução e Extensão de Kim et al. (2026)

**Referência:** Kim, J. et al. "XGBoost-Based Anomaly Detection Framework for SOME/IP in In-Vehicle Networks", 2026.

---

## 1.1 Dataset (Kim et al.)

7 capturas PCAP em ambiente de simulação vSOME/IP:

| PCAP | Tipo | Pacotes (aprox.) |
|---|---|---|
| normal.pcap | Tráfego legítimo | ~4,0 M |
| dos_1.pcap, dos_2.pcap | Flood SOME/IP | ~1,5 M |
| mitm_single.pcap | MITM (1 atacante) | ~2,0 M |
| mitm_multi.pcap | MITM (múltiplos atacantes) | ~2,0 M |
| fuzzy_1.pcap, fuzzy_2.pcap | Fuzzing de interfaces | ~2,5 M |

**Total parseado:** 14.233.354 pacotes → `parsed_packets.csv` (3,46 GB)  
**Split:** 70% treino / 30% teste, corte temporal por cenário  
**Dataset publicado por Kim:** 7.116.674 × 12 features (subconjunto)

---

## 1.2 O que Caracteriza Cada Ataque (Kim dataset)

**DoS — Flood de SOME/IP-SD**  
Inunda o daemon de Service Discovery com mensagens `OfferService` falsas em alta frequência. Todos os pacotes de ataque têm o mesmo payload (zero diversidade). f01 (intervalo entre pacotes) colapsa, f17 (taxa por IP) explode.

**Fuzzy — Probing de Interfaces de Serviço**  
Envia chamadas a métodos aleatórios com payloads variados. Anomalia: f15 (someip_payload_len) atinge 1332 bytes (max do dataset) nos pacotes de SD flood embutidos. **Particularidade deste dataset:** 83% dos pacotes rotulados como "fuzzy" são tráfego TCP/UDP de fundo do PCAP, indistinguíveis do normal — label é PCAP-level, não packet-level.

**MITM (Single e Multi-Attacker)**  
Atacante captura pacotes do IP vítima (normal.pcap) e os reinjecta com IP de origem modificado (mitm.pcap). Detectável via f14 (hash de payload cross-PCAP): o mesmo hash de payload aparece com src_ip diferente entre PCAPs.

---

## 1.3 Protocolo SOME/IP — Comportamento UDP e TCP (Kim dataset)

| Camada | Transporte | Porta | Papel no Ataque |
|---|---|---|---|
| SOME/IP-SD | **UDP** obrigatório | 30490 | DoS flood, MITM na descoberta |
| Serviços (eventos) | UDP preferencial | 30501–30503 | Fuzzy targeting |
| Serviços (req-resp) | TCP quando confiável | 30501–30503 | Fuzzy, MITM payload relay |

**Como o código distingue:** `tl_len = l4.len - 8` (UDP) vs. `len(bytes(l4.payload))` (TCP). O fluxo é rastreado por chave `(src_ip, dst_ip, sport, dport, transport)` — UDP e TCP nunca contaminam o estado um do outro.

---

## 1.4 Modelo de Ameaça (Kim dataset)

**Premissa:** Atacante com acesso à rede Ethernet interna (após comprometimento de ECU via OTA, porta OBD-II ou dispositivo físico conectado). Não possui material criptográfico.

| Ataque | Objetivo | Camada | Evidência no tráfego |
|---|---|---|---|
| DoS | Indisponibilidade do SD | SOME/IP-SD | f17 alto, f18 baixo, f15=1332 B |
| Fuzzy | Descoberta de vulnerabilidades | Serviço (métodos) | f15/f16 variáveis, f08 alto |
| MITM Single | Interceptação / Redireciamento | Rede (IP spoofing) | f14: hash collision cross-PCAP |
| MITM Multi | Interceptação distribuída | Rede (múltiplos IPs) | f14 + múltiplos src_ip atacantes |

---

## 1.5 Pipeline de Treino e Validação (Kim dataset)

```
PCAPs (7 arquivos)
      |
      v
01_parse.py  ─────────────────>  parsed_packets.csv  (14,2M linhas)
      |
      v
03_features.py  ──────────────>  X_train.npy / X_test.npy  (18 features)
                                  train_features.csv / test_features.csv
      |
      v
04_train.py --mode binary  ───>  model_binary.json
      |
      v
05_evaluate.py  ──────────────>  métricas por tipo de ataque
```

**Decisões de projeto:**
- Split temporal 70/30 por cenário (evita vazamento de dados de um ataque no treino de outro)
- Sem balanceamento de classes (classes majoritárias — normal é ~56% do dataset)
- 18 features: 12 de Kim + f13 (repeat rate) + f14 (hash cross-PCAP) + f15/f16 (tamanhos reais) + f17 (taxa por IP) + f18 (diversidade de payload por IP)

---

## 1.6 É Possível Discernir o Tipo de Ataque? (Kim dataset)

**Modo binário (normal vs. ataque) — RESULTADO PROPOSTO (18 features):**

| Métrica | Kim reporta | Reprodução (16 feat.) | **Proposto (18 feat.)** |
|---|---|---|---|
| F1-Score | 0,98 | 0,7435 | **0,9979** |
| Recall | — | 0,5935 | **0,9972** |
| Precision | — | 0,9952 | ~0,999 |
| AUC-ROC | — | 0,9540 | **1,0000** |

**Recall por tipo de ataque (binário, 18 features):**

| Tipo | Recall | Observação |
|---|---|---|
| DoS | **0,9998** | f17 e f01 detectam flood com quase zero erro |
| Fuzzy | **0,9991** | f17 captura todos os pacotes do IP atacante |
| MITM | **0,9947** | f14 detecta relay cross-PCAP |
| Normal (FP) | — | FP = 633 pacotes (~0,01% do normal) |

**Por que Kim (F1=0,98) não é reproduzível com 16 features?**  
O dataset publicado (12 features, 7,1M rows) é um subconjunto processado. Sem f17 e f18 — features de contexto comportamental por IP — o recall do Fuzzy fica em 0,342 (limite fundamental: 83% dos pacotes "fuzzy" são tráfego de fundo indistinguível).

**Modo multi-classe (4 classes) no dataset Kim — NÃO recomendado:**  
Recall normal=0,535, DoS=0,539, MITM=0,642. O label é PCAP de origem, não por pacote. Tráfego de fundo em PCAPs de ataque parece idêntico ao normal → confusão estrutural insolucionável sem re-rotulação.

---

## 1.7 Matriz de Confusão — Kim dataset (binário, 18 features)

```
              Predito Normal   Predito Ataque
Real Normal      TN = ~4,0M      FP = 633
Real Ataque      FN = ~1.700     TP = ~4,2M
```

> FP=633 é artificialmente baixo: o dataset Kim não contém servidores legítimos de alta taxa (streaming, CDN), que poderiam disparar falsos positivos via f17.

---

## 1.8 Feature Discriminante Principal por Ataque (Kim dataset)

| Ataque | Feature-chave | Por quê |
|---|---|---|
| DoS | **f17** src_packet_rate | Atacante tem taxa >> qualquer IP normal |
| Fuzzy | **f17** src_packet_rate | Captura TODO o IP atacante, inclusive tráfego de fundo |
| MITM | **f14** duplicate_source | Mesmo hash de payload com src_ip diferente entre PCAPs |

**Paradoxo de f17:** Importância XGBoost = 3,6% (poucos nós de divisão), mas impacto real é dominante — um único limiar em f17 separa DoS e Fuzzy com recall >0,999. Isso demonstra a limitação da métrica padrão de importância do XGBoost para features de alto ganho e baixa frequência.

**Ressalva para dissertação:** f17 não generaliza se atacantes rotacionam endereços IP. Mencionada como "feature de contexto comportamental" com limitações conhecidas.

---

---

# PARTE 2 — Classificador Multiclasse SOME/IP

**Referência do dataset:** Alkhatib, H., Ghauch, H., & Danger, J.-L. (2021). *Here comes SAID: A SOME/IP Attention-based Intrusion Detection system*. Dataset: someip_traces.

---

## 2.1 Dataset (someip_traces + FakeClientID)

| Classe | Treino | Teste | Total | % |
|---|---|---|---|---|
| Benigno | 768.878 | 329.540 | 1.098.398 | 57,02% |
| DoS | 107.430 | 46.042 | 153.472 | 7,97% |
| Fuzzy | 196.018 | 84.008 | 280.026 | 14,54% |
| MITM_Multi | 149.622 | 64.124 | 213.746 | 11,10% |
| MITM_Single | 126.178 | 54.077 | 180.255 | 9,36% |
| FakeClientID | 463 | 463 | 926 | 0,02% |
| **Total** | **1.348.589** | **578.254** | **1.926.843** | |

> FakeClientID é extensão ao dataset original de Alkhatib et al. — adicionado como contribuição nova.

---

## 2.2 O que Caracteriza Cada Ataque (someip_traces)

### Assinatura de Features (médias reais — conjunto de treino)

| Feature | Benigno | DoS | Fuzzy | MITM_Multi | MITM_Single | FakeClientID |
|---|---|---|---|---|---|---|
| f13 repeat rate | 0,46 | **0,70** | 0,68 | 0,39 | **0,73** | ~0 |
| f17 pkt/s | 216 | 206 | **323** | 154 | 260 | 2,2 |
| f18 payload diversity | 0,07 | 0,007 | **0,18** | 0,015 | 0,07 | **0,97** |
| f19 is_SD | 0,009 | **0,60** | 0,022 | 0,019 | **0,85** | 0 |
| f20 service diversity | 1,55 | 2,00 | 1,22 | 1,79 | 1,49 | **2,97** |
| f21 relay service | ~0 | ~0 | ~0 | **0,38** | ~0 | ~0 |
| f22 clientid diversity | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | **7,95** |
| f08 payload change | 0,005 | 0,006 | **0,081** | ~0 | 0,022 | 0,360 |
| f11 ip len change (B) | 0,02 | 0,69 | **3,48** | 0,11 | 0,11 | 6,89 |

**DoS:** 60% dos pacotes são SOME/IP-SD (f19=0,60) — flood de `OfferService`/`FindService` em UDP. Payloads idênticos repetidos (f18=0,007, f13=0,70).

**Fuzzy:** Maior taxa de pacotes (323 pkt/s). Chamadas a métodos aleatórios com payloads variados (f08=0,081, f18=0,18). Não ataca o SD (f19=0,02).

**MITM Multi-Attacker:** Único ataque que usa o relay service (f21=0,38 → 38% dos pacotes têm `service_id=0x100B`). Payload passa quase inalterado (f08~0).

**MITM Single-Attacker:** 85% das mensagens são SD (f19=0,85) — intercepção na fase de descoberta. Maior repeat rate (f13=0,73) — replay de mensagens SD.

**FakeClientID:** f22=7,95 (7–8 client_ids distintos por IP). Taxa muito baixa (2,2 pkt/s) — ataque lento e furtivo. Sem f22, o padrão é indistinguível do benigno.

---

## 2.3 Protocolo SOME/IP — Comportamento UDP e TCP (someip_traces)

| Função | Transporte | Porta | Evidência no modelo |
|---|---|---|---|
| Service Discovery | **UDP** obrigatório | 30490 | f19=1 → sempre UDP |
| Eventos / Notificações | UDP preferencial | 30490–30503 | Predominante no benigno |
| Request-Response | UDP ou TCP | 30490–30503 | Fluxos TCP rastreados separadamente |
| Relay (0x100B) | UDP ou TCP | variável | f21=1 no MITM_Multi |

**Qual transporte cada ataque usa** (derivado de f19 e f21):

| Ataque | Transporte | Evidência |
|---|---|---|
| DoS | **UDP** | f19=0,60 (flood SD, sempre UDP) |
| Fuzzy | **UDP** | ataque de serviço, sem overhead de handshake |
| MITM_Multi | UDP + TCP | f21=0,38 (relay pode usar ambos) |
| MITM_Single | **UDP** | f19=0,85 (ataca SD, sempre UDP) |
| FakeClientID | **UDP** | f19=0 (serviço, não SD) |

---

## 2.4 Modelo de Ameaça (someip_traces)

| Ataque | Objetivo | Camada SOME/IP | Furtividade | Feature-chave |
|---|---|---|---|---|
| DoS | Disponibilidade | SOME/IP-SD (flood) | Baixa | f19 + f18 |
| Fuzzy | Descoberta de vuln. | Serviço (métodos) | Baixa | f08 + f17 |
| MITM Multi | Integridade/Conf. | Relay (0x100B) | Média | f21 |
| MITM Single | Integridade/Conf. | SOME/IP-SD | Média | f19 + f13 |
| FakeClientID | Impersonação | Serviço (client_id) | **Alta** | f22 |

---

## 2.5 Pipeline de Treino e Validação (someip_traces)

```
PCAPs (someip_traces)       Labels (CSVs Alkhatib)
        |                          |
        v                          v
fake_client_id/01_features.py ──────────────>  fake_client_id/data/features.csv
        |
        v
multiclass/01b_merge_fakeclientid.py ────────>  X_train.npy / X_test.npy (13 feat.)
                                                 norm_params.json
        |
        v
multiclass/02_train.py ──────────────────────>  multiclass_classifier.json
        |
        v
multiclass/03_test_outofscope.py ────────────>  outofscope_results.json
```

**Decisões de projeto:**
- Split 70/30 estratificado por classe (seed=42)
- Normalização min-max com `norm_params.json` persistido para inferência
- Pesos inversamente proporcionais à frequência, **com cap 100x** (FakeClientID seria 700x sem o cap)
- Features stateful: janela deslizante por fluxo `(src,dst,sport,dport,transport)`
- f22 filtrada por `msg_type < 0x80` (apenas REQUEST — servidores não rotacionam client_ids)

---

## 2.6 É Possível Discernir o Tipo de Ataque? (someip_traces)

**Sim.** Métricas globais:

| Métrica | Valor |
|---|---|
| Acurácia global | **99,91%** |
| F1 macro | **99,92%** |
| F1 weighted | **99,91%** |
| Latência por pacote | 0,79 ms |
| Throughput | 225.813 pkt/s |

**F1 por classe — comparação com detectores binários especializados (One-vs-Rest):**

| Classe | F1 Multiclasse | F1 One-vs-Rest | Delta |
|---|---|---|---|
| Benigno | 0,9992 | — | — |
| DoS | **0,9999** | 0,9998 | +0,0001 |
| Fuzzy | 0,9995 | 0,9990 | +0,0005 |
| MITM_Multi | 0,9973 | 0,9979 | -0,0006 |
| MITM_Single | 0,9991 | 0,9994 | -0,0003 |
| FakeClientID | **1,0000** | — | — |

> O classificador único equivale a cinco detectores especializados simultaneamente.

---

## 2.7 Matriz de Confusão (someip_traces)

```
               Benigno      DoS    Fuzzy  MITM_Mu  MITM_Si  FakeC
Benigno      1.097.219        8      111      841      219      0
DoS                  0  153.465        2        4        1      0
Fuzzy              126        7  279.889        3        1      0
MITM_Multi         304        8        2  213.432        1      0
MITM_Single         78        0        1        7  180.169      0
FakeClientID         0        0        0        0        0    463
```

**Análise dos erros:**

- **Benigno → MITM_Multi (841):** Pacotes benignos que passam pelo relay service (0x100B) elevam f21=1, coincidindo com a assinatura do MITM_Multi. Falso positivo estrutural — inerente à feature.
- **Benigno → MITM_Single (219):** Pacotes SD benignos com alta repetição ficam no limiar da classe MITM_Single (ambos têm f19 alto + f13 alto).
- **FakeClientID: 0 erros.** f22 é discriminante perfeita — separação absoluta.

---

## 2.8 Resultados Out-of-Scope (ataques não treinados)

| Cenário | SOME/IP pkts | Benigno | Classificação dominante | Interpretação |
|---|---|---|---|---|
| Error on Error | 90.871 | **99,98%** | Benigno | Violação de estado SD invisível às features de fluxo |
| Error on Event | 1.638 | **99,94%** | Benigno | Idem |
| Delete Request | 2.228 | **99,91%** | Benigno | Manipulação pontual de SD |
| Delete Response | 2.423 | **99,88%** | Benigno | Idem |
| Wrong Interface | 2.183 | 66,3% | **34% FakeClientID** | Usa client_ids variados → f22>1 |
| Wrong Interface 2 | 1.615 | 90,6% | **9,4% FakeClientID** | Idem, intensidade menor |
| Delete Request Test | 1.363 | **99,93%** | Benigno | SD manipulation |

> **Wrong Interface → FakeClientID é semanticamente correto:** o ataque usa interfaces de serviço incorretas com client_ids variados, compartilhando a assinatura real de FakeClientID.

---

---

# PARTE 3 — Comparação e Limitações

## 3.1 Comparação entre os Dois Eixos

| | Kim et al. (Eixo 1) | Multiclasse (Eixo 2) |
|---|---|---|
| Tipo de detecção | Binário (ataque/normal) | Multiclasse (6 tipos) |
| Dataset | Kim 2026 (7 PCAPs) | Alkhatib 2021 (someip_traces) |
| Total de amostras | ~14,2 M pacotes | ~1,9 M amostras válidas |
| Features | 18 (inclui contexto por IP) | 13 (stateful por fluxo) |
| F1 | 0,9979 | 0,9992 (macro) |
| Identifica o tipo? | Não (binário) | Sim (6 classes) |
| Ataque mais difícil | Fuzzy (label PCAP-level) | MITM_Multi (relay sobreposição) |

## 3.2 Limitações Identificadas

**Eixo 1 (Kim):**
- F1=0,98 reportado por Kim não é reproduzível com o dataset publicado (12 features)
- f17 não generaliza para atacantes com IP rotativo
- FP=633 subestimado — dataset sem servidores legítimos de alta taxa

**Eixo 2 (Multiclasse):**
- Ataques de manipulação de estado de protocolo (Error on Error, Delete Request) são invisíveis às features statísticas de fluxo — exigem modelagem de máquina de estados SD
- MITM_Multi tem 304 falsos negativos estruturais (benigno via relay confunde com ataque)
- FakeClientID com 463 amostras de treino — generalização limitada

## 3.3 Extensões Naturais

1. Camada de verificação de sequência de mensagens SOME/IP-SD (máquina de estados) para cobrir ataques out-of-scope do Eixo 2
2. f19 como alternativa a f17 para ambientes com IP rotativo (comportamento SD é menos afetado por rotação)
3. Avaliar o modelo multiclasse no dataset Kim (e vice-versa) para medir transferibilidade

---

## Referências dos Datasets

> **Eixo 1:** Kim, J. et al. "XGBoost-Based Anomaly Detection Framework for SOME/IP in In-Vehicle Networks", 2026. Dataset disponível publicamente.

> **Eixo 2:** Alkhatib, H., Ghauch, H., & Danger, J.-L. "Here comes SAID: A SOME/IP Attention-based Intrusion Detection system", 2021. Dataset: someip_traces.
