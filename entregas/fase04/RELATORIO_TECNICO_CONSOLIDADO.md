# Relatorio Tecnico - Fase 4
## Tarefa 1 - Analise de Videos de Reabilitacao com Estimativa de Pose

**FIAP - Inteligencia Artificial para Desenvolvedores**  
**Grupo 69**

> O classificador e um prototipo experimental para pesquisa/ensino, nao uma
> ferramenta clinica. Resultados devem ser interpretados dentro das limitacoes
> do dataset e da separacao por participantes descritas abaixo.

## 1. Objetivo

Implementar um fluxo para analisar videos de exercicios de reabilitacao,
estimar postura/movimento e classificar repeticoes como corretas ou incorretas.
A API e a interface apresentam o resultado como experimental, sem emitir
diagnosticos ou substituir a avaliacao de um fisioterapeuta.

## 2. Dataset selecionado

O REHAB24-6, publicado no [Zenodo](https://zenodo.org/records/13305826), e
diretamente relacionado ao objetivo: disponibiliza videos RGB de duas cameras,
sequencias de articulacoes 2D/3D e marcadores de captura de movimento. Suas
anotacoes incluem 1.072 repeticoes, intervalos por frame, rotulos binarios de
execucao correta/incorreta, direcao do exercicio e condicoes de iluminacao.

O conjunto cobre seis exercicios: abducao de braco, movimento de bracos em V/W,
flexoes apoiadas em mesa, abducao de perna, afundo e agachamento. Os videos
foram gravados a 30 FPS; as anotacoes de segmentacao sao baseadas nessa taxa.
Participaram dez voluntarios saudaveis, com 65 gravacoes no total.

### Adequacao e limites

- **Videos de exercicios:** adequado para prototipar o fluxo de analise de
  movimento em fisioterapia; os participantes sao voluntarios saudaveis.
- **Pose e movimento:** as articulacoes 2D/3D permitem avaliar estimadores como
  OpenPose e comparar resultados com uma referencia de movimento.
- **Repeticoes incorretas:** ha rotulo correto/incorreto por repeticao, mas a
  descricao do dataset nao garante anotacao detalhada da categoria ou causa de
  cada erro.
- **YOLOv8:** o dataset pode servir como entrada visual, mas a documentacao nao
  informa caixas delimitadoras de objetos/areas criticas; nao e suficiente, por
  si so, para treinar e avaliar detector supervisionado de objetos.
- **Generalizacao clinica:** participantes saudaveis e ambiente controlado nao
  representam pacientes em tratamento; os resultados nao devem ser extrapolados
  para pacientes em recuperacao.
- **Identidade e licenca:** uso restrito a pesquisa academica ou nao comercial,
  conforme o registro Zenodo; nao redistribuir os arquivos e seguir a citacao
  indicada.

## 3. Organizacao no repositorio

```text
entregas/fase04/
├── data/
│   ├── README.md
│   └── raw/rehab24-6/       <- arquivos Zenodo, excluidos do Git
├── src/train_classifier.py  <- treinamento e avaliacao
├── results/                 <- relatorio de metricas gerado pelo treino
├── download_dataset.py      <- download e verificacao de checksums
├── README.md
└── RELATORIO_TECNICO_CONSOLIDADO.md
```

O pipeline reutilizavel esta em `backend/rehab_video/`; a rota de upload esta
em `backend/main.py` e a tela Streamlit em `frontend/app.py`. O modelo treinado
fica em `backend/model/rehab_pose_classifier.joblib`; videos e pesos YOLO sao
artefatos locais excluidos do Git.

## 4. Fluxo experimental implementado

1. Validar `Segmentation.csv` e os videos Camera17 a 30 FPS. O CSV usa intervalos
   inclusivos e numeracao de frames a partir de 1; a leitura converte para
   indices zero-based do OpenCV. Uma anotacao termina um frame apos o video e e
   limitada ao ultimo frame disponivel.
2. Separar participantes em treino, validacao e teste, sem sobreposicao entre
   particoes.
3. Amostrar 12 frames de cada repeticao, estimar 17 pontos de pose COCO com
   YOLOv8n-pose pre-treinado e derivar caracteristicas temporais normalizadas.
4. Treinar regressao logistica balanceada e comparar no conjunto de teste com
   um baseline de classe majoritaria (`DummyClassifier`).
5. Salvar metricas agregadas, por classe e por exercicio em
   `results/training_report.json`.
6. Expor inferencia via `POST /analyze-rehabilitation-video` e pela tela
   Streamlit. Upload limitado a 250 MiB por padrao; a API aceita MP4 e frame
   range zero-based.

Como os ZIPs de articulacoes 2D/3D nao estao disponiveis localmente, nao foi
possivel comparar a pose estimada com ground truth. O baseline e somente a
classe majoritaria, nao uma referencia de pose.

## 5. Estado da entrega

- [x] Definicao do escopo de video de reabilitacao.
- [x] Selecao e avaliacao de adequacao do REHAB24-6.
- [x] Dados: anotacoes e 130 videos extraidos em `Ex1` a `Ex6`.
- [x] Pipeline de pose, treinamento, avaliacao e testes automatizados.
- [x] Integracao com API FastAPI e interface Streamlit.
- [x] Treinamento integral e consolidacao das metricas no conjunto de teste.
- [x] Smoke test de inferencia real pela rota FastAPI.
- [ ] Comparacao com ground truth de pose: pendente dos ZIPs de articulacoes.

Os 77 testes automatizados da Fase 4 passaram (52 de audio e 25 de video). O
experimento utilizou os 65
videos da Camera17, com 1.072 repeticoes anotadas. A divisao por participante
resultou em 600 repeticoes para treino (participantes 3, 4, 5, 7, 8 e 10), 250
para validacao (1 e 9) e 222 para teste (2 e 6), sem participantes compartilhados.
O ambiente nao tinha CUDA; a extracao completa foi executada na CPU.

## 6. Resultados experimentais

| Conjunto | Acuracia | F1 macro | Repeticoes |
|---|---:|---:|---:|
| Validacao | 63,20% | 62,23% | 250 |
| Teste | 59,46% | 57,80% | 222 |
| Baseline majoritario (teste) | 43,69% | 30,41% | 222 |

No teste, a matriz de confusao (linhas reais e colunas previstas na ordem
incorreta/correta) foi `[[44, 53], [37, 88]]`. Para repeticoes incorretas, a
precisao foi 54,32%, o recall 45,36% e o F1 49,44%; para corretas, 62,41%,
70,40% e 66,17%, respectivamente. O resultado supera o baseline simples, mas
nao e suficiente para uso clinico.

| Exercicio | Repeticoes de teste | Acuracia | F1 macro |
|---|---:|---:|---:|
| 1 - abducao de braco | 30 | 70,00% | 69,97% |
| 2 - movimentos V/W | 46 | 52,17% | 52,08% |
| 3 - flexao apoiada | 18 | 55,56% | 55,56% |
| 4 - abducao de perna | 45 | 62,22% | 47,20% |
| 5 - afundo | 40 | 42,50% | 39,43% |
| 6 - agachamento | 43 | 74,42% | 71,39% |

Os resultados variam por exercicio; em particular, o recall de repeticoes
incorretas e baixo no exercicio 4 e o desempenho geral e fraco no exercicio 5.
As metricas completas estao em
[`results/training_report.json`](results/training_report.json).

## 7. Tarefa 2 - Analise de Audio em Reabilitacao Fisioterapeutica

Objetivo: transcrever a interacao fisioterapeuta-paciente em sessoes de
reabilitacao, identificar termos criticos do exercicio e sinalizar indicios
vocais de cansaco, desconforto, dificuldade respiratoria ou disartria. O
resultado deve ser tratado como apoio experimental e sempre exigir revisao do
fisioterapeuta.

### Estado por item da tarefa

| Item da tarefa | Estado |
|---|---|
| Processar audios de consultas medicas | Implementado para sessao de reabilitacao, nao verificado contra o Azure |
| Detectar alteracoes vocais indicativas | Implementado (inclui disartria); sinal fraco em base de voz rotulada (ROC-AUC 0,61) |
| Azure Speech to Text | Integrado no codigo; nao executado com credenciais reais |
| Termos criticos e sentimento com Azure Text Analytics | Integrado no codigo; nao executado com credenciais reais |

**1. Processar audios de consultas medicas - implementado em escopo restrito.**
- A rota `POST /analyze-rehabilitation-audio` (`backend/main.py`) aceita apenas
  `.wav` (PCM 16 kHz mono, formato exigido pelo endpoint REST de audio curto do
  Azure), valida formato e tamanho (25 MiB por padrao), processa em threadpool e
  remove os arquivos temporarios.
- Limite: 60 s por audio (`MAX_REHAB_AUDIO_SECONDS`), imposto pelo endpoint REST
  de audio curto do Azure. Sessoes completas exigiriam a API de transcricao em
  lote, que nao foi implementada.
- Escopo atendido: sessoes de fisioterapia. Consultas medicas genericas e
  cirurgias nao sao atendidas.
- Nao verificado: nao houve execucao contra o Azure real; os testes cobrem a rota
  com o Azure simulado.

**2. Detectar alteracoes vocais indicativas de condicoes medicas - implementado;
sinal fraco para alteracao vocal e conservador para disartria.**
- Implementado em `backend/rehab_audio/acoustics.py`: 11 caracteristicas acusticas
  - 7 temporais/energeticas (energia RMS, proporcao de voz, proporcao de pausas,
  duracao media de pausa, taxa de fala e variabilidade de pitch) e 4 espectrais
  (centroide espectral, rolloff, fluxo espectral e taxa de cruzamento por zero) -
  e 3 sinalizadores heuristicos com `measured`/`expected`: `possivel_fadiga`,
  `possivel_desconforto_respiratorio` e `fala_instavel`.
- Marcadores de voz calculados via Praat (`praat-parselmouth`) no mesmo modulo:
  `jitter_local`, `shimmer_local` e `hnr`. Eles alimentam o sinalizador
  `possivel_disartria` (regra em `backend/rehab_audio/dysarthria.py`): jitter
  >= 1,04% ou shimmer >= 3,81% ou HNR <= 20 dB (faixa de normofonia), exigindo ao
  menos 2 marcadores fora da faixa.
- Avaliacao offline no ICBHI 2017 com 920 gravacoes (633 com crackles ou wheezes
  e 287 sem), executada por `entregas/fase04/src/evaluate_audio_flags.py`:

| Previsor | VP | FP | Precisao | Recall | F1 | Acuracia |
|---|---:|---:|---:|---:|---:|---:|
| `possivel_fadiga` | 0 | 1 | 0,0% | 0,0% | 0,0% | 31,1% |
| `possivel_desconforto_respiratorio` | 1 | 1 | 50,0% | 0,2% | 0,3% | 31,2% |
| `fala_instavel` | 79 | 41 | 65,8% | 12,5% | 21,0% | 35,3% |
| `qualquer_alerta` | 80 | 42 | 65,6% | 12,6% | 21,2% | 35,3% |

- O baseline "sempre normal" atinge 31,2% de acuracia; portanto `qualquer_alerta`
  (35,3%) fica praticamente no nivel trivial. Em 798 das 920 gravacoes nenhum
  alerta foi acionado.
- Um baseline supervisionado (as mesmas caracteristicas acusticas, validacao
  cruzada agrupada por sujeito) obteve ROC-AUC 0,580 e PR-AUC (average precision)
  0,750, ainda proximo do acaso.
- Sonda de embeddings profundos (`--include-embeddings`), executada por custo de
  CPU em uma subamostra limitada de 120 gravacoes (a mesma subamostra para as
  duas representacoes): a sonda linear sobre a media temporal do
  `facebook/wav2vec2-base` (768 dimensoes) obtem ROC-AUC 0,679 e PR-AUC 0,793,
  contra ROC-AUC 0,654 e PR-AUC 0,755 do baseline manual nessa subamostra. As
  duas representacoes ficam proximas (sem ganho claro) e a subamostra nao
  reproduz o ROC-AUC 0,580 da base completa; o numero serve como comparacao de
  representacoes, nao como metrica da base.
- Conclusao: os sinalizadores nao demonstram sinal mensuravel de anormalidade
  respiratoria nesta base.
- Nao avaliado: cansaco. Nao existe base publica com rotulo de cansaco em sessoes
  de reabilitacao, portanto esse sinalizador permanece sem validacao.

Validacao em base de voz rotulada (VOICED):

- O VOICED (PhysioNet) reune 208 gravacoes de vogal sustentada 'a', com 57 vozes
  saudaveis e 151 com diagnostico vocal verificado clinicamente. E a base
  rotulada de voz mais proxima da afirmacao "alteracoes vocais".
- Sinalizadores heuristicos: `qualquer_alerta` com precisao 90,0%, recall 6,0%,
  F1 11,2% e acuracia 31,2%; em 198 das 208 gravacoes nenhum alerta foi
  acionado. `possivel_fadiga` nao acionou nenhum verdadeiro positivo,
  `possivel_desconforto_respiratorio` acertou 4 e `fala_instavel` acertou 6.
- Baseline supervisionado (as mesmas caracteristicas acusticas, validacao cruzada
  agrupada por sujeito): ROC-AUC 0,615 e PR-AUC 0,816, apenas acima do acaso. As
  4 caracteristicas espectrais adicionadas elevam o ROC-AUC de 0,600 para 0,615.
- Sonda de embeddings profundos (`--include-embeddings`): uma regressao logistica
  sobre a media temporal do `facebook/wav2vec2-base` (768 dimensoes, 16 kHz)
  atinge ROC-AUC 0,580, PR-AUC 0,812, F1 73,1% e recall 72,9%, na mesma validacao
  cruzada agrupada por sujeito. Fica abaixo do baseline manual (ROC-AUC 0,615):
  a representacao self-supervised generica, treinada em fala espontanea, nao
  melhora o sinal na vogal sustentada.
- Limitacao: vogal sustentada nao tem pausas nem onsets de fala, entao os
  sinalizadores baseados em pausa nao podem ser exercitados nessa base.
- Conclusao: assim como no ICBHI, as caracteristicas acusticas carregam no
  maximo sinal fraco; o sinalizador de cansaco segue sem validacao. Metricas em
  [`results/voice_flags_evaluation.json`](results/voice_flags_evaluation.json).

Validacao do sinalizador de disartria (UCI Parkinson's):

- A base aberta UCI Parkinson's Disease (CC BY 4.0) reune 195 medidas de voz
  (vogal sustentada) de 32 sujeitos, sendo 147 rotulados com doenca de Parkinson
  e 48 saudaveis. E a base aberta mais proxima do construto "disartria".
- Sinalizador `possivel_disartria` (ao menos 2 marcadores fora da faixa): VP 39,
  FP 2, precisao 95,1%, recall 26,5%, F1 41,5% e acuracia 43,6%. O sinalizador e
  conservador: quando dispara, quase sempre corresponde a um caso positivo.
- Calibracao de limiares: quando os cortes de normofonia sao recalculados a partir
  da subamostra saudavel (percentis 5/95 de jitter, shimmer e HNR) em vez de usar
  as normas fixas de Praat/MDVP, o F1 sobe de 41,5% para 58,5% com especificidade
  93,8%. A calibracao usa o proprio subconjunto saudavel da base (nao e uma
  validacao independente), mas mostra que a escolha de limiar e decisiva.
- Baseline supervisionado (jitter/shimmer/HNR, validacao cruzada agrupada por
  sujeito): ROC-AUC 0,651, PR-AUC 0,866, F1 80,2% e acuracia 68,2%.
- Limitacao: a base entrega medidas MDVP precomputadas, e nao audio bruto, entao
  apenas jitter/shimmer/HNR sao reconstruidos; a disartria hipocinetica de
  Parkinson e proxy do sinalizador, nao ground truth de sessao de fisioterapia.
  Metricas em
  [`results/dysarthria_evaluation.json`](results/dysarthria_evaluation.json).

Generalizacao entre bases (VOICED x ICBHI):

- O harness tambem avalia transferencia entre dominios (`--dataset cross`): treina
  o modelo em uma base e mede o desempenho na outra. Nos dois sentidos o ROC-AUC
  fica proximo ou abaixo do acaso (0,499 no sentido VOICED -> ICBHI e 0,444 no
  inverso). Ou seja, as caracteristicas acusticas nao generalizam entre vogal
  sustentada e som respiratorio, o que reforca que o sinal observado e especifico
  do dominio, nao uma capacidade clinica. Metricas em
  [`results/audio_cross_evaluation.json`](results/audio_cross_evaluation.json).

Metodologia de avaliacao:

- As metricas de regra e de modelo incluem precisao, recall (sensibilidade),
  especificidade, F1, acuracia, ROC-AUC e PR-AUC. O F1, o ROC-AUC e o PR-AUC vem
  com intervalo de confianca de 95% por bootstrap percentil (1000 reamostragens
  da gravacao).
- O PR-AUC (average precision) e reportado porque as classes sao desbalanceadas;
  a acuracia fica dominada pela classe majoritaria e engana nesse cenario.
- Cada base mantem baselines triviais (`sempre_normal`, `sempre_alerta`) para
  enquadrar os numeros.
- O modo `--include-embeddings` acrescenta um segundo baseline supervisionado:
  uma sonda linear (`StandardScaler + LogisticRegression`) sobre a media temporal
  dos estados ocultos do `facebook/wav2vec2-base` (768 dimensoes, audio
  resampleado para 16 kHz). Ela usa a mesma validacao cruzada agrupada por sujeito
  do baseline manual, isolando o efeito de trocar caracteristicas desenhadas a mao
  por uma representacao self-supervised. O checkpoint e generico (fala
  espontanea/leitura), entao a sonda compara representacoes, nao constitui
  classificador validado clinicamente.

### Componente reutilizavel (modelo persistido)

- No padrão do video (`backend/rehab_video`), o mesmo harness passa a produzir um
  artefato reutilizavel: `evaluate_audio_flags.py --dataset voiced` treina e
  persiste `backend/model/voice_alteration_classifier.joblib`
  (`HistGradientBoostingClassifier` sobre as 11 caracteristicas acusticas) com a
  versao das caracteristicas e as metricas de validacao gravadas no proprio
  artefato.
- `backend/rehab_audio/voice_model.py` carrega o artefato uma vez
  (`lru_cache`), valida a versao (`FEATURE_VERSION`, erro explicito em caso de
  incompatibilidade) e produz o campo `voice_alteration` do relatorio
  (`probability_pathological_voice`, `flagged`, `validation_roc_auc`, ...).
- O modelo e opcional: sem o artefato o campo vai `null` e a analise segue apenas
  com as regras heuristicas; com o artefato, o resultado aparece na resposta da
  API e na interface Streamlit.
- Ressalva: o ROC-AUC de validacao e 0,61; o componente e apoio experimental, nao
  um classificador clinico.

**3. Utilizar Azure Speech to Text - integrado, nao verificado.**
- `backend/rehab_audio/transcription.py` chama o endpoint REST de fala para texto
  com chave e regiao, `language=pt-BR`, saida detalhada e tempos por palavra. O
  endpoint de audio curto usado limita cada arquivo a 60 segundos, sem
  diarizacao.
- Nao verificado: nenhuma transcricao real foi produzida; faltam
  `AZURE_SPEECH_KEY` e `AZURE_SPEECH_REGION`.

**4. Identificar termos criticos e sentimentos com Azure Text Analytics -
integrado, nao verificado.**
- `backend/rehab_audio/text_analysis.py` chama o Azure AI Language (denominacao
  atual do Text Analytics) para sentimento, frases-chave e entidades. Os tres
  resultados aparecem no relatorio como `sentiment`, `azure_key_phrases` e
  `azure_entities`.
- Termos criticos: extraidos por dicionario local em `backend/rehab_audio/terms.py`,
  com as categorias `sintoma`, `esforco`, `escalada` e `instrucao`, por
  correspondencia de palavra inteira.
- Nao verificado: faltam `AZURE_LANGUAGE_KEY` e `AZURE_LANGUAGE_ENDPOINT`.

### Adequacao dos datasets avaliados

- **PhysioNet:** aderencia parcial. Foi a origem da base efetivamente usada para
  validar o construto de alteracao vocal (VOICED, 208 gravacoes rotuladas).
  Contem ainda Bridge2AI-Voice, mas nenhuma dessas bases representa sessoes de
  fisioterapia.
- **AudioSet:** aderencia parcial apenas como apoio, com classes genericas como
  `speech`, `breathing` e `cough`. Clips curtos e sem contexto clinico.
- **Coswara:** opcao auxiliar para respiracao, tosse, voz e contagem. Nao foi
  utilizada nesta entrega: o pacote publicado tem 13 GB e o rotulo disponivel e
  status de COVID-19, nao cansaco ou desconforto.
- **ICBHI 2017:** opcao auxiliar para sons respiratorios anormais; usado na
  primeira avaliacao acima.
- **VOICED:** base de voz com rotulo clinico saudavel/patologico; usada na
  segunda avaliacao e no treino do componente reutilizavel.
- **UCI Parkinson's Disease:** base aberta de medidas vocais (jitter, shimmer,
  HNR) com rotulo saudavel/Parkinson; usada para exercitar o sinalizador de
  disartria.

### Confundimento identificado no ICBHI

- Todas as 35 gravacoes dos sujeitos rotulados como saudaveis usaram o mesmo
  dispositivo (`Meditron`); os outros tres dispositivos nao tem gravacao
  saudavel, e o nivel medio de sinal difere muito entre eles.
- Com isso, o rotulo de diagnostico e previsivel a partir das caracteristicas
  acusticas por reconhecimento do equipamento, e nao por patologia: o baseline
  "sempre anormal" ja atinge 96,2% de acuracia, igual ao modelo supervisionado.
- Esse resultado secundario nao foi tratado como desempenho clinico e esta
  registrado no bloco `confounds` de
  [`results/audio_flags_evaluation.json`](results/audio_flags_evaluation.json).

### Estrategia recomendada

- **Base principal:** conjunto proprio de audios de sessoes de reabilitacao, com
  instrucoes do fisioterapeuta e respostas do paciente. E a unica forma de validar
  os sinalizadores vocais, pois nenhuma base publica tem esse rotulo.
- **Apoio respiratorio:** ICBHI 2017 para eventos respiratorios anormais.
- **Apoio acustico:** Coswara, se houver justificativa de custo e armazenamento.

### Verificacao

```powershell
python -m pytest tests\fase04 -q
python entregas\fase04\download_audio_dataset.py --dataset both
python entregas\fase04\download_audio_dataset.py --dataset parkinson
python entregas\fase04\src\evaluate_audio_flags.py --dataset icbhi
python entregas\fase04\src\evaluate_audio_flags.py --dataset voiced
python entregas\fase04\src\evaluate_audio_flags.py --dataset parkinson
python entregas\fase04\src\evaluate_audio_flags.py --dataset cross
# Sonda de embeddings profundos (requer torch/transformers):
python entregas\fase04\src\evaluate_audio_flags.py --dataset voiced --include-embeddings
python entregas\fase04\src\evaluate_audio_flags.py --dataset icbhi --include-embeddings
```

A suite da Fase 4 passa com 93 testes verdes e 2 pulados (os pulados exigem
credenciais reais do Azure) com o Azure simulado. O comando `--dataset both`
baixa as bases para `data/raw/icbhi/` e `data/raw/voiced/`; `--dataset parkinson`
baixa a base da UCI para `data/raw/parkinson/`. Os comandos de avaliacao
reproduzem as metricas acima; o de VOICED regrava o artefato
`backend/model/voice_alteration_classifier.joblib` consumido pela API e o
`--dataset cross` grava
[`results/audio_cross_evaluation.json`](results/audio_cross_evaluation.json).
As duas ultimas linhas acrescentam a comparacao com os embeddings profundos; sem
`torch`/`transformers` instalados a sonda e registrada como `skipped` e o restante
da avaliacao segue normalmente.

## 8. Referencia

Černek, A., Sedmidubsky, J., Budikova, P. *REHAB24-6: Physical Therapy Dataset
for Analyzing Pose Estimation Methods*. SISAP, Springer, 2024.
[Registro e arquivos do dataset](https://zenodo.org/records/13305826).

Rocha, B. M. et al. *An open access database for the evaluation of respiratory
sound classification algorithms*. Physiological Measurement 40 035001, 2019.
[ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).

Cesari, U., De Pietro, G., Marciano, E., Niri, C., Sannino, G., Verde, L. *A new
database of healthy and pathological voices*. Computers & Electrical Engineering
68:310-321, 2018.
[VOICED Database (PhysioNet)](https://physionet.org/content/voiced/1.0.0/).

Little, M. A., McSharry, P. E., Hunter, E. J., Spielman, J., Ramig, L. O.
*Suitability of dysphonia measurements for telemonitoring of Parkinson's
disease*. IEEE Transactions on Biomedical Engineering 56(4):1015-1022, 2007.
[UCI Parkinson's Disease (ID 174)](https://archive.ics.uci.edu/dataset/174/parkinsons).
