# Fase 4 — Monitoramento multimodal em reabilitação

## Executar um caso multimodal

1. Inicie o backend e o Streamlit conforme o [README da raiz](../../README.md#como-executar).
2. Selecione **Caso multimodal (Fase 4)**. Em **Demonstração sintética**, use um identificador novo com prefixo `DEMO-` e clique em **Executar caso sintético**. A nota, a transcrição, o sentimento e os achados visuais e vocais são simulados. O detector processa 51 observações sintéticas e gera seis alertas persistidos.
3. Abra **Detecção de anomalias (Fase 4)** com o mesmo identificador para conferir o histórico, o resultado de envio e a marcação como visto.

Em **Enviar áudio e vídeo**, indique o paciente já acompanhado, o início e o fim da sessão com fuso horário, a nota, o exercício e os arquivos. Use um WAV PCM de 16 bits, 16 kHz, mono, de até 60 s e um MP4 recortado para uma repetição. A janela temporal inclui os dois limites. A API `POST /analyze-rehabilitation-case` também aceita `start_frame` e `end_frame` para arquivos com várias repetições. Os índices começam em zero.

### Preparar análise real

- **Azure Speech** usa `AZURE_SPEECH_KEY` e `AZURE_SPEECH_REGION`. **Azure AI Language** usa `AZURE_LANGUAGE_KEY` e `AZURE_LANGUAGE_ENDPOINT`. Configure no ambiente do backend ou em um `.env` local com base no [arquivo de exemplo](../../.env.example). O padrão é `pt-BR`.
- **Vídeo** usa `backend/model/rehab_pose_classifier.joblib` e `yolov8n-pose.pt`. Para caminhos alternativos, use `REHAB_POSE_MODEL_PATH` e `REHAB_POSE_WEIGHTS`. Veja [Treinar vídeo](#treinar-vídeo) para gerar o classificador.
- **Canal de teste** usa `ALERT_WEBHOOK_URL`. Reserve um banco de demonstração via `ANOMALY_DB_PATH` e use um paciente sintético novo a cada ensaio.

Para validar as chamadas ao Azure, exporte as variáveis acima no terminal de teste, aponte `AZURE_AUDIO_SAMPLE_FILE` para o WAV usado no ensaio e execute:

```sh
python -m pytest tests/fase04/test_azure_audio_live.py -q -rs
```

O [roteiro de gravação](ROTEIRO_VIDEO.md) define o que mostrar em cada tela e como comprovar a entrega do alerta.

Na demonstração final, o WAV sintético foi processado com Azure Speech e Azure AI Language, e o alerta de oxigenação foi conferido no Slack e marcado como visto. A [validação integrada](TESTE_INTEGRADO.md#validação-da-entrega-final-em-23112026) descreve o caso e os resultados.

### Usar os arquivos do ensaio local

O ensaio desta revisão deixou em `tmp/fase04-ensaio/` um MP4 do REHAB24-6, uma fala sintética em português no formato WAV, os pesos de pose, o classificador experimental de agachamento e suas métricas. A pasta é local e ignorada pelo Git. Para abrir a aplicação com esse classificador, ajuste no ambiente do backend:

```dotenv
REHAB_POSE_MODEL_PATH=tmp/fase04-ensaio/rehab_pose_classifier_agachamento.joblib
REHAB_POSE_WEIGHTS=tmp/fase04-ensaio/yolov8n-pose.pt
ANOMALY_DB_PATH=tmp/fase04-ensaio/anomalias.db
```

Selecione **Agachamento** e envie `ensaio_agachamento.mp4` e `ensaio_audio.wav`. Use a configuração Azure descrita acima para transcrever a amostra e analisar o sentimento. Em outro checkout, prepare os arquivos a partir do [dataset](data/README.md), do [treino de vídeo](#treinar-vídeo) e de uma fala encenada.

## Ponto 3 — Detecção de anomalias

O módulo de sinais vitais, prescrições e movimentação está disponível na opção
**Detecção de anomalias (Fase 4)**. Consulte [ANOMALIAS.md](ANOMALIAS.md) para
execução, contrato de entrada, métodos, avaliação sintética e roteiro de demonstração.

Verificação da aplicação completa: [Teste integrado](TESTE_INTEGRADO.md), com
resultados, falhas existentes no assistente e limitações por recursos ausentes.

Esta pasta contem o fluxo experimental para estimar pose em videos de exercicios
de reabilitacao, classificar repeticoes como corretas/incorretas e produzir
metricas e relatorios. O dataset REHAB24-6 contem voluntarios saudaveis; o modelo
nao deve ser usado para decisao clinica.

## Estrutura

```text
entregas/fase04/
├── data/
│   ├── README.md
│   └── raw/
│       └── rehab24-6/   <- arquivos originais do Zenodo (nao versionados)
├── src/                 <- treinamento e avaliacao
├── results/             <- metricas do experimento
├── download_dataset.py  <- download com verificacao de checksum
├── README.md
└── RELATORIO_TECNICO_CONSOLIDADO.md
```

O codigo integrado da aplicacao esta em `backend/rehab_video/`, com a rota em
`backend/main.py` e a tela no `frontend/app.py`. O modelo treinado e o peso do
YOLO sao artefatos locais ignorados pelo Git. Os testes automatizados ficam em
`tests/fase04/` na raiz do repositorio.

## Dataset e uso

Veja [data/README.md](data/README.md) para a fonte, os arquivos previstos, a
licenca e o procedimento de verificacao. Os arquivos brutos sao grandes e estao
excluidos do Git.

## Treinar vídeo

Com Python 3.11, dependencias instaladas e os videos em
`data/raw/rehab24-6/Ex1` a `Ex6`, execute a partir da raiz do repositorio:

```powershell
py -3.11 entregas\fase04\src\train_classifier.py
```

O comando valida anotacoes e videos, estima pose com YOLOv8n-pose, separa os
participantes em treino/validacao/teste e salva o classificador e as metricas
em `backend/model/rehab_pose_classifier.joblib` e
`entregas/fase04/results/training_report.json`. A inferencia na API exige o
artefato treinado. O resultado medido e o baseline estao documentados no
[relatorio tecnico](RELATORIO_TECNICO_CONSOLIDADO.md). Na API, `POST /analyze-rehabilitation-video` classifica a repeticao e
`POST /annotate-rehabilitation-video` devolve um MP4 com o esqueleto (pose
estimada) desenhado sobre o video; a tela da Fase 4 exibe e permite baixar esse
video. A resposta de analise inclui `deviation_report`, um relatorio automatico
de desvios (amplitude articular reduzida, assimetria entre lados e inclinacao do
tronco) calculado por regras heuristicas sobre os angulos da pose; os limiares
nao foram validados clinicamente. Para executar os testes:

```powershell
py -3.11 -m pytest tests\fase04 -q
```
