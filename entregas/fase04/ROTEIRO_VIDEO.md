# Roteiro de apresentação da Fase 4

**FIAP · Inteligência Artificial para Desenvolvedores · Grupo 69 · 15:00 com os créditos**

O vídeo acompanha um caso demonstrativo de fisioterapia: duas mídias processadas na aplicação, uma nota textual e uma série de monitoramento sintética. A última tomada segue um alerta de oxigenação do painel até o canal da equipe e volta para registrar a leitura.

## Preparação fora da gravação

### Arquivos e telas

| Onde | Caminho ou valor | O que fazer |
|---|---|---|
| Finder, ao explicar o dataset | `entregas/fase04/data/raw/rehab24-6/` | Mostre os nove arquivos já baixados e verificados, em uma tomada curta. Abra `Segmentation.csv` para mostrar a estrutura das anotações. |
| **Análise de movimento** e **Caso multimodal** | `tmp/fase04-ensaio/ensaio_agachamento.mp4` | Suba este MP4 nas duas telas. É o trecho de 3,37 segundos já recortado de uma repetição de agachamento da Camera17 do REHAB24-6. |
| **Análise de áudio** e **Caso multimodal** | `tmp/fase04-ensaio/ensaio_audio.wav` | Suba este WAV nas duas telas. Contém 7 segundos de fala sintética em português, em PCM de 16 bits, mono, 16 kHz. |
| **Detecção de anomalias** | Botão **Reproduzir demonstração sintética** | O botão cria e envia as observações do paciente selecionado. Deixe **Importar histórico JSON** vazio. |
| Backend, antes de gravar | `tmp/fase04-ensaio/rehab_pose_classifier_agachamento.joblib` e `tmp/fase04-ensaio/yolov8n-pose.pt` | São artefatos de execução. Configure os caminhos no backend e deixe-os fora dos campos de upload. |
| Repositório, no encerramento | `README.md` e [Relatório Técnico Consolidado](RELATORIO_TECNICO_CONSOLIDADO.md) | Deixe abertas a lista de arquivos e as seções de método, métricas e anomalias. |

No seletor de arquivos do macOS, use **Cmd + Shift + G** e abra a pasta `tmp/fase04-ensaio` a partir do seu checkout. Use o mesmo MP4 e o mesmo WAV em todas as tomadas do caso. Mantenha os ZIPs locais. Para o vídeo acadêmico, use o trecho do REHAB24-6 conforme os termos do registro, sem monetização, e cite o artigo na descrição. Se optar por uma filmagem própria, grave uma repetição de agachamento em MP4 com autorização para publicação e reensaie a análise antes de gravar. [Registro, termos e referência do REHAB24-6](https://zenodo.org/records/13305826#description)

Os nove arquivos do dataset já estão disponíveis. O MP4 curto está pronto para o upload. A extração de `videos.zip` ao lado de `Segmentation.csv` fica para quando o grupo quiser reproduzir o treinamento com as pastas `Ex1` a `Ex6`. [Guia dos dados](data/README.md)

### Conferência técnica

1. Abra o Streamlit no Dia em [127.0.0.1:8501](http://127.0.0.1:8501) e confirme que as quatro telas da Fase 4 estão disponíveis. Se precisar iniciar o ambiente, use os comandos abaixo a partir da raiz do projeto, em dois terminais. Use apenas uma instância do backend na porta 8888. O `.env` local deve carregar as credenciais de Speech e Language e, para a tomada do canal, `ALERT_WEBHOOK_URL`.
2. Abra o Azure Portal na visão geral do grupo `rg-fiap-tech-challenge-fase4`. Deixe à vista os recursos `fiap-g69-fase4-speech` e `fiap-g69-fase4-language-retry`. Capture somente essa lista ou as visões gerais dos recursos. Feche terminais, a tela **Keys and Endpoint** e qualquer página que mostre credenciais.
3. Separe um canal de teste autorizado do grupo. Se usar Slack, habilite **Incoming Webhooks** no app do grupo, use **Add New Webhook to Workspace** para selecionar o canal e salve a URL em `ALERT_WEBHOOK_URL` no ambiente privado do backend. Reinicie o backend após a configuração. Abra o canal e deixe o painel de anomalias em outra aba. [Passos oficiais do Slack](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/#getting-started-with-incoming-webhooks)
4. Faça um ensaio com um paciente novo, como `DEMO-G69-ENSAIO-01`. Confira que a repetição é classificada e anotada, que o áudio gera transcrição e frases-chave e que o mesmo ID de alerta aparece no painel e no canal. Para a gravação final, use `DEMO-G69-VIDEO-01` se estiver livre, ou troque o sufixo. Use esse mesmo ID em todas as telas da gravação. Cada repetição do replay com o mesmo ID reutiliza os eventos e evita alertas duplicados.
5. Grave em 1080p com os botões e rótulos legíveis. Abra os arquivos e as abas com antecedência. Na edição, corte a espera por análise, preservando o clique, o começo do processamento e o resultado da mesma execução. Reserve os 10 segundos finais para a cartela e mantenha o arquivo exportado dentro de 15:00.

```sh
ITEM1_PRELOAD=false REHAB_AUDIO_LANGUAGE=pt-BR \
  REHAB_POSE_MODEL_PATH=tmp/fase04-ensaio/rehab_pose_classifier_agachamento.joblib \
  REHAB_POSE_WEIGHTS=tmp/fase04-ensaio/yolov8n-pose.pt \
  ANOMALY_DB_PATH=tmp/fase04-ensaio/anomalias-gravacao.db \
  .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8888 --env-file .env
```

```sh
.venv/bin/python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

**Resultado com esses arquivos.** A API retornou HTTP 200 com os dois serviços Azure, classificou a repetição como possivelmente correta, apontou amplitude reduzida e assimetria e retornou **Revisão prioritária** com seis alertas associados. Na gravação final, o ID do alerta de oxigenação foi conferido no caso, no Slack e no painel, que registrou a marcação como visto. [Validação da entrega](TESTE_INTEGRADO.md#validação-da-entrega-final-em-23112026)

## Gravação, fala por fala

Leia apenas os blocos de **Fala**. Use o restante do tempo de cada tomada para executar os cliques, deixar os resultados legíveis e tocar o vídeo anotado.

### 00:00–00:40 | Apresentação e percurso

**Na tela**

1. Mostre a identificação da FIAP, a Fase 4 e o Grupo 69. Abra **Caso multimodal (Fase 4)** e deixe visível o menu lateral com as telas de movimento, áudio e anomalias.
2. Mantenha o cursor no menu enquanto apresenta a sequência. Passe para o Finder no fim da fala.

**Fala**

> Olá, somos o Grupo 69 da pós-graduação em Inteligência Artificial para Desenvolvedores. O desafio desta fase é reunir dados de diferentes fontes para ajudar uma equipe de saúde a identificar sinais que merecem revisão. Vamos acompanhar um caso de fisioterapia. Primeiro, analisamos um agachamento e uma fala em português. Em seguida, reproduzimos o monitoramento de uma internação e mostramos como um alerta chega ao canal da equipe. No fim, voltamos ao mesmo caso para registrar que o alerta foi visto.

### 00:40–01:35 | Dataset e escolha dos arquivos

**Na tela**

1. Abra `entregas/fase04/data/raw/rehab24-6/`. Aponte para `videos.zip`, os quatro ZIPs de articulações e marcadores 2D/3D, `Segmentation.csv`, `Segmentation.txt` e os dois arquivos de nomes.
2. Abra uma prévia de `Segmentation.csv`. Mostre `first_frame`, `last_frame` e `correctness` e, em seguida, mostre `ensaio_agachamento.mp4` no Finder.

**Fala**

> Para estudar o movimento, usamos o REHAB24-6. O conjunto reúne 65 gravações, duas câmeras, seis exercícios e 1.072 repetições anotadas. O download local já inclui os nove arquivos. O CSV informa onde começa e termina cada repetição e traz o rótulo de execução. Usamos os vídeos da Camera17 no experimento e separamos uma repetição de agachamento para esta demonstração. As articulações e os marcadores são referência para estudos de pose. A comparação quantitativa com essa referência fica como trabalho futuro.

### 01:35–02:30 | Upload do vídeo e inferência

**Na tela**

1. No Streamlit, escolha **Análise de movimento (Fase 4)**. Em **Exercício**, selecione **6 — Agachamento**.
2. Em **Vídeo da repetição**, selecione `ensaio_agachamento.mp4`. Deixe **Frame inicial** em `0` e **Frame final** em `0`, que nessa tela significa usar até o último frame.
3. Confira **Gerar também o vídeo com o esqueleto desenhado** e clique em **Analisar repetição**. Deixe o nome do arquivo e o início do processamento visíveis antes de cortar a espera.

**Fala**

> Aqui é onde a equipe envia o vídeo do exercício. Seleciono agachamento e envio o trecho com uma única repetição. Como o arquivo já foi recortado, uso todos os quadros. O YOLOv8 Pose estima pontos do corpo, como quadris, joelhos e tornozelos. A partir dos quadros amostrados, calculamos características do movimento e usamos uma regressão logística para estimar a classe da repetição. A opção marcada também gera um vídeo com a pose desenhada sobre as imagens.

### 02:30–03:35 | Resultado do vídeo e relatório de desvios

**Na tela**

1. Aponte para **Classificação**, **Probabilidade estimada de execução correta** e **Frames amostrados**. Leia a classificação exibida.
2. Em **Relatório de desvios**, destaque um achado visível e abra as métricas por alguns segundos. Aponte para **Baixar relatório da análise (JSON)**, que gera `relatorio_analise_reabilitacao.json`.
3. Em **Vídeo com pose estimada**, dê play na repetição inteira. Aponte para **Baixar vídeo anotado (MP4)**, que gera `analise_reabilitacao_anotado.mp4`.

**Fala**

> O primeiro resultado é a classificação estimada para a repetição, com a probabilidade do modelo. Abaixo, o relatório descreve o que as regras de movimento conseguiram sinalizar. No ensaio com este clipe, apareceram amplitude reduzida e assimetria. Repare que a classificação e as regras respondem a perguntas diferentes e podem produzir resultados distintos para o mesmo vídeo. Agora vou reproduzir o trecho anotado. O esqueleto ajuda a localizar os pontos em que a aplicação baseou as medidas. O MP4 anotado e o relatório podem ser baixados para revisão.

### 03:35–04:25 | Resultados experimentais do movimento

**Na tela**

1. Mostre a seção **Resultados experimentais** do relatório técnico. Aponte para as linhas **Teste** e **Baseline majoritário (teste)**.
2. Volte para a tela da aplicação ou para uma cartela simples com os resultados do modelo de ensaio de agachamento. Deixe os números legíveis antes de passar ao Azure.

**Fala**

> Na avaliação original com seis exercícios, mantivemos participantes diferentes entre treino, validação e teste. O modelo teve 59,46 por cento de acurácia e 57,80 por cento de F1 macro no teste. A referência de sempre prever a classe mais comum teve 43,69 por cento de acurácia. Hoje usamos um modelo de ensaio só para agachamento. Nesse recorte, ele acertou 18 de 23 repetições, empatou em acurácia com a referência e encontrou apenas 20 por cento das repetições incorretas. São resultados experimentais obtidos com voluntários saudáveis.

### 04:25–05:10 | Recursos do Azure e upload do áudio

**Na tela**

1. No Azure Portal, aponte na lista para `fiap-g69-fase4-speech` e `fiap-g69-fase4-language-retry`. Volte à aplicação.
2. Abra **Análise de áudio (Fase 4)**. Em **Áudio da sessão**, selecione `ensaio_audio.wav`. Clique em **Analisar áudio** e mostre o início do processamento.

**Fala**

> Para a fala, o backend usa dois serviços gerenciados no Azure. O Speech to Text recebe o áudio e devolve a transcrição em português. O Azure AI Language, chamado de Text Analytics no enunciado, recebe o texto reconhecido e devolve o sentimento, as frases-chave e as entidades identificadas. Vou enviar um WAV de sete segundos com uma fala sintética de fisioterapia. Quando clico em analisar, a aplicação chama os serviços e reúne as respostas na tela.

### 05:10–06:10 | Transcrição, sentimento e termos

**Na tela**

1. Em **Transcrição**, realce “Estou com dor no joelho e cansaço quando agacho”. Deixe o restante do texto legível.
2. Passe por **Sentimento** e mostre o rótulo e as três pontuações. Desça até **Termos e entidades (Azure Text Analytics)** e aponte para “dor”, “joelho” ou “cansaço”.
3. Mostre **Termos críticos encontrados nos trechos destacados pelo Azure**. Se o resultado variar, aponte para os termos efetivamente visíveis.

**Fala**

> Aqui está o texto reconhecido pelo Speech. A frase “Estou com dor no joelho e cansaço quando agacho” já aparece pronta para leitura. Na análise do Language, o sentimento deste exemplo saiu como negativo. As pontuações se referem ao texto da fala. O serviço também destacou termos como dor, joelho e cansaço. A aplicação cruza os trechos destacados pelo Azure com seu dicionário de termos críticos e mostra onde as duas análises coincidem. Isso permite voltar ao trecho que motivou o destaque e ler a frase inteira.

### 06:10–06:55 | Indicadores vocais e exportação

**Na tela**

1. Mostre a tabela **Termos críticos** e volte a descer até **Indicadores vocais**. No ensaio, a amostra gerou um indicador de possível disartria. Mostre a mensagem que aparecer e leia o resultado como um indicador experimental.
2. Abra as métricas recolhidas, se necessário, e aponte para **Baixar relatório da análise (JSON)**, que gera `relatorio_analise_audio.json`.

**Fala**

> O projeto também analisa o sinal de áudio. Ele calcula medidas como energia da voz, proporção de pausas e taxa de fala. No ensaio, apareceu um indicador experimental de possível disartria. Essas medidas variam com a gravação e precisam de interpretação profissional. O relatório reúne a transcrição, o sentimento, os termos e as medidas. Vamos agora olhar o histórico de monitoramento associado ao caso.

### 06:55–07:45 | Entrada das observações sintéticas

**Na tela**

1. Abra **Detecção de anomalias (Fase 4)**. Em **Identificador do paciente**, digite `DEMO-G69-VIDEO-01` ou o ID novo escolhido. Pressione Enter e mostre o painel inicial.
2. Clique em **Reproduzir demonstração sintética**. Mostre o progresso, o aviso de alerta automático e os contadores **Observações**, **Alertas registrados** e **Não vistos** ao final.
3. Na edição final, exiba a legenda **Replay de histórico já processado · os seis alertas já estavam registrados** no início da tomada para explicar os contadores preenchidos.

**Fala**

> Agora simulamos o monitoramento de uma internação. Este identificador representa um paciente fictício e vai acompanhar todo o caso. O botão envia 51 observações, uma de cada vez, na ordem do histórico. São 24 observações de sinais vitais, 24 de atividade e três de prescrição. A reprodução é acelerada para caber na demonstração. À medida que a API recebe os eventos, ela atualiza o histórico e aplica as regras. Nesta sequência padrão, o painel registra seis alertas automáticos.

### 07:45–08:45 | Batimentos, pressão e oxigenação

**Na tela**

1. Em **Sinais vitais**, percorra `heart_rate`, `systolic`, `diastolic` e `spo2`. Pare alguns segundos no gráfico de oxigenação e aponte para a queda.
2. Em **Alertas para revisão**, abra `spo2_low`, confirmado às `01:05 UTC` no replay padrão. Mostre a evidência com a leitura observada, a mediana do histórico e o escore. Mantenha o alerta como **Novo**.

**Fala**

> Aqui vemos a frequência cardíaca, as duas medidas de pressão arterial e a oxigenação. O detector usa a mediana e a dispersão robusta das leituras anteriores do próprio paciente para avaliar cada mudança. Para os sinais vitais, duas observações consecutivas na mesma direção confirmam o desvio. Neste exemplo, a oxigenação cai e continua baixa na leitura seguinte. O alerta mostra a variável, o valor observado, a referência usada e o escore. Assim, a equipe consegue abrir o registro e entender o motivo da sinalização.

### 08:45–09:35 | Atividade e prescrição

**Na tela**

1. Mostre **Atividade por janela de 5 minutos** e aponte para as janelas com atividade reduzida.
2. Mostre **Evolução das prescrições**. Aponte para `MEDICAMENTO-FICTICIO` e a mudança simulada de `10` para `20 mg`. Em **Alertas para revisão**, mostre também os alertas de atividade e de prescrição.

**Fala**

> O mesmo histórico inclui uma série de atividade em janelas de cinco minutos. Neste trecho, a fração de tempo ativo caiu em relação ao padrão anterior. Essa atividade vem das observações do simulador. Ela representa a internação, enquanto o MP4 representa uma repetição isolada da fisioterapia. A prescrição tem uma regra própria. Ela compara versões consecutivas e, neste exemplo fictício, sinaliza a mudança de 10 para 20 miligramas. A equipe revisa a alteração com o contexto do caso.

### 09:35–10:20 | Escolha do alerta que será acompanhado

**Na tela**

1. Volte ao alerta de oxigenação. Selecione e copie o **ID do alerta**. Anote também o paciente e o horário em UTC, fora da área de captura.
2. Mostre o status de entrega que a tela exibir. Guarde o clique **Marcar como visto** para o final. Passe para **Caso multimodal (Fase 4)** mantendo o mesmo paciente.

**Fala**

> Vamos acompanhar este alerta de oxigenação até o fim. Ele tem um identificador próprio, além do paciente e do horário da confirmação. Quando a observação confirma o desvio, a API grava o alerta e a pendência de notificação na mesma transação. O envio para o canal acontece em segundo plano. Vou guardar este identificador porque ele vai aparecer no relatório da sessão e na mensagem que a equipe recebe. Depois, voltaremos a este alerta para registrar a leitura.

### 10:20–11:20 | Preenchimento do caso multimodal

**Na tela**

1. Em **Caso multimodal (Fase 4)**, selecione **Enviar áudio e vídeo** em **Origem dos dados**.
2. Preencha o formulário conforme a tabela. Mostre os campos e os nomes dos dois arquivos antes de clicar.
3. Clique em **Analisar sessão**. Preserve na edição o começo do processamento e o resultado da mesma execução.

| Campo | Valor para a gravação |
|---|---|
| **Identificador no painel de anomalias** | O mesmo do replay, por exemplo `DEMO-G69-VIDEO-01` |
| **Início da sessão (ISO 8601 com fuso)** | `2026-01-01T00:00:00Z` |
| **Fim da sessão (ISO 8601 com fuso)** | `2026-01-01T01:55:00Z` |
| **Nota da sessão** | `Paciente relatou dor no joelho e cansaço durante o agachamento.` |
| **Exercício** | **Agachamento** |
| **Áudio** | `tmp/fase04-ensaio/ensaio_audio.wav` |
| **Vídeo de uma repetição** | `tmp/fase04-ensaio/ensaio_agachamento.mp4` |

**Fala**

> Esta é a etapa que reúne as modalidades. Seleciono o envio de áudio e vídeo e uso o mesmo paciente do monitoramento. A janela vai de meia-noite até uma hora e cinquenta e cinco, em UTC, e cobre o período reproduzido. A nota registra o relato de dor no joelho e cansaço durante o agachamento. Em seguida, escolho o exercício e envio os dois arquivos que já vimos. Ao clicar em analisar sessão, o backend processa as mídias e procura os alertas confirmados para esse paciente dentro da janela.

### 11:20–12:00 | Evidências de texto, áudio e vídeo

**Na tela**

1. Mostre o título de revisão, o paciente, a janela e o motivo exibido pelo sistema.
2. Passe pelas seções **Texto da sessão**, **Áudio e transcrição** e **Vídeo e pose**. Em cada uma, deixe pelo menos um achado ou o resumo visível.

**Fala**

> O relatório mostra a nota da sessão primeiro. Depois, aparecem a transcrição, o sentimento e os achados de áudio. Na seção de vídeo, vemos o resumo do movimento analisado. A regra experimental dá prioridade quando há alertas confirmados na janela e sinais em pelo menos duas modalidades. Neste caso, podemos ler a justificativa na própria tela. A decisão final sobre o que fazer com esses achados fica com o profissional que revisar a sessão.

### 12:00–12:35 | Associação dos alertas ao caso

**Na tela**

1. Role até **Alertas confirmados na janela**. Encontre o alerta de oxigenação e compare o ID com o que foi copiado do painel.
2. Aponte para **Baixar relatório do caso (JSON)**. Baixe `caso_multimodal.json` se quiser mostrar a saída completa. Deixe o foco no ID e passe ao canal.

**Fala**

> Aqui estão os alertas do monitoramento associados à sessão. O identificador de oxigenação é o mesmo que copiamos do painel. Esse vínculo permite sair de um achado no caso e localizar a observação e a evidência que deram origem ao alerta. A opção de baixar o JSON guarda juntos os resultados de texto, áudio e vídeo, a prioridade de revisão e os alertas vinculados.

### 12:35–13:25 | Recebimento no canal da equipe

**Na tela**

1. Abra apenas o canal de teste do Slack ou Teams. Localize a mensagem buscando pelo **ID do alerta** ou pelo identificador do paciente.
2. Aproxime o ID, a variável de oxigenação, o valor e o contexto que aparecem na mensagem. Compare o ID com a linha do relatório e volte para **Detecção de anomalias (Fase 4)**.

**Fala após conferir o recebimento**

> Esta mensagem foi enviada automaticamente para o canal de teste, que representa a equipe responsável. Encontramos o mesmo identificador do alerta que apareceu no painel e no relatório multimodal. A mensagem traz o paciente, o horário e o motivo da sinalização. Para este alerta de oxigenação, também mostra o valor e a referência do histórico. Com esses dados, a equipe consegue identificar qual registro precisa abrir para revisar a ocorrência.

### 13:25–14:15 | Resposta da equipe no painel

**Na tela**

1. Confirme o mesmo paciente. Se necessário, clique em **Atualizar painel**. Abra o alerta de oxigenação usando o mesmo ID.
2. Mostre **Enviado à equipe** se o canal recebeu a mensagem. Clique em **Marcar como visto**. Abra novamente o alerta após a atualização e mostre **Visto** e o horário de visualização.

**Fala**

> De volta ao painel, abro o alerta pelo mesmo identificador e confiro o status da notificação. Agora vou marcar o alerta como visto, simulando a resposta da equipe. O painel atualiza o registro e mostra o horário da leitura. Esse passo registra que o alerta foi consultado. Com o identificador, conseguimos acompanhar o caminho desde a observação que confirmou o desvio, passando pela notificação, até o primeiro registro de revisão pela equipe.

### 14:15–14:50 | Repositório e limites da solução

**Na tela**

1. Abra o repositório Git e mostre o `README.md` com o link do **Relatório Técnico Consolidado**. Aponte para o código-fonte e para as seções sobre fluxo multimodal, modelos, métricas e exemplos de anomalias.
2. Volte brevemente ao relatório do caso ou deixe a tela do repositório visível para a última fala.

**Fala**

> No repositório estão o código-fonte e o relatório técnico, com o fluxo, os modelos, as métricas e exemplos de anomalias. A avaliação aponta limitações, como a baixa sensibilidade a mudanças sutis no monitoramento sintético. O resultado é um protótipo de apoio à revisão humana. A equipe pode abrir o caso, entender um alerta e registrar que ele foi visto.

### 14:50–15:00 | Encerramento

**Na tela**

1. Mostre a cartela com **FIAP · Tech Challenge Fase 4 · Grupo 69** e o endereço do repositório. Deixe a cartela até o fim, com os créditos dentro dos 15 minutos.

**Fala**

> Obrigado. Este foi o Tech Challenge da Fase 4 do Grupo 69.

## Se a gravação precisar de uma nova tomada

1. **Vídeo ou áudio demorou.** Corte a espera pela mesma execução. Se o vídeo ainda falhar, use um ensaio gravado anteriormente com a mesma entrada e identifique-o na tela. Se a API Azure estiver indisponível, use uma captura de uma execução real identificada como ensaio ou regrave a etapa antes de publicar.
2. **Mudou o resultado.** Leia os rótulos e os achados que aparecerem. Se o caso retornar **Revisão indicada**, diga “A regra encontrou evidências para revisão nesta sessão. O título mostra a prioridade calculada para estes dados.” Use somente a fala correspondente ao resultado mostrado.
3. **Canal ainda indisponível.** Mostre o status real no painel e marque o alerta como visto. Diga “O alerta está registrado no painel. Ainda precisamos confirmar a entrega no canal do grupo.” Se tiver gravado uma entrega no receptor HTTP local, identifique-o como **receptor de teste local**. Para a versão final pedida pelo enunciado, complete a tomada com a mensagem no canal do grupo e o mesmo ID. Use um paciente novo depois de corrigir o webhook. O sistema trata os resultados **Falha no envio à equipe** e **Somente no painel** como estados finais.

## Conferência e publicação

1. Exporte a apresentação como `FIAP_Fase4_Grupo69.mp4`. Confira se o arquivo tem até **15:00** e inclui o MP4 de entrada, pose anotada e relatório de desvios, WAV, transcrição e sentimento reais do Azure, achados acústicos, sinais vitais, atividade, prescrição, caso integrado e identificação do mesmo alerta no painel, no canal e na marcação de visto.
2. No **YouTube Studio**, clique em **Criar → Enviar vídeos** e envie `FIAP_Fase4_Grupo69.mp4`. Use o título **FIAP Tech Challenge Fase 4 | Grupo 69 | Monitoramento multimodal em reabilitação**. Na descrição, insira o link do repositório e a referência de Černek, Sedmidubsky e Budikova, SISAP 2024, com o DOI `10.5281/zenodo.13305826`. Em **Visibilidade**, selecione **Não listado** ou **Público** e salve. O enunciado também aceita publicação no Vimeo com uma dessas visibilidades. [Instruções oficiais de envio pelo computador](https://support.google.com/youtube/answer/57407?co=GENIE.Platform%3DDesktop&hl=pt-BR)
3. Abra o vídeo fora da conta que publicou e confirme que ele pode ser reproduzido. Registre o link na seção **Documentação da Fase 4** do `README.md` do repositório.
