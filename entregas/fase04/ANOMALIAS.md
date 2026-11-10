# Ponto 3 — Detecção de anomalias

## Escopo implementado

Protótipo de monitoramento temporal com três frentes: sinais vitais, evolução de
prescrições e atividade do paciente durante a internação. Recebe observações JSON,
compara com o histórico anterior e registra alertas no SQLite. A interface Streamlit
mostra gráficos, evolução de prescrições, evidências, alertas pendentes e exportação
JSON. Marcar como visto não significa resolver a anomalia nem aprovar uma conduta.

O módulo está integrado ao mesmo FastAPI e à navegação da aplicação de áudio/vídeo.
Não altera esses modelos. A atividade é uma medida temporal recebida pela API:
não há conexão automática a sensores nem extração de atividade do vídeo de
fisioterapia. Esse vídeo mede exercícios específicos e não representa, sozinho,
movimentação durante a internação. O exemplo reproduz internação sintética.

## Executar

Na aplicação completa, use o ambiente Python 3.11 e as dependências da raiz,
inicie `uvicorn backend.main:app --port 8888` e `streamlit run frontend/app.py`.
Selecione **Detecção de anomalias (Fase 4)** na navegação.

Também é possível executar somente o ponto 3, sem carregar LLM, YOLO ou TensorFlow:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-anomalies.txt
.\.venv\Scripts\python.exe -m uvicorn backend.anomalies.standalone:app --port 8888
```

Em outro terminal, na raiz:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/anomalies_app.py
```

Abra a URL exibida pelo Streamlit, mantenha o paciente sintético proposto e clique
em **Reproduzir demonstração sintética**. São 51 eventos (24 sinais vitais, 24
movimento, três prescrições). Na demonstração padrão surgem seis alertas: quatro
variáveis vitais, um de atividade e um de prescrição. A reprodução é acelerada,
com timestamps históricos; não representa captura ao vivo. A API também aceita
novas observações individualmente, avaliadas assim que recebidas.

## Contrato de entrada

`POST /anomalies/observations` recebe `{"observations": [...]}`, de 1 a 500 eventos.
Um lote é atômico: conflito em qualquer evento desfaz todo o lote.

Campos comuns: `event_id` (único global), `patient_id`, `timestamp` ISO 8601 com
fuso, `kind` e `context` (`awake` ou `rest`, padrão `awake`). Use IDs novos para
novos eventos. Repetir um evento idêntico não duplica observações ou alertas;
reutilizar seu ID com dados diferentes retorna HTTP 409. Datas são normalizadas
para UTC. Eventos devem chegar em ordem estritamente crescente por paciente e
fluxo; cada medicamento tem um fluxo independente. A API não reordena nem
reprocessa eventos retroativos.

| kind | Conteúdo | Unidade |
|---|---|---|
| vitals | metrics: heart_rate, systolic, diastolic, spo2 | bpm, mmHg, mmHg, % |
| movement | metrics: activity_fraction | Fração de tempo ativo de 0 a 1 em janela de 5 minutos |
| prescription | prescription: medication, dose, unit, frequency_per_day, route, status | Dose por administração; frequência diária; active ou stopped |

As janelas de movimento devem ser consecutivas, não sobrepostas e medidas pelo
mesmo método, usando o timestamp de fim da janela. A fração deve ser calculada
por uma fonte externa ou pelo simulador: não é a probabilidade de acerto de um
exercício. Uma janela ausente não deve ser enviada como atividade zero.

Os quatro sinais vitais são obrigatórios por observação. Valores não finitos,
campos desconhecidos, intervalos inválidos e pressão diastólica maior ou igual à
sistólica são rejeitados (422). Os intervalos de validação são restrições técnicas
amplas, não limites de normalidade. Não há imputação de valores ausentes.

Para prescrição, envie eventos explícitos de mudança/suspensão. A ausência de um
medicamento em um lote não significa suspensão. Use nomes canônicos consistentes;
sinônimos e princípios ativos não são reconciliados. Mudança de unidade ou via
gera revisão sem comparar numericamente as doses nem tentar convertê-las.

Exemplos completos podem ser baixados no painel ou em `GET /anomalies/demo`.

## Método e parâmetros fixos

Sinais vitais e atividade usam mediana e desvio absoluto mediano (MAD) das últimas
12 observações anteriores comparáveis. São necessárias pelo menos seis leituras.
O escore é `abs(valor - mediana) / max(1.4826 * MAD, piso)`; candidato se >= 3,5.
Pisos experimentais: batimentos 3, sistólica 4, diastólica 3, SpO2 0,8 e atividade
0,04, nas unidades da tabela. Não são limiares clínicos e não foram calibrados em
pacientes reais. O método é estatístico e não requer treinamento supervisionado.

Dois candidatos consecutivos da mesma variável e direção confirmam a anomalia.
Um pico isolado fica registrado, mas não gera alerta. Não há repetição de alerta
enquanto o mesmo código permanece confirmado. Uma leitura sem esse candidato
encerra a sequência; nova confirmação pode gerar outro alerta. "Visto" não altera
o detector. O estado inicial é `warming_up`, não "normal".

A referência usa apenas passado e é isolada por paciente/fluxo. Mudança de contexto
ou lacuna maior que 15 minutos reinicia a referência. Movimento exige janelas de
cinco minutos, com tolerância de ±30 segundos para a variação normal do coletor;
cadência diferente disso também reinicia o histórico.

Prescrições usam comparação com a última versão do mesmo medicamento: variação
relativa >= 50% na dose ou frequência ativa, mudança de via/unidade/status. Cada
mudança é um evento discreto e pode gerar novo alerta imediatamente. A primeira
prescrição estabelece a referência. A regra não verifica se um tratamento está
correto, e pode sinalizar alterações planejadas.

## Histórico e alertas

- Banco padrão: `backend/data/anomalies.db` (ignorado pelo Git).
- Sobrescrita: variável `ANOMALY_DB_PATH`, útil para testes e execução isolada.
- `GET /anomalies/patients/{patient_id}`: últimas 500 observações e 500 alertas.
- `POST /anomalies/alerts/{alert_id}/acknowledge`: registra horário de visualização;
  repetição preserva o primeiro horário. ID inexistente retorna 404.
- Cada alerta contém paciente, evento, timestamp, modalidade, evidência, método
  e necessidade de revisão humana.

As gravações são transacionais e serializadas por SQLite.

### Entrega do alerta à equipe médica

Fluxo de um alerta confirmado:

1. A observação chega em `POST /anomalies/observations` e é avaliada na hora.
2. O alerta e sua pendência de envio (`notification_outbox`) são gravados no
   SQLite dentro da mesma transação da observação.
3. Depois da resposta HTTP, uma tarefa em segundo plano envia o alerta para o
   canal da equipe, se `ALERT_WEBHOOK_URL` estiver configurada. É um webhook de
   entrada do Microsoft Teams ou do Slack: corpo `{"text": ...}` com paciente,
   horário, variável, valor, mediana do histórico e escore. Um webhook lento ou
   fora do ar não atrasa nem desfaz a ingestão.
4. O resultado de cada envio fica registrado imediatamente por alerta (`sent`, `failed` ou
   `panel_only` quando não há webhook) e aparece no painel e no relatório JSON.
5. O painel Streamlit se atualiza sozinho a cada 5 segundos (pode ser desligado)
   e mostra um aviso na tela quando chega um alerta novo, inclusive enviado por
   outra fonte. A equipe marca o alerta como visto.

Configuração, no `.env` ou no ambiente do backend:

```powershell
$env:ALERT_WEBHOOK_URL = "https://<seu-webhook-do-teams-ou-slack>"
```

Um processador inicia com a API principal e com a API independente. Ele consulta
a fila ao iniciar e, quando não há trabalho, a cada cinco segundos. Pendências
persistem após reinicialização. Envios em andamento são reservados por 60 segundos;
após esse prazo, uma tentativa interrompida pode ser recuperada por outro
processador. A reserva transacional evita que tarefas HTTP e processadores
concorrentes enviem o mesmo alerta enquanto a reserva estiver válida.

O painel mostra `pending` (pendente), `sending` (em andamento), o número de
tentativas iniciadas e os resultados. URLs inválidas são registradas como
`failed`, sem expor o endereço ou token nos detalhes. Falhas HTTP/rede já
registradas e `panel_only` são finais: não há reenvio automático desses estados.
Configurar um webhook depois não envia retroativamente alertas marcados como
somente no painel. A recuperação automática abrange tentativas interrompidas,
não resultados finais conhecidos.

Não há garantia de entrega exatamente uma vez: se o destinatário aceitar a
mensagem e o processo cair antes de gravar `sent`, a recuperação pode reenviá-la.
O ID estável do alerta vai no texto para identificar duplicatas no destino.
Não existe transação distribuída entre SQLite e Teams/Slack. A deduplicação das
observações continua evitando novas pendências para o mesmo alerta.

A fila é criada para alertas novos a partir desta versão. Alertas antigos sem
registro de envio não são reenviados automaticamente, pois não é possível saber
se o destino já recebeu a mensagem. Permanecem disponíveis no painel.
Este protótipo local não inclui autenticação, autorização, identificação do
profissional, política de retenção ou integração com prontuário real. Não deve ser
exposto como serviço clínico sem implementar essas camadas.

## Validação reproduzível

```powershell
.\.venv\Scripts\python.exe -m pytest tests/fase04/test_anomalies.py tests/fase04/test_anomalies_ui.py tests/fase04/test_notification_recovery.py tests/fase04/test_integrated_application.py -q
.\.venv\Scripts\python.exe -m entregas.fase04.src.evaluate_anomalies
```

Resultados numéricos: `results/anomalies_evaluation.json`. A avaliação usa 60
pacientes sintéticos, 20 sementes por cenário (1000–1019), diferentes da semente
42 da interface. Cenários: normal, mudança forte e mudança sutil. Os rótulos vêm
dos intervalos de injeção, não da saída do detector. Parâmetros fixos, sem ajuste
nesses casos e sem divisão treino/teste de um modelo aprendido.

A unidade de avaliação é observação por modalidade após aquecimento. O primeiro
ponto alterado conta como falso negativo se ainda não confirmado. Alertas
deduplicados não são contados como observações negativas quando a anomalia
continua ativa. São reportadas matriz de confusão, precisão, recall, F1, acurácia e
baseline sempre normal. Métricas indefinidas são `null`, não zero artificial.

Resultados da execução de 10/11/2026:

| Cenário | Frente | Precisão | Recall | F1 | Acurácia | Baseline sempre normal |
|---|---|---:|---:|---:|---:|---:|
| Mudança forte | Sinais vitais | 100% | 75% | 85,71% | 94,44% | 77,78% |
| Mudança forte | Movimento | 100% | 75% | 85,71% | 94,44% | 77,78% |
| Mudança forte | Prescrição | 100% | 100% | 100% | 100% | 50% |
| Mudança sutil | Sinais vitais | Indefinida | 0% | 0% | 77,78% | 77,78% |
| Mudança sutil | Movimento | Indefinida | 0% | 0% | 77,78% | 77,78% |
| Mudança sutil | Prescrição | Indefinida | 0% | 0% | 50% | 50% |

No cenário normal houve zero falsos positivos em 360 observações vitais, 360 de
movimento e 40 de prescrição. Os números altos nas mudanças fortes refletem a
simplicidade da simulação. O recall zero nas mudanças sutis explicita a limitação
dos parâmetros escolhidos, em vez de apresentar o protótipo como detector geral.

Verificação ampliada: os testes incluem validação, detecção, persistência, API,
tolerância de cadência, webhook local, falhas e ausência de webhook, interface,
recuperação após interrupção, reserva expirada, concorrência entre processadores,
URL malformada e rollback da fila junto com o lote. A integração exercita a API
principal e navegação. O banco de testes é isolado; webhooks externos ficam
desabilitados por padrão nos testes. Ambiente de verificação do módulo: Python 3.14, FastAPI
0.162.2, Pydantic 2.14.3, Streamlit 1.67.0, pytest 9.2.0. A aplicação completa
mantém a recomendação de Python 3.11 por suas dependências anteriores. A suíte
completa também foi executada; serviços Azure/LLM reais continuam sem validação
por falta de credenciais/dependências/artefatos no ambiente.

## Limitações a apresentar

- Dados exclusivamente sintéticos: a avaliação verifica comportamento do código,
  não eficácia médica. Alterações sutis e picos isolados podem não gerar alertas.
- Com cadência de cinco minutos, confirmar dois pontos acrescenta cinco minutos
  à detecção. Falta histórico inicial e não há regra de emergência independente.
- A janela móvel pode absorver uma alteração muito prolongada como nova referência.
- Contexto awake/rest é fornecido pela origem, não inferido. Não distingue sono,
  sedação, exercício e artefato de sensor de forma autônoma.
- Mudanças planejadas de prescrição podem gerar falsos alertas; não há conhecimento
  farmacológico nem interpretação de justificativas de prescrição.
- Medidas importadas precisam de fonte e unidades consistentes. Vídeo/áudio e Azure
  continuam nos módulos dos outros integrantes; não há fusão automática dos seus
  resultados com este detector nesta versão.

## Roteiro de demonstração (aproximadamente 3 minutos)

1. Apresentar as três frentes e informar que os dados são sintéticos.
2. Reproduzir o histórico e mostrar aquecimento, mudança e alertas automáticos.
3. Mostrar os gráficos, a prescrição alterada e a evidência de um alerta.
4. Com `ALERT_WEBHOOK_URL` apontando para um canal do Teams, mostrar a mensagem
   chegando ao canal da equipe e o status "Enviado à equipe" no alerta. Marcar
   um alerta como visto para mostrar persistência.
5. Repetir o envio: sem duplicação. Baixar o relatório e apresentar as métricas,
   incluindo a limitação nas alterações sutis.
6. No vídeo do grupo, conectar esse trecho às demonstrações de áudio/vídeo e Azure;
   o limite de 15 minutos é do vídeo completo, não apenas deste módulo.
