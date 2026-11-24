# Teste integrado da Fase 4

## Validação da entrega final em 23/11/2026

O Grupo 69 processou no Streamlit um WAV sintético de sete segundos em português e um MP4 de uma repetição de agachamento do REHAB24-6. A análise real pelo Azure Speech to Text transcreveu o relato de dor no joelho e cansaço. O Azure AI Language retornou sentimento negativo e seis frases-chave. O projeto identificou dois termos críticos, `dor` e `cansaco`, e sinalizou um indicador acústico experimental de possível disartria.

O YOLOv8 Pose gerou o MP4 anotado. O classificador de ensaio indicou execução possivelmente correta, e as regras de pose apontaram amplitude reduzida e assimetria. O caso `DEMO-G69-VIDEO-01` reuniu a nota, o áudio e o vídeo na janela de `2026-01-01T00:00:00Z` a `2026-01-01T01:55:00Z`. A aplicação retornou **Revisão prioritária** com seis alertas associados a 51 observações sintéticas de sinais vitais, atividade e prescrição.

O backend registrou o envio dos seis alertas ao canal de teste do Slack. O alerta de oxigenação `0f79839b-cfab-46d7-bfb8-9e445cfb2ffb` foi conferido no caso, na mensagem no Slack e no painel, onde a marcação como visto ficou registrada. A saturação observada no exemplo sintético foi de 89,8%, ante a mediana de 97,0%. Na tomada editada do painel, o replay usa o histórico já processado desse paciente.

A suíte da Fase 4 com o WAV sintético e ambos os serviços Azure ativos terminou com **129 testes aprovados e um ignorado**. A avaliação permanece experimental e usa monitoramento sintético e gravações de demonstração.

## Ensaio integrado anterior em 23/11/2026

O ensaio HTTP enviou as 51 observações sintéticas uma a uma para a aplicação principal. O detector registrou seis alertas, o receptor HTTP local recebeu os mesmos seis IDs e a API confirmou a marcação de um alerta como visto. O caso multimodal anexou esses seis alertas e retornou **Revisão prioritária**. Os achados de mídia desse cenário eram simulados.

O ensaio também enviou uma repetição real de agachamento do REHAB24-6 (Camera17) para a API de anotação. O YOLOv8 estimou pose em 121 de 121 frames. A rota devolveu um MP4 anotado compatível com navegador. Foram sinalizadas amplitude reduzida e assimetria segundo as regras experimentais. Essa repetição tem rótulo de execução correta no dataset, o que ilustra a diferença entre o rótulo e as regras de desvios.

Para testar a classificação, o script oficial treinou um artefato de ensaio restrito a agachamento, com 113 repetições de cinco participantes. No participante de teste, acertou 18 de 23 repetições, mesmo total do baseline majoritário. O recall de repetições incorretas foi de 20%. Pelo endpoint multimodal, um trecho desse participante foi classificado com YOLO e o artefato treinado, e o pipeline de áudio foi executado com respostas HTTP do Azure simuladas. O caso associou seis alertas e retornou **Revisão prioritária**. Os arquivos de ensaio e as métricas estão em `tmp/fase04-ensaio/` no checkout local.

| Suíte | Aprovados | Falhas | Ignorados |
|---|---:|---:|---:|
| Fase 4 | 127 | 0 | 3 |
| Repositório completo | 195 | 7 | 3 |

Os testes ignorados foram dois de Azure real e um de embeddings. As sete falhas restantes são do fallback do assistente da Fase 3, já observadas no checkout antes desta implementação. Azure Speech, Azure AI Language e a entrega ao canal do grupo dependem das credenciais e do webhook de teste. O ensaio usou somente o receptor local.

O teste de contrato do Azure Speech agora usa o formato detalhado da resposta, no qual o texto pode chegar em `NBest[0].Display`. O cliente envia o tipo de conteúdo com codec PCM e taxa de 16 kHz. [Referência da Microsoft](https://learn.microsoft.com/azure/ai-services/speech-service/rest-speech-to-text-short)

A API rejeita WAV com codificação incompatível e arquivos de outro formato renomeados como `.wav` antes de chamar o Azure. O painel de anomalias exibe o ID que também aparece no relatório multimodal e na mensagem do webhook, permitindo verificar a entrega na gravação.

Ambiente de validação: Python 3.11.15, FastAPI 0.165.0, Pydantic 2.14.4, Uvicorn 0.54.0, Streamlit 1.67.0, NumPy 1.26.4, scikit-learn 1.6.1, OpenCV 4.11.0.86, PyArrow 23.0.1, Ultralytics 8.4.216, PyTorch 2.15.0 e pytest 9.2.0. O PyArrow 23.0.1 foi fixado em `requirements.txt` para compatibilidade com o NumPy do projeto.

## Histórico da validação em 10/11/2026

### Atualização após correções

Depois desta rodada:

- A regra de proteção de saída (`backend/assistant/guardrails.py`) foi corrigida.
  A alternativa vazia foi removida, e a negação de diagnóstico que o comentário
  descrevia foi implementada. Ela só descarta a frase quando a frase é apenas a
  negação. Os testes de `test_guardrails.py` passam todos.
- O ponto 3 ganhou entrega do alerta por webhook (Teams/Slack), painel com
  atualização automática e tolerância de cadência no movimento, com 3 testes novos.
- O envio do alerta passou a usar uma fila persistente (`notification_outbox`),
  com recuperação após interrupção, coberta por `test_notification_recovery.py`
  ([notification_recovery_tests.json](results/notification_recovery_tests.json)).

Suíte completa após as correções: **188 passaram, 9 falharam, 3 ignorados**.
As 9 falhas restantes são as dos itens 2 e 3 abaixo (contrato do fallback no
`test_workflow.py`, `test_llm_adapter.py` e contagem em `test_quality.py`). Elas
dependem de decidir o comportamento esperado do assistente, não de credenciais.

### Resultado da rodada original

A Fase 4 passou nos testes disponíveis, incluindo o módulo de anomalias montado
na aplicação principal. A solução completa **não está com a suíte aprovada**:
existem falhas no assistente da Fase 3 e faltam recursos para validar inferência
real de vídeo, Azure e LLM.

| Grupo | Passaram | Falharam | Ignorados |
|---|---:|---:|---:|
| Fase 4, incluindo integração e interface | 111 | 0 | 3 |
| Assistente da Fase 3 | 48 | 19 | 0 |
| Testes das fases anteriores restantes | 8 | 0 | 0 |
| **Total** | **167** | **19** | **3** |

Resultado estruturado, nomes de todos os testes com falha e versões do ambiente:
[integration_tests.json](results/integration_tests.json).

### O que foi exercitado

1. Suíte completa `tests`, sem ocultar falhas anteriores.
2. Inicialização e OpenAPI de `backend.main:app`, com as rotas de anomalias,
   assistente, áudio, vídeo e anotação na mesma aplicação.
3. Envio de 51 observações sintéticas; geração de seis alertas; consulta de
   histórico; marcação como visto; reenvio sem duplicação.
4. Navegação pelas seis telas Streamlit, sem exceções, e reprodução de anomalias
   usando a aplicação principal como destino (não apenas o servidor independente).
5. Pipeline real de áudio com WAV sintético de dois segundos: upload, leitura,
   características acústicas, termos críticos, relatório e remoção do temporário.
   Apenas os clientes Azure foram simulados nesse teste de sucesso. Os demais
   testes de áudio/vídeo também usam substitutos explicitamente definidos nas suítes.
6. Falta de Azure e classificador de vídeo retornando HTTP 503, sem derrubar a
   aplicação nem interromper os alertas. Assistente sem LLM retornando síntese
   determinística com indicação de revisão humana.

Além do TestClient, foi iniciado um processo Uvicorn real em endereço de loopback,
em porta livre, com banco de teste separado. OpenAPI e anomalias responderam HTTP
200; áudio sem credenciais e vídeo sem classificador responderam 503; o assistente
retornou fallback determinístico. Os seis alertas continuaram acessíveis depois
dessas chamadas. O servidor foi encerrado ao final. Nenhum serviço externo foi
chamado. Evidência: [integration_http_smoke.json](results/integration_http_smoke.json).

### Falhas encontradas no assistente existente

Os arquivos `backend/assistant/`, `tests/assistant/` e
`llm_finetuning/context_examples.py` não têm alterações em relação ao HEAD desta
verificação. As falhas não foram introduzidas por uma edição desses arquivos na
implementação do ponto 3. Isso não atribui a origem histórica nem substitui uma
comparação em outro ambiente.

#### 1. Defeito na regra de proteção de saída

Dez testes de `test_guardrails.py` falharam. Em
`backend/assistant/guardrails.py:187`, a expressão que remove negativas termina
com `|` antes do fechamento do grupo. Isso cria uma alternativa vazia, permitindo
que frases iniciadas por palavra sejam descartadas antes da checagem. Assim,
exemplos que deveriam ser bloqueados por prescrição ou diagnóstico fechado passam.

Esta é uma falha concreta de lógica, não apenas falta de credenciais. Corrigir a
expressão e executar novamente a bateria adversarial é necessário antes de
apresentar os guardrails como funcionando. Não foi alterada nesta rodada de teste.

#### 2. Contrato de fallback e expectativas dos testes

Sete testes de `test_workflow.py` esperam `blocked` ou `error`, enquanto o grafo
atual direciona certas respostas rejeitadas/falhas de geração para uma síntese
determinística com `status: success`. Parte dessa política está descrita nos
comentários do grafo. É preciso revisar o contrato e a segurança do conteúdo
retornado; simplesmente trocar as asserções para fazer a suíte passar não valida
o comportamento. Uma das falhas também se relaciona à proteção de saída acima.

Um teste de `test_llm_adapter.py` espera ignorar o provedor primário quando
`is_available()` é falso; a implementação atual tenta a geração mesmo assim.
Essa diferença também precisa de um contrato coerente entre código e testes.

#### 3. Quantidade de exemplos de fine-tuning

`test_quality.py` espera 60 exemplos, mas `build_context_examples()` devolve 130.
A implementação contém categorias adicionais. O teste está desalinhado com a
quantidade atual; antes de atualizá-lo, revisar também agrupamento e conteúdo dos
exemplos, não apenas substituir o número.

### Limitações da rodada original

- **Azure real:** faltam as variáveis de credenciais de Speech e Language. Os dois
  testes ao vivo foram ignorados; sucesso com substitutos não valida esses serviços.
- **Inferência de vídeo real:** não há classificador treinado em `backend/model/`
  e o peso YOLO esperado não está disponível na raiz. O endpoint foi validado nas
  condições cobertas pelos testes e na falha por ausência do classificador.
- **LLM local e embeddings reais:** o ambiente não tem torch/transformers. O teste
  de embeddings foi ignorado. Não se validou a qualidade da geração da LLM.
- **Classificador vocal persistido:** o artefato não está na pasta do modelo;
  os testes de classificador usam artefatos gerados para teste. A análise acústica
  e a montagem do relatório foram executadas.
- **Sensores e fusão multimodal:** movimentação continua simulada/importada. As
  modalidades compartilham a aplicação, mas não há fusão automática de resultados
  de áudio/vídeo no detector temporal.
- **Ambiente recomendado de produção:** esta rodada usou Python 3.14.3 com
  dependências compatíveis disponíveis (versões no JSON). O projeto recomenda
  Python 3.11, NumPy 1.26.4 e scikit-learn 1.6.1. Não se afirma compatibilidade de
  artefatos antigos com as versões mais recentes do ambiente de teste.

## Reproduzir a suíte

Na raiz, com as dependências instaladas:

```powershell
$env:ITEM1_PRELOAD = 'false'
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=tmp/integrated-suite-repeat --tb=short
```

`ITEM1_PRELOAD=false` evita tentar carregar o modelo pesado a cada fixture da
suíte. No smoke HTTP real, a inicialização foi verificada com preload habilitado:
a ausência da dependência foi tratada pelo fallback existente.

Os três testes adicionais reproduzíveis estão em
`tests/fase04/test_integrated_application.py`. Eles isolam os bancos em diretórios
temporários e não dependem de credenciais reais. Esta rodada acrescentou testes e
evidências; não modificou o comportamento dos módulos anteriores.

### Pendências apontadas em 10/11

1. Corrigir a regra de proteção de saída e reconciliar os contratos/testes do
   assistente; repetir a suíte até as falhas estarem resolvidas.
2. Disponibilizar os artefatos treinados e executar áudio/vídeo reais com as
   credenciais Azure do grupo, no ambiente recomendado.
3. Gravar a demonstração distinguindo análise real, simulação e fallback.
