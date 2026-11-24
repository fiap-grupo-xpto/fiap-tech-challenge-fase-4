# FIAP Tech Challenge Fase 4 — Monitoramento multimodal em reabilitação

Protótipo do Grupo 69 que reúne notas de sessão, análise de voz e vídeo e alertas de monitoramento para revisão humana. A aplicação usa FastAPI e Streamlit.

> Protótipo experimental para pesquisa/ensino. Não substitui avaliação profissional e não foi validado para pacientes.

## Integrantes do Grupo 69

- Otoniel da Silva Isidoro (RM368069)
- André Roberto Figueiró (RM365608)
- Gustavo César de Souza (RM370800)
- Thales Ernane de Souza (RM372083)

## Documentação da Fase 4

- [Vídeo da apresentação no YouTube](https://youtu.be/bgRv314A1F8) · 14:59 · não listado.
- [Guia da Fase 4](entregas/fase04/README.md): treinamento, execução e testes.
- [Relatório Técnico Consolidado](entregas/fase04/RELATORIO_TECNICO_CONSOLIDADO.md): escopo, metodologia, resultados e limitações.
- [Instruções do dataset](entregas/fase04/data/README.md): fonte, licença e download.
- [Detecção de anomalias](entregas/fase04/ANOMALIAS.md) para sinais vitais, prescrições, atividade e envio de alertas.
- [Validação integrada](entregas/fase04/TESTE_INTEGRADO.md) com os resultados do Azure, do caso multimodal e da entrega ao Slack.

## Como executar

Use Python 3.11 em um ambiente virtual. Instale as dependências e inicie o backend e a interface em terminais separados.

```sh
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --port 8888
python -m streamlit run frontend/app.py
```

Abra **Caso multimodal (Fase 4)** e clique em **Executar caso sintético**. A aplicação combina uma nota e resultados simulados de áudio e vídeo com alertas produzidos pelo detector. Consulte os mesmos alertas pelo identificador do paciente na tela **Detecção de anomalias**.

Para analisar mídias enviadas pelo usuário, configure Azure Speech e Azure AI Language no backend e prepare o classificador e os pesos de pose. Para demonstrar a entrega externa, configure o webhook de um canal de teste. O [guia da Fase 4](entregas/fase04/README.md) descreve as variáveis, o treino e os testes.

Na demonstração final, o Azure transcreveu e analisou um WAV sintético em português. O caso associou seis alertas às evidências de texto, áudio e vídeo e retornou **Revisão prioritária**. O alerta de oxigenação chegou ao Slack e foi marcado como visto no painel.

## Resultados do classificador de vídeo no conjunto de teste

| Métrica | Modelo | Baseline majoritário |
|---|---:|---:|
| Acurácia | 59,46% | 43,69% |
| F1 macro | 57,80% | 30,41% |

Detalhes por exercício e limitações no relatório técnico.
