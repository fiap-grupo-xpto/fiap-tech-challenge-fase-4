# FIAP Tech Challenge Fase 4 - Análise de Vídeos de Reabilitação

Repositório da **Fase 4 do Tech Challenge FIAP** (Grupo 69): estimativa de pose com YOLOv8 e classificação de repetições de exercícios de reabilitação (correta/incorreta) usando o dataset REHAB24-6.

> Protótipo experimental para pesquisa/ensino. Não substitui avaliação profissional e não foi validado para pacientes.

## Integrantes do Grupo 69

- Otoniel da Silva Isidoro (RM368069)
- André Roberto Figueiró (RM365608)
- Gustavo César de Souza (RM370800)
- Thales Ernane de Souza (RM372083)

## Documentação da Fase 4

- [Guia da Fase 4](entregas/fase04/README.md): treinamento, execução e testes.
- [Relatório Técnico Consolidado](entregas/fase04/RELATORIO_TECNICO_CONSOLIDADO.md): escopo, metodologia, resultados e limitações.
- [Instruções do dataset](entregas/fase04/data/README.md): fonte, licença e download.

## Como executar

Requer Python 3.11 e as dependências de `requirements.txt`.

```powershell
pip install -r requirements.txt
```

**Baixar o dataset** (verificação de checksum):

```powershell
py -3.11 entregas\fase04\download_dataset.py
```

**Treinar e avaliar:**

```powershell
py -3.11 entregas\fase04\src\train_classifier.py
```

**Testes automatizados:**

```powershell
py -3.11 -m pytest tests\fase04 -q
```

**Aplicação** (backend FastAPI + frontend Streamlit; o modelo treinado deve existir em `backend/model/`):

```powershell
uvicorn backend.main:app --port 8888
streamlit run frontend/app.py
```

Na interface, selecione **Análise de movimento (Fase 4)**. A API expõe `POST /analyze-rehabilitation-video`.

## Resultados (conjunto de teste)

| Métrica | Modelo | Baseline majoritário |
|---|---:|---:|
| Acurácia | 59,46% | 43,69% |
| F1 macro | 57,80% | 30,41% |

Detalhes por exercício e limitações no relatório técnico.
