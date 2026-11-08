# 📖 Documentação da API - Tech Challenge Fase 2

Esta documentação descreve de forma estática os endpoints expostos pelo backend do sistema de apoio ao diagnóstico de câncer de pulmão. A API é construída com **FastAPI** e serve aos modelos otimizados via Algoritmo Genético (AG).

A API roda por padrão na porta `8888` e pode ser acessada localmente em `http://localhost:8888`.

---

## 1. Endpoint: Análise Tabular (Pré-Triagem)

Analisa dados demográficos e sintomas dos pacientes a partir de um arquivo CSV e estima a probabilidade de risco, integrando-se à LLM para gerar uma interpretação em linguagem natural.

* **Rota:** `/analyze-tabular`
* **Método:** `POST`
* **Content-Type:** `multipart/form-data`

### 1.1 Parâmetros de Entrada
A requisição deve enviar um arquivo no campo `csv_file`.

| Campo | Tipo | Descrição |
|---|---|---|
| `csv_file` | `file` (CSV) | Arquivo CSV contendo as colunas de sintomas descritas abaixo. |

#### Formato Esperado do CSV (Colunas)
O arquivo CSV enviado deve conter obrigatoriamente as seguintes colunas (com valores binários `1` para Não e `2` para Sim, exceto idade e gênero):

1. `GENDER`: Gênero (`M` ou `F`).
2. `AGE`: Idade (inteiro).
3. `SMOKING`: Fumante (`1` ou `2`).
4. `YELLOW_FINGERS`: Dedos amarelados (`1` ou `2`).
5. `ANXIETY`: Ansiedade (`1` ou `2`).
6. `PEER_PRESSURE`: Pressão de grupo (`1` ou `2`).
7. `CHRONIC_DISEASE`: Doença crônica (`1` ou `2`).
8. `FATIGUE`: Fadiga (`1` ou `2`).
9. `ALLERGY`: Alergia (`1` ou `2`).
10. `WHEEZING`: Chiado no peito (`1` ou `2`).
11. `ALCOHOL_CONSUMING`: Consumo de álcool (`1` ou `2`).
12. `COUGHING`: Tosse constante (`1` ou `2`).
13. `SHORTNESS_OF_BREATH`: Falta de ar (`1` ou `2`).
14. `SWALLOWING_DIFFICULTY`: Dificuldade de engolir (`1` ou `2`).
15. `CHEST_PAIN`: Dor no peito (`1` ou `2`).

---

### 1.2 Exemplo de Resposta de Sucesso (`200 OK`)
Se o arquivo CSV for processado e inferido corretamente pelos classificadores (por padrão, KNeighborsClassifier otimizado pelo AG), a API retorna o seguinte JSON:

```json
{
  "status": "success",
  "model_name": "KNeighborsClassifier",
  "threshold": 0.1,
  "interpretation_engine": "gemini-2.5-flash",
  "results": [
    {
      "row_index": 0,
      "disease_detected": true,
      "probability": 0.8888,
      "risk_level": "alto",
      "llm_interpretation": "Resumo: O modelo KNeighborsClassifier indicou a possibilidade de risco alto de câncer de pulmão baseado nos sintomas demográficos informados.\n\nRisco: Alto\n\nJustificativa: O paciente apresentou probabilidade estimada de 88.88%, que supera o limiar clínico estabelecido de 10.00%.\n\nRecomendação: Encaminhamento imediato para avaliação clínica presencial e exames de imagem adicionais.\n\nObservação: Este resultado é apenas um apoio à decisão clínica e não substitui a consulta médica presencial."
    }
  ]
}
```

*Nota: Para evitar estouro de cotas da LLM, o backend apenas gera explicações detalhadas (`llm_interpretation`) para os **5 primeiros registros** do arquivo CSV (configurável pela variável `MAX_LLM_INTERPRETATIONS`). Para as linhas subsequentes, o campo conterá uma mensagem padrão informativa.*

---

### 1.3 Exemplo de Resposta de Erro (`200 OK` com payload de erro)
Caso o arquivo enviado não seja um CSV válido ou faltem colunas obrigatórias, a API retornará:

```json
{
  "status": "error",
  "message": "CSV missing required columns: ['GENDER', 'AGE']"
}
```

---

### 1.4 Comando para teste (curl)
```bash
curl -X POST -F "csv_file=@/caminho/para/seu/arquivo.csv" http://localhost:8888/analyze-tabular
```

---

## 2. Endpoint: Análise de Imagens (Visão Computacional)

Processa múltiplas imagens de ressonância magnética (exame de imagem), prediz a presença de achados suspeitos através da CNN (baseada em EfficientNetB0 otimizada via AG) e retorna a análise da LLM.

* **Rota:** `/analyze-images`
* **Método:** `POST`
* **Content-Type:** `multipart/form-data`

### 2.1 Parâmetros de Entrada
Aceita uma lista de arquivos de imagem e o valor do threshold clínico para corte de classificação.

| Campo | Tipo | Descrição |
|---|---|---|
| `files` | `file` (Lista) | Um ou mais arquivos de imagem (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`). |
| `probability_threshold` | `float` (Form) | Limiar de probabilidade usado para corte de diagnóstico (ex: `0.10` para priorizar sensibilidade/recall). |

---

### 2.2 Exemplo de Resposta de Sucesso (`200 OK`)
Retorna uma lista contendo a inferência do modelo de imagem e a interpretação em linguagem natural para cada arquivo processado:

```json
{
  "results": [
    {
      "filename": "exame_pulmao_paciente1.png",
      "status": "success",
      "file_path": "/app/uploads/exame_pulmao_paciente1.png",
      "disease_detected": true,
      "probability": 0.9421,
      "threshold_used": 0.1,
      "llm_interpretation": "Resumo: Análise automática da imagem de ressonância identificou presença de achado suspeito maligno.\n\nRisco: Alto\n\nJustificativa: A probabilidade estimada pela CNN foi de 94.21%, superando o limite clínico de 10.00%.\n\nRecomendação: Avaliação clínica e radiológica imediata com contraste para complementação diagnóstica.\n\nObservação: Este é um relatório preliminar automático de apoio e não substitui o laudo radiológico oficial."
    }
  ]
}
```

---

### 2.3 Exemplo de Resposta de Erro (`200 OK` por imagem)
Caso uma das imagens enviadas possua extensão não suportada, a API retorna o status de erro para aquele arquivo individual:

```json
{
  "results": [
    {
      "filename": "documento_invalido.pdf",
      "status": "error",
      "message": "Invalid file type. Only ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') are allowed"
    }
  ]
}
```

---

### 2.4 Comando para teste (curl)
```bash
curl -X POST \
  -F "files=@imagem1.png" \
  -F "files=@imagem2.jpg" \
  -F "probability_threshold=0.10" \
  http://localhost:8888/analyze-images
```

---

## 3. Endpoint: Análise de Áudio de Reabilitação (Fase 4)

Transcreve a fala do paciente durante a sessão de reabilitação (Azure Speech to Text), aplica análise de linguagem (Azure AI Language) e extrai métricas acusticas heurísticas, retornando um relatório experimental de apoio à triagem. As métricas acusticas combinam características temporais/energéticas e espectrais (centroide, rolloff, fluxo espectral e taxa de cruzamento por zero) com marcadores de qualidade vocal calculados via Praat (`jitter_local`, `shimmer_local`, `hnr`), que alimentam o achado de triagem `possivel_disartria` (disartria). O resultado **não é diagnóstico** e exige revisão do fisioterapeuta.

Quando o classificador reutilizável (`backend/model/voice_alteration_classifier.joblib`, treinado por `entregas/fase04/src/evaluate_audio_flags.py --dataset voiced`) está presente, o relatório inclui um bloco `voice_alteration` com a probabilidade estimada de alteração vocal; sem o artefato o campo é `null` e apenas as regras heurísticas são retornadas.

* **Rota:** `/analyze-rehabilitation-audio`
* **Método:** `POST`
* **Content-Type:** `multipart/form-data`

### 3.1 Parâmetros de Entrada
A requisição deve enviar o arquivo de áudio (`.wav`, PCM 16 kHz mono) e, opcionalmente, o idioma. A transcrição usa o endpoint REST de áudio curto, limitado a 60 segundos por arquivo.

| Campo | Tipo | Descrição |
|---|---|---|
| `audio` | `file` (áudio) | Arquivo de áudio da sessão no formato `.wav` (PCM 16 kHz mono, máx. 60 s). |
| `language` | `string` (Form) | Idioma da fala (ex: `pt-BR`). Vazio usa o padrão configurado. |

---

### 3.2 Exemplo de Resposta de Sucesso (`200 OK`)
Retorna a transcrição, os termos críticos detectados, o sentimento e as métricas acusticas com os achados de triagem:

```json
{
  "status": "success",
  "transcript": "repita o movimento, estou com cansaco",
  "segments": [
    {
      "start": 0.0,
      "end": 0.5,
      "text": "repita",
      "confidence": 0.95
    }
  ],
  "critical_terms": [
    {
      "term": "cansaco",
      "category": "sintoma",
      "count": 1,
      "severity": "warning"
    }
  ],
  "sentiment": {
    "label": "negative",
    "positive": 0.1,
    "neutral": 0.2,
    "negative": 0.7
  },
  "azure_key_phrases": ["dor", "movimento"],
  "azure_entities": [
    {
      "text": "joelho",
      "category": "BodyPart"
    }
  ],
  "acoustic_findings": [
    {
      "code": "possivel_fadiga",
      "severity": "warning",
      "message": "Pausas prolongadas combinadas com baixa energia de voz podem indicar fadiga durante o esforco; sugerida revisao do fisioterapeuta.",
      "measured": 0.01,
      "expected": 0.05
    },
    {
      "code": "possivel_disartria",
      "severity": "warning",
      "message": "Instabilidade vocal (jitter/shimmer elevados ou relacao harmonico-ruido baixa) pode indicar disartria; resultado experimental de apoio, nao diagnostico.",
      "measured": 2.14,
      "expected": 1.04
    }
  ],
  "metrics": {
    "duration_seconds": 30.0,
    "rms_energy": 0.01,
    "voiced_ratio": 0.3,
    "pause_ratio": 0.7,
    "mean_pause_seconds": 1.5,
    "speaking_rate": 0.5,
    "pitch_variability": 0.1,
    "spectral_centroid_hz": 612.4,
    "spectral_rolloff_hz": 1840.0,
    "spectral_flux": 12.35,
    "zero_crossing_rate": 0.081,
    "jitter_local": 2.14,
    "shimmer_local": 4.52,
    "hnr": 15.3
  },
  "voice_alteration": {
    "probability_pathological_voice": 0.82,
    "threshold": 0.5,
    "flagged": true,
    "model": "HistGradientBoostingClassifier",
    "dataset": "VOICED",
    "validation_roc_auc": 0.6146,
    "feature_version": "rehab-audio-acoustic-v2"
  },
  "risk_flags": ["possivel_fadiga", "sintoma"],
  "requires_human_review": true,
  "warning": "Resultado experimental de apoio a reabilitacao; nao e diagnostico e exige revisao do fisioterapeuta.",
  "method": "Azure Speech to Text + Azure AI Language + regras acusticas heuristicas (nao validadas clinicamente).",
  "report_version": "1.0"
}
```

---

### 3.3 Exemplos de Resposta de Erro
A rota mapeia falhas de validação e de integração para códigos HTTP específicos:

```json
{
  "detail": "Only .wav audio files are supported."
}
```
`415 Unsupported Media Type` — extensão de arquivo não suportada.

```json
{
  "detail": "Audio exceeds the upload limit of 25 MB."
}
```
`413 Payload Too Large` — arquivo acima de `MAX_REHAB_AUDIO_BYTES`.

```json
{
  "detail": "Uploaded audio is empty."
}
```
`400 Bad Request` — arquivo vazio.

```json
{
  "detail": "Audio exceeds the 60 second limit for this analysis."
}
```
`422 Unprocessable Entity` — áudio acima de `MAX_REHAB_AUDIO_SECONDS`.

```json
{
  "detail": "Credenciais do Azure nao configuradas para analise de audio."
}
```
`503 Service Unavailable` — credenciais do Azure ausentes ou falha na integração.

---

### 3.4 Comando para teste (curl)
```bash
curl -X POST \
  -F "audio=@sessao_reabilitacao.wav" \
  -F "language=pt-BR" \
  http://localhost:8888/analyze-rehabilitation-audio
```
