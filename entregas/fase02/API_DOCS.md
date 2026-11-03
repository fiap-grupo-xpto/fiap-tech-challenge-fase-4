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
