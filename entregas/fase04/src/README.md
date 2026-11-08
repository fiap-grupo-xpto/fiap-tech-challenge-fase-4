# Experimentos

Scripts de pesquisa reproduziveis da Fase 4 ficam neste diretorio. Organize-os
por etapa (preprocessamento, estimativa de pose, classificacao e avaliacao) e
registre parametros e seeds. Ainda nao ha pipeline de treino implementado.

## `evaluate_audio_flags.py`

Harness offline da Tarefa 2. Alem das caracteristicas acusticas manuais, o flag
`--include-embeddings` adiciona uma sonda de embeddings profundos
(`facebook/wav2vec2-base`, media temporal, 768 dimensoes) para comparar uma
representacao aprendida com as caracteristicas manuais. Requer `torch` e
`transformers`; sem eles a sonda e registrada como `skipped` e o restante da
avaliacao segue normalmente. Em CPU o ICBHI (920 gravacoes de som respiratorio)
fica caro com o flag ligado; use `--limit` para uma subamostra comparavel entre
as duas representacoes.

```powershell
python entregas/fase04/src/evaluate_audio_flags.py --dataset voiced --include-embeddings
python entregas/fase04/src/evaluate_audio_flags.py --dataset icbhi --include-embeddings
```
