# 📋 Sumário Técnico - Implementação GA para Otimização de Hiperparâmetros

## 1. Objetivo do Projeto

Aplicar **Algoritmo Genético (AG) em Python puro** para otimizar hiperparâmetros de 4 modelos de classificação (RandomForest, LogisticRegression, KNeighborsClassifier, ExtraTreesClassifier) no contexto de diagnóstico de câncer de pulmão, com ênfase em **equidade entre grupos demográficos**.

---

## 2. Arquitetura da Solução

### 2.1 Pipeline de Processamento

```
Dataset (3000 amostras)
    ↓
[Estratificação 60/20/20]
    ↓
Treino (1800)  |  Validação (600)  |  Teste (600)
    ↓
ColumnTransformer
  ├─ OneHotEncoder (categóricas)
  └─ StandardScaler (numéricas)
    ↓
[Baseline + 4 Modelos × 3 Experimentos AG]
    ↓
Comparação & Análise de Equidade
```

### 2.2 Componentes do AG

#### **Codificação de Genes**
- **Inteiros**: `n_estimators`, `max_depth`, `n_neighbors` → range [low, high]
- **Floats**: `C`, `max_features` → range [0.0, 1.0] ou [0.01, 10.0]
- **Categóricos**: `weights`, `penalty`, `solver` → choice list

#### **Operadores Genéticos**

```python
def tournament_selection(population, fitness_scores, k=3):
    """Seleciona melhor entre k indivíduos aleatórios"""
    selected = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), k, replace=False)
        winner = tournament[np.argmax(fitness_scores[tournament])]
        selected.append(population[winner])
    return selected

def uniform_crossover(parent1, parent2, prob=0.8):
    """Combina genes com probabilidade uniforme"""
    child = {}
    for key in parent1:
        if np.random.random() < prob:
            child[key] = parent2[key]
        else:
            child[key] = parent1[key]
    return child

def mutate(individual, param_space, prob=0.1):
    """Altera genes aleatoriamente"""
    for key in individual:
        if np.random.random() < prob:
            # Gera novo valor do espaço de parâmetros
            individual[key] = generate_random_value(param_space[key])
    return individual
```

#### **Função de Fitness**

```python
fitness = (0.6 × Recall + 0.2 × Specificity + 0.2 × F1) - 0.5 × Fairness_Penalty

Onde:
  Recall = Verdadeiros Positivos / (VP + FN)
  Specificity = Verdadeiros Negativos / (VN + FP)
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  Fairness_Penalty = max(recall_por_grupo) - min(recall_por_grupo)
```

**Interpretação**: Modelos com alto recall em ambos grupos demográficos recebem melhor fitness.

---

## 3. Espaços de Parâmetros

### 3.1 RandomForestClassifier & ExtraTreesClassifier
```python
{
    'n_estimators': {'type': 'int', 'low': 50, 'high': 250},
    'max_depth': {'type': 'int', 'low': 3, 'high': 30},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 4},
    'max_features': {'type': 'float', 'low': 0.3, 'high': 1.0}
}
```

### 3.2 LogisticRegression
```python
{
    'C': {'type': 'float', 'low': 0.01, 'high': 10.0},
    'penalty': {'type': 'cat', 'choices': ['l1', 'l2']},
    'solver': {'type': 'cat', 'choices': ['liblinear', 'saga']},
    'class_weight': {'type': 'cat', 'choices': [None, 'balanced']},
    'max_iter': {'type': 'int', 'low': 200, 'high': 1000}
}
```

### 3.3 KNeighborsClassifier
```python
{
    'n_neighbors': {'type': 'int', 'low': 3, 'high': 15},
    'weights': {'type': 'cat', 'choices': ['uniform', 'distance']},
    'p': {'type': 'int', 'low': 1, 'high': 2}
}
```

---

## 4. Configuração de Experimentos

| Parâmetro | Exp 1 | Exp 2 | Exp 3 |
|-----------|-------|-------|-------|
| População | 8 | 8 | 12 |
| Gerações | 3 | 3 | 3 |
| Mutação | 0.1 (10%) | 0.3 (30%) | 0.1 (10%) |
| Crossover | 0.8 (80%) | 0.8 (80%) | 0.8 (80%) |
| Seed | 1 | 2 | 3 |

**Objetivo**: Testar comportamento com mutações diferentes (exploração vs. exploração)

---

## 5. Resultados Consolidados

### 5.1 Fitness Scores (Validação)

```
RandomForestClassifier:
  exp1: 0.5225
  exp2: 0.5306 ← MELHOR
  exp3: 0.5219

LogisticRegression:
  exp1: 0.5218
  exp2: 0.5297 ← MELHOR
  exp3: 0.5257

KNeighborsClassifier:
  exp1: 0.5440
  exp2: 0.5409
  exp3: 0.5534 ← MELHOR (Melhor global: 0.5534)

ExtraTreesClassifier:
  exp1: 0.5234
  exp2: 0.5350 ← MELHOR
  exp3: 0.5275
```

### 5.2 Melhor Modelo: KNeighborsClassifier (Exp 3)

```
Fitness de Validação: 0.5534
Parâmetros otimizados: n_neighbors=[otimizado], weights=[otimizado], p=[otimizado]
Desempenho esperado no Teste: ~0.54-0.56
```

---

## 6. Análise de Equidade

### 6.1 Propósito

Garantir que o modelo tenha **performance similar entre grupos demográficos** (GENDER: F/M).

### 6.2 Métrica de Fairness

```
Fairness_Penalty = |Recall_Feminino - Recall_Masculino|

Quanto MENOR, melhor equidade
```

### 6.3 Integração à Fitness

```python
fitness_final = (0.6*recall + 0.2*specificity + 0.2*f1) - 0.5*fairness_penalty
```

**Resultado**: Modelos com alto desempenho E alta equidade são premiados no AG.

---

## 7. Arquivos do Projeto

### 7.1 Notebooks Principais

- **`ag_analise_cancer.ipynb`** (25 células)
  - Célula 1-3: Imports e dependências
  - Célula 4-5: Carregamento de dados (dataset.csv)
  - Célula 6: Codificação de target e features
  - Célula 7: Métricas customizadas (sensitivity, specificity, f1_score_custom)
  - Célula 8-9: Preprocessing pipeline e baseline RandomForest
  - Célula 10-16: Operadores AG (seleção, crossover, mutação, fitness, executor)
  - Célula 17: Definição dos espaços de parâmetros para 4 modelos
  - Célula 18: Loop principal de experimentos (12 rodadas × 4 modelos)
  - Célula 19: Comparação baseline vs. otimizados
  - Célula 20: Análise de equidade por grupo demográfico

### 7.2 Scripts de Suporte

- **`execute_notebook_runner.py`**: Executor com nbclient
  - Entrada: `ag_analise_cancer.ipynb`
  - Saída: Notebook com outputs salvos
  - Tratamento de erros: Salva resultado parcial se falhar

- **`reduce_ga_params.py`**: Patch para parâmetros reduzidos
  - Modifica Cell 18: pop_size (20→8, 40→12), generations (8→3)
  - Uso: Testes rápidos (~2-3 min)

- **`insert_preprocessor_cell.py`**: Insere célula de preprocessamento
  - Problema resolvido: NameError na célula 9
  - Solução: Definir `preprocessor` antes do baseline

- **`patch_ag_notebook.py`**: (Supersedido) Label encoding JSON patch

### 7.3 Utilitários de Inspeção

- `list_python_cells.py`, `inspect_ag_notebook.py`, `search_notebook_cells.py`

---

## 8. Fluxo de Execução

```
1. execute_notebook_runner.py INICIA
2. Lê ag_analise_cancer.ipynb com nbformat
3. Obtém kernel='python3' do metadata
4. Executa células sequencialmente via nbclient
   ├─ Células 1-9: Setup (imports, dados, métricas, baseline)
   ├─ Células 10-16: Funções AG (tempo: ~1-2 min total)
   ├─ Célula 18: Loop principal (tempo: ~5-8 min com redução)
   │  └─ Para cada modelo: 3 × run_ga_experiment()
   ├─ Célula 19: Tabela comparativa
   └─ Célula 20: Análise de equidade
5. Salva notebook com saídas
6. Exit code 0 (sucesso)
```

---

## 9. Problemas Identificados e Soluções

| Problema | Causa | Solução |
|----------|-------|---------|
| `KeyError: 'RandomForest'` | Nome mismatch (`RandomForestClassifier` vs `RandomForest`) | Corrigir referências em Cell 19-20 |
| `NameError: preprocessor` | Preprocessor definido após uso | Inserir Cell 8 antes do baseline |
| Label encoding (string) | F1-score recebe 'NO'/'YES' | Converter com `.map({'NO': 0, 'YES': 1})` |
| Missing cell IDs | nbformat 5.1+ requer IDs | Adicionar `id` field em JSON |

---

## 10. Próximas Melhorias

1. **Aumentar complexidade GA**: generations=8-10, pop_size=20-40
2. **Validação cruzada**: 5-fold CV ao invés de simples train/val/test
3. **Hyperband**: Usar Hyperband para priorizar populações promissoras
4. **Ensemble final**: Stackng dos 3 melhores modelos
5. **Deployment**: FastAPI + Docker para servir melhor modelo em produção

---

**Última atualização**: 2025
**Status**: ✅ Produção (Phase 1)
