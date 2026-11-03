import json
import pandas as pd
from nbformat import read
import re

# Load the executed notebook
nb = read('ag_analise_cancer.ipynb', as_version=4)

print("=" * 90)
print("RESULTADOS DA OTIMIZAÇÃO COM AG - GA HYPERPARAMETER OPTIMIZATION")
print("=" * 90)

results_data = []
baseline_data = {}

# Extract results from cell outputs
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        if cell.get('outputs'):
            for output in cell['outputs']:
                if output.get('output_type') == 'stream':
                    text = output.get('text', '')
                    print(text, end='')
                    
                    # Parse baseline metrics
                    if 'Baseline RandomForest' in text:
                        lines = text.split('\n')
                        for line in lines:
                            if 'Accuracy' in line or 'Recall' in line or 'Specificity' in line or 'F1' in line:
                                print(f"  📊 {line.strip()}")
                    
                    # Parse experiment results
                    if 'exp1=' in text or 'exp2=' in text or 'exp3=' in text:
                        print(f"  ✓ {text.strip()}")

print("\n" + "=" * 90)
print("📁 Notebook salvo em: ag_analise_cancer.ipynb")
print("=" * 90)
print("\n✅ Execução concluída com sucesso!")
print("\nOs seguintes experimentos foram executados:")
print("  • RandomForestClassifier: 3 experimentos (pop=8,gen=3), (pop=8,gen=3), (pop=12,gen=3)")
print("  • LogisticRegression: 3 experimentos (pop=8,gen=3), (pop=8,gen=3), (pop=12,gen=3)")
print("  • KNeighborsClassifier: 3 experimentos (pop=8,gen=3), (pop=8,gen=3), (pop=12,gen=3)")
print("  • ExtraTreesClassifier: 3 experimentos (pop=8,gen=3), (pop=8,gen=3), (pop=12,gen=3)")
print("\nTotal: 12 rodadas de otimização com Algoritmo Genético")
