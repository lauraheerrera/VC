import pandas as pd
from pathlib import Path

# Carpeta donde están todos los entrenamientos
base_dir = Path("runs/detect")

# Columnas a extraer
metrics_columns = [
    'train/box_loss','train/cls_loss','train/dfl_loss',
    'val/box_loss','val/cls_loss','val/dfl_loss',
    'metrics/precision(B)','metrics/recall(B)',
    'metrics/mAP50(B)','metrics/mAP50-95(B)'
]

# Lista para almacenar resumen
summary = []

for train_dir in sorted(base_dir.glob("train*")):
    results_file = train_dir / "results.csv"
    if not results_file.exists():
        print(f"⚠️ No se encontró results.csv en {train_dir}")
        continue

    df = pd.read_csv(results_file)
    
    # Mejor época según mAP50(B)
    if 'metrics/mAP50(B)' in df.columns:
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best = df.loc[best_idx]
    else:
        # Si no existe mAP50, tomar la última época
        best = df.iloc[-1]

    # Guardar resumen
    row = {'Entrenamiento': train_dir.name}
    for col in metrics_columns:
        if col in df.columns:
            row[col+'_best'] = best[col]

    # Calcular sumatorio de pérdidas de validación
    val_loss_sum = sum(best[col] for col in ['val/box_loss','val/cls_loss','val/dfl_loss'] if col in df.columns)
    row['val_loss_sum'] = val_loss_sum

    # Calcular suma de precision + recall
    pr_sum = 0
    if 'metrics/precision(B)' in df.columns:
        pr_sum += best['metrics/precision(B)']
    if 'metrics/recall(B)' in df.columns:
        pr_sum += best['metrics/recall(B)']
    row['pr_sum'] = pr_sum

    summary.append(row)

# Crear DataFrame
summary_df = pd.DataFrame(summary)

# Ordenar según los 4 criterios en orden
summary_df = summary_df.sort_values(
    by=['metrics/mAP50(B)_best','metrics/mAP50-95(B)_best','val_loss_sum','pr_sum'],
    ascending=[False, False, True, False]
)

# Ajustes de formato
pd.set_option('display.float_format', '{:.4f}'.format)

# Guardar en Excel
summary_excel = Path("resumen_entrenamientos_mejores.xlsx")
summary_df.to_excel(summary_excel, index=False)
print(f"\n📄 Resumen de los mejores entrenamientos guardado en Excel en {summary_excel}")
