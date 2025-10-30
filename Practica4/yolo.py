import subprocess

# Lista de configuraciones de entrenamiento
train_configs = [
    {"name": "T2", "imgsz": 416, "batch": 4, "epochs": 100, "lr0": 0.001},
    {"name": "T3", "imgsz": 512, "batch": 4, "epochs": 100, "lr0": 0.001},
    {"name": "T4", "imgsz": 640, "batch": 4, "epochs": 50,  "lr0": 0.001},
    {"name": "T5", "imgsz": 512, "batch": 8, "epochs": 60,  "lr0": 0.001},
    {"name": "T6", "imgsz": 416, "batch": 4, "epochs": 100, "lr0": 0.001},
    {"name": "T7", "imgsz": 768, "batch": 2, "epochs": 25,  "lr0": 0.001},
    {"name": "T8", "imgsz": 640, "batch": 4, "epochs": 80,  "lr0": 0.001},
    {"name": "T9", "imgsz": 512, "batch": 4, "epochs": 100, "lr0": 0.01},
]

# Ruta del modelo base y del archivo data.yaml
model_path = "yolo11n.pt"
data_yaml = "data.yaml"
device = 0  # GPU a usar (0 = primera GPU)

# Ejecutar los entrenamientos secuencialmente
for cfg in train_configs:
    print(f"\n=== Iniciando entrenamiento {cfg['name']} ===\n")
    cmd = [
        "yolo", "detect", "train",
        f"model={model_path}",
        f"data={data_yaml}",
        f"imgsz={cfg['imgsz']}",
        f"batch={cfg['batch']}",
        f"device={device}",
        f"epochs={cfg['epochs']}",
        f"lr0={cfg['lr0']}"
    ]
    subprocess.run(cmd, check=True)

print("\nTodos los entrenamientos han finalizado.")
