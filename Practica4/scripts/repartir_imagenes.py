import os
import shutil
import random
import time

# El script se ejecuta dentro de "Practica4"
dataset_dir = os.path.join("TGC_RBNW")
todo_dir = os.path.join("todo")

# Directorios de origen
todo_images = os.path.join(todo_dir, "images")
todo_labels = os.path.join(todo_dir, "labels")

# Definición de splits y cantidades
splits = {
    "test": {"count": 30},
    "train": {"count": 96},
    "val": {"count": 24}
}

# Obtener todas las imágenes del conjunto "todo"
all_images = [f for f in os.listdir(todo_images) if os.path.isfile(os.path.join(todo_images, f))]

# Semilla aleatoria distinta cada vez (basada en el tiempo actual)
random.seed(time.time())

# Mezcla 
for _ in range(5):  
    random.shuffle(all_images)

# Limpiar carpetas destino antes de repartir
for split in splits.keys():
    split_img_dir = os.path.join(dataset_dir, split, "images")
    split_lbl_dir = os.path.join(dataset_dir, split, "labels")

    if os.path.exists(split_img_dir):
        shutil.rmtree(split_img_dir)
    if os.path.exists(split_lbl_dir):
        shutil.rmtree(split_lbl_dir)

    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

start_idx = 0

# Repartir las imágenes y labels
for split, info in splits.items():
    num = info["count"]
    end_idx = start_idx + num
    selected_images = all_images[start_idx:end_idx]
    start_idx = end_idx

    split_img_dir = os.path.join(dataset_dir, split, "images")
    split_lbl_dir = os.path.join(dataset_dir, split, "labels")

    for img_name in selected_images:
        label_name = os.path.splitext(img_name)[0] + ".json"
        src_img = os.path.join(todo_images, img_name)
        src_lbl = os.path.join(todo_labels, label_name)
        dst_img = os.path.join(split_img_dir, img_name)
        dst_lbl = os.path.join(split_lbl_dir, label_name)

        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)
        else:
            print(f"⚠️ Saltado: falta imagen o label para {img_name}")

print("✅ Reparto completado con éxito (carpetas destino limpiadas).")
