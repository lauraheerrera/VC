import json
import os
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---

# Mapa de clases: adapta según tus etiquetas de LabelMe
CLASS_MAP = {
    'matricula': 0
    # Añade más clases si las tienes, por ejemplo:
    # 'coche': 1,
}


def convert_labelme_to_yolo(json_dir, image_dir, class_map):
    """
    Convierte anotaciones .json de LabelMe (polígonos) a formato .txt de YOLO.
    Guarda los .txt directamente en la carpeta 'labels/' correspondiente.
    """

    # Crear carpeta de salida (labels/) si no existe
    output_dir = json_dir  # Guardar en la misma carpeta que los JSON
    os.makedirs(output_dir, exist_ok=True)

    # 1. Buscar archivos JSON
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No se encontraron archivos .json en {json_dir}")
        return

    print(f"Encontrados {len(json_files)} archivos JSON en {json_dir}. Iniciando conversión...")

    # 2. Procesar cada archivo JSON
    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error al leer {json_file}: {e}")
            continue

        # 3. Dimensiones de la imagen
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')

        if img_width is None or img_height is None:
            print(f"\nError: No se encuentran 'imageWidth' o 'imageHeight' en {json_file}. Saltando.")
            continue

        # 4. Nombre de la imagen
        img_filename = data.get('imagePath')
        if not img_filename:
            print(f"\nAdvertencia: No se encuentra 'imagePath' en {json_file}. Saltando.")
            continue

        yolo_annotations = []

        # 5. Iterar sobre cada shape
        for shape in data.get('shapes', []):
            label = shape.get('label')

            if label not in class_map:
                print(f"\nAdvertencia: Etiqueta '{label}' en {json_file} no está en CLASS_MAP. Saltando.")
                continue

            class_id = class_map[label]
            points = shape.get('points')

            if shape.get('shape_type') != 'polygon' or not points:
                continue

            # Calcular el bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            box_width = x_max - x_min
            box_height = y_max - y_min
            x_center = x_min + box_width / 2
            y_center = y_min + box_height / 2

            # Normalizar
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height

            # Formato YOLO
            yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_annotations.append(yolo_line)

        # 6. Guardar el archivo .txt directamente en labels/
        txt_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.txt'
        txt_output_path = os.path.join(output_dir, txt_filename)

        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_annotations))

    print(f"\n✅ Conversión completada.")
    print(f"Archivos TXT guardados en: {output_dir}")


# --- 2. EJECUCIÓN ---

if __name__ == "__main__":
    print("--- PROCESANDO CARPETA TEST ---")
    JSON_DIR_TEST = 'TGC_RBNW/test/labels/' 
    IMAGE_DIR_TEST = 'TGC_RBNW/test/images/'
    convert_labelme_to_yolo(JSON_DIR_TEST, IMAGE_DIR_TEST, CLASS_MAP)

    
    print("--- PROCESANDO CARPETA TRAIN ---")
    JSON_DIR_TRAIN = 'TGC_RBNW/train/labels/' 
    IMAGE_DIR_TRAIN = 'TGC_RBNW/train/images/'
    convert_labelme_to_yolo(JSON_DIR_TRAIN, IMAGE_DIR_TRAIN, CLASS_MAP)

    
    print("--- PROCESANDO CARPETA VAL ---")
    JSON_DIR_VAL = 'TGC_RBNW/val/labels/' 
    IMAGE_DIR_VAL = 'TGC_RBNW/val/images/'
    convert_labelme_to_yolo(JSON_DIR_VAL, IMAGE_DIR_VAL, CLASS_MAP)
