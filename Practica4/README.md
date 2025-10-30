# Práctica 4 y 4b

**Autores:**  
- Laura Herrera Negrín  
- Ayman Asbai Ghoudan

**Universidad:** Universidad de Las Palmas de Gran Canaria  
**Asignatura:** Visión por Computador  

---
## Contenidos
- [Librerías utilizadas](#librerias)
- [Práctica 4 - Detección de vehículos y matrículas](#práctica4) 
    - [Preparación del dataset para YOLO](#dataset) 
    - [Entrenamiento YOLO](#entrenamiento)
    - [Resultados del entrenamiento](#resultados)
    - [Instrucciones para ejecutar el script](#script) 
- [Práctica 4b - ](#tarea2)
---

<a name="librerias"></a>
## Librerías utilizadas
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)  
- Framework principal para entrenamiento de modelos YOLO.  
- Soporte de GPU mediante CUDA para acelerar el entrenamiento.  
- Incluye módulos como `torchvision` y `torchaudio` para manipulación de datos multimodales.  

[![CUDA](https://img.shields.io/badge/CUDA-%230edc0f?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)  
- Librería de aceleración por GPU utilizada por PyTorch.  

[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLO-%23FF6F61?style=for-the-badge&logo=ultralytics&logoColor=white&labelColor=%23FF6F61)](https://github.com/ultralytics/ultralytics)
- Implementación moderna de YOLO (YOLOv11).  
- Facilita entrenamiento, validación y detección de objetos con modelos preentrenados y personalizados.  

[![LabelMe](https://img.shields.io/badge/LabelMe-%23F6A623?style=for-the-badge&logo=labelme&logoColor=white)](https://github.com/wkentaro/labelme)  
- Herramienta gráfica para anotación de imágenes.  
- Generar archivos `.json` con las coordenadas de objetos (matrículas).  

[![lap](https://img.shields.io/badge/lap-%23007ACC?style=for-the-badge)](https://pypi.org/project/lap/)  
- Librería para resolver problemas de asignación lineal, útil en seguimiento de objetos.  

--- 

<a name="práctica4"></a>
## Práctica 4 - Detección de vehículos y matrículas
El objetivo de esta práctica es desarrollar un prototipo para detectar y seguir vehículos y personas, así como la localización y reconocimiento de las matrículas de dichos vehículos a partir de un vídeo. Para ello, se han empleado modelos de detección de objetos YOLO (You Only Look Once).

<a name="entorno"></a>
### 🖥️ Prearación del entorno
Para evitar conflictos entre librerías y garantizar la compatibilidad con el módulo de **OCR** utilizado posteriormente, se creó un nuevo entorno de **Conda** con **Python 3.9.5**:
```bash
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ultralytics
pip install lap
```
La tercera instrucción instala PyTorch junto con sus librerías asociadas (torchvision y torchaudio) y habilita el soporte de CUDA 11.8 para aprovechar la aceleración por GPU.

El paquete Ultralytics permite acceder a las versiones más recientes de YOLO (YOLOv11 y YOLOv12), facilitando tanto el uso de modelos preentrenados como el entrenamiento de modelos personalizados.
  
<a name= "dataset"></a>
### 🖼️ Preparación del dataset para YOLO
Para la detección de matrículas, se decidió entrenar un modelo YOLO personalizado, ya que los modelos preentrenados no incluyen esta clase de objeto por defecto.

El proceso seguido fue el siguiente:
#### 1. Obtención y preparación del dataset
Se recopiló un conjunto de imágenes que contuvieran vehículos con matrículas visibles.  
Este dataset fue creado de forma colaborativa entre los miembros del equipo de la asignatura, garantizando la variedad de condiciones (ángulos, iluminación, tipos de vehículos, etc.).  
En total, se recopilaron 150 imágenes, que se guardaron en la carpeta [_todo_](https://github.com/lauraheerrera/VC/tree/P4/Practica4/todo), para posteriormente etiquetarlas y, una vez etiquetadas organizarlas siguiendo la estructura de YOLO.

#### 2. Anotación de imágenes
Para anotar las matrículas dentro de las imágenes se utilizó la herramienta **LabelMe**, que permite dibujar regiones rectangulares alrededor del objeto de interés (la matrícula).  

Para el uso de esta herramienta, se creó otro entorno:
```bash
conda create --name=labelme python=3.9
conda activate labelme
pip install labelme
```
Una vez instalado, se tecleó _labelme_ desde _AnacondaPrompt_ y se abrió una interfaz intuitiva para anotar las zonas alrededor de las matrículas presentes.

Cada imagen anotada genera un archivo `.json` con la información de las regiones seleccionadas.  

#### 3. Estructura de directorios
Las imágenes recolectadas se organizaron siguiendo la estructura esperada por **YOLO** para el entrenamiento, validación y prueba del modelo.  
Cada subconjunto contiene sus respectivas carpetas de imágenes (`images/`) y etiquetas (`labels/`).

<pre>
📂 <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW">TGC_RBNW/</a>
├── 📂 <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/train">train/</a>
│   ├── <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/train/images">images/</a>
│   └── <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/train/labels">labels/</a>
├── 📂 <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/val">val/</a>
│   ├── <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/val/images">images/</a>
│   └── <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/val/labels">labels/</a>
└── 📂 <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/test">test/</a>
    ├── <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/test/images">images/</a>
    └── <a href="https://github.com/lauraheerrera/VC/tree/P4/Practica4/TGC_RBNW/test/labels">labels/</a>
</pre>

Para crear esta estructura, se desarrolló, con ayuda de la IA, un [**script en Python**](https://github.com/lauraheerrera/VC/blob/P4/Practica4/repartir.py) que tomó todas las imágenes y etiquetas almacenadas inicialmente en la carpeta [_todo_](https://github.com/lauraheerrera/VC/tree/P4/Practica4/todo) y las dividió en tres subconjuntos de forma automática:
- **80%** del total del dataset se destinó a **entrenamiento y validación**.  
- **20%** restante se reservó para **pruebas (test)**.  
- Del **80% inicial**, se dividió de nuevo en:
  - **80%** para **entrenamiento (train)**
  - **20%** para **validación (val)**
> [!IMPORTANT]
> Para ejecutar el script, sigue las instrucciones que se indican en la sección [Instrucciones para ejecutar el script](#script) ].

De esta forma, se garantiza una distribución equilibrada y representativa del dataset, cumpliendo con las prácticas recomendadas para el entrenamiento de modelos de detección de objetos.

#### 4. De `json` a formato YOLO
Una vez creada la estructura de carpetas del dataset, es importante recordar  que las anotaciones generadas con **LabelMe** se guardan inicialmente en formato `.json`. Para que el modelo **YOLO** pueda utilizarlas, es necesario convertirlas al formato de etiquetas propio del framework.

Para ello, se desarrolló un [**script en Python**](https://github.com/lauraheerrera/VC/blob/P4/Practica4/scripts/json_to_txt.py) que recorre todas las etiquetas en formato `.json` y las convierte en archivos `.txt` con la estructura estándar de YOLO:
`<class_id> <x_center> <y_center> <width> <height>`

> [!IMPORTANT]
> Para ejecutar el script, sigue las instrucciones que se indican en la sección [Instrucciones para ejecutar el script](#script) ].

Cada línea del archivo `.txt` corresponde a un objeto detectado dentro de la imagen y contiene la siguiente información:
- **class_id** → identificador numérico de la clase del objeto (por ejemplo, `0` para matrículas).  
- **x_center** → coordenada **x** del centro del contenedor delimitador.  
- **y_center** → coordenada **y** del centro del contenedor delimitador.  
- **width** → ancho del contenedor delimitador.  
- **height** → alto del contenedor delimitador.  

Las coordenadas del centro (`x_center`, `y_center`) y las dimensiones (`width`, `height`) se encuentran **normalizadas**, es decir, divididas por el ancho y alto total de la imagen, para que sus valores estén comprendidos entre 0 y 1.

De esta forma, las etiquetas resultantes son totalmente compatibles con los modelos **YOLO**, permitiendo entrenar el detector de matrículas de manera eficiente.

#### 5. Archivo de configuración del dataset

Se creó un archivo [`data.yaml`](https://github.com/lauraheerrera/VC/blob/P4/Practica4/scripts/data.yaml), que define las rutas del conjunto de datos utilizadas durante el entrenamiento, validación y prueba del modelo.  
Además, este archivo especifica el número de clases y sus nombres, información necesaria para que **YOLO** interprete correctamente el dataset.

El contenido del archivo tiene la siguiente estructura:
```yaml
# TGCRBNW paths

train: C:/Users/laura/OneDrive/Desktop/VC/Practica4/TGC_RBNW/train/
val: C:/Users/laura/OneDrive/Desktop/VC/Practica4/TGC_RBNW/val/
test: C:/Users/laura/OneDrive/Desktop/VC/Practica4/TGC_RBNW/test/

# number of classes
nc: 1

# class names
names: [ 'license_plate' ]
```
---

<a name= "entrenamiento"></a>
### 📈 Proceso para el entrenamiento YOLO
A continuación, se entrenará el modelo YOLO:
#### 1. Activar el entorno para entrenamiento si no se ha hecho previamente
`conda activate VC_P4`

#### 2. Comprobar que la GPU está disponible
```bash
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

#### 3. Ejecutar entrenamiento YOLO
Desde la carpeta donde está `data.yaml` y las imágenes:
```
cd "C:\Users\laura\OneDrive\Desktop\VC\Practica4"
```
1. Train 1 (T1) – Entrenamiento rápido de referencia
```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=512 batch=4 device=0 epochs=40 lr0=0.01
```
2. Train 2 (T2) – Entrenamiento largo con imágenes pequeñas
```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=416 batch=4 device=0 epochs=100 lr0=0.001
```
3.  Train 3 (T3) – Entrenamiento largo con resolución media
```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=512 batch=4 device=0 epochs=100 lr0=0.001
```
4. Train 4 (T4) – Entrenamiento con imágenes grandes y pocas épocas
```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=640 batch=4 device=0 epochs=50 lr0=0.001
```
5. Train 5 (T5) – Entrenamiento con batch grande
```
yolo detect train model=yolo11n.pt data=data.yaml imgsz=512 batch=8 device=0 epochs=60 lr0=0.001
```
6. Train 6 (T6) – Repetición para comparar consistencia
```
yolo detect train model=yolo11n.pt data=data.yaml imgsz=416 batch=4 device=0 epochs=100 lr0=0.001
```
7. Train 7 (T7) – Entrenamiento de alta resolución, pocas épocas
```
yolo detect train model=yolo11n.pt data=data.yaml imgsz=768 batch=2 device=0 epochs=25 lr0=0.001
```
8. Train 8 (T8) – Entrenamiento balanceado entre resolución y duración
```
yolo detect train model=yolo11n.pt data=data.yaml imgsz=640 batch=4 device=0 epochs=80 lr0=0.001
```
9. Train 9 (T9) – Entrenamiento con learning rate alto
```
yolo detect train model=yolo11n.pt data=data.yaml imgsz=512 batch=4 device=0 epochs=100 lr0=0.01
```

Parámetros:
- `model` → modelo base/preentrenado (`yolo11n.pt`)
- `data` → archivo YAML con rutas y clases (`data.yaml`)
- `imgsz` → tamaño de entrada de las imágenes (512, 640, etc.)
- `batch` → tamaño de batch por iteración (4, 8, etc.)
- `device=0` → GPU utilizada (0 para la primera GPU, cpu si no hay GPU)
- `epochs` → número de épocas de entrenamiento (40, 100, etc.)
- `lr0` → learning rate inicial para el entrenamiento (0.001, 0.01, etc.)
  
Se realizaron 9 entrenamientos para evaluar distintas combinaciones de tamaño de imagen, número de épocas, batch y learning rate:
- T1: Entrenamiento rápido de referencia, pocas épocas y tamaño medio.
- T2: Largo con imágenes pequeñas, para evaluar convergencia con menor detalle.
- T3: Largo con resolución media, comparando precisión con T2.
- T4: Imágenes grandes y pocas épocas, para capturar detalles sin mucho tiempo de entrenamiento.
- T5: Batch grande, probando estabilidad y suavidad de la convergencia.
- T6: Repetición de un entrenamiento largo, para validar consistencia de resultados.
- T7: Alta resolución y pocas épocas, ideal para matrículas pequeñas o lejanas.
- T8: Balance entre resolución y duración, buscando un modelo sólido.
- T9: Learning rate alto, para observar efecto en velocidad de convergencia y estabilidad.

Este conjunto permite comparar cómo cada parámetro afecta la precisión y eficiencia del modelo de detección de matrículas.

<a name="resultados"></a>
### 📊 Resultados del entrenamiento
Tras ejecutar los distintos entrenamientos, YOLO genera automáticamente los resultados en la carpeta:
Dentro de esta carpeta, se crean subcarpetas por cada ejecución, por ejemplo `train1`, `train2`, etc. Cada subcarpeta contiene los siguientes elementos:
- **`weights/`** → Modelos entrenados:
  - **`best.pt`** → Modelo que obtuvo la mejor precisión durante el entrenamiento.  
  - **`last.pt`** → Modelo final después de completar todas las épocas, aunque no sea el más preciso.
- **`results.png`** → Gráfica que muestra la evolución de las métricas de entrenamiento: precisión, recall y loss.
- **`results.csv`** -> Registro época por época de las métricas durante el entrenamiento y la validación. Cada fila corresponde a una éopca

Para determinar qué entrenamiento se considera el mejor, se ha tenido en cuenta las principales métricas, que reflejan la calidad de la detección:

| Métrica                | Qué indica                                             | Consideración para evaluación                                                 |
| ---------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| `metrics/precision(B)` | Qué porcentaje de las predicciones fueron correctas    | Más alto = mejor                                                   |
| `metrics/recall(B)`    | Qué porcentaje de los objetos reales fueron detectados | Más alto = mejor                                                   |
| `metrics/mAP50(B)`     | Precisión promedio considerando IoU ≥ 0.5              | Más alto = mejor; ideal >0.7 para muchos casos                     |
| `metrics/mAP50-95(B)`  | Precisión promedio considerando IoU entre 0.5 y 0.95   | Más robusta que mAP50, porque penaliza predicciones menos precisas |

Otras métricas importantes son las de pérdidas, que reflejan qué tan bien aprende el modelo:
| Métrica                           | Qué indica                                                  | Consideración para evaluación    |
| --------------------------------- | ----------------------------------------------------------- | ---------------- |
| `train/box_loss` y `val/box_loss` | Error de localización (qué tan bien encaja el bounding box) | Más bajo = mejor |
| `train/cls_loss` y `val/cls_loss` | Error de clasificación (qué tan bien clasifica el objeto)   | Más bajo = mejor |
| `train/dfl_loss` y `val/dfl_loss` | Loss de distribución focal (refina boxes)                   | Más bajo = mejor |

Para evaluar la calidad de un modelo, se ha priorizado las métricas `*_best` de mAP y pérdidas de validación, ya que reflejan el mejor rendimiento alcanzado durante el entrenamiento.

Sabiendo esto, se ha desarrolado otro [script de Python](https://github.com/lauraheerrera/VC/blob/P4/Practica4/scripts/guardar_metricas_yolo.py) que recorre automáticamente todas las carpetas de entrenamiento (`train`,  `train2`, etc.), extrae las métricas de cada ejecución y genera un resumen de los mejores resultados para cada entrenamiento. Además, ordena automáticamente los entrenamientos según el siguiente criterio de prioridad, de manera que el primero en la lista corresponde al modelo mejor considerado:
1. `mAP50(B)` más alto → La métrica principal para determinar precisión de detección.
2. `mAP50-95(B)` alto → Evalúa robustez frente a predicciones menos perfectas.
3. Loss de validación bajos (`val/box_loss`, `val/cls_loss`, `val/dfl_loss`) → Indican que el modelo aprendió bien sin sobreajustarse.
4. Precision y recall equilibrados → Evita falsos positivos o falsos negativos excesivos, asegurando un modelo confiable.


> [!IMPORTANT]
> Para ejecutar el script, sigue las instrucciones que se indican en la sección [Instrucciones para ejecutar el script](#script) ].

> De esta manera, al abrir el [Excel generado por el script](https://github.com/lauraheerrera/VC/blob/P4/Practica4/resumen_entrenamientos_mejores.xlsx), los entrenamientos aparecen ordenados según estas prioridades, facilitando la identificación del mejor modelo sin necesidad de revisar manualmente cada métrica.

La siguiente tabla muestra cómo se presentan los entrenamientos en el _Excel_:
| Entrenamiento | train/box_loss_best | train/cls_loss_best | train/dfl_loss_best | val/box_loss_best | val/cls_loss_best | val/dfl_loss_best | metrics/precision(B)_best | metrics/recall(B)_best | metrics/mAP50(B)_best | metrics/mAP50-95(B)_best | val_loss_sum | pr_sum |
|---------------|------------------|-------------------|-------------------|-----------------|-----------------|-----------------|---------------------------|------------------------|----------------------|-------------------------|-------------|--------|
| train8        | 1.09635          | 1.2899            | 1.05174           | 0.87794         | 0.79134         | 0.96896         | 0.99986                   | 1.0000                 | 0.9950               | 0.74894                 | 2.63824     | 1.99986|
| train         | 0.96920          | 1.15768           | 0.98533           | 0.96195         | 0.83262         | 0.96145         | 0.99636                   | 1.0000                 | 0.9950               | 0.69216                 | 2.75602     | 1.99636|
| train3        | 0.96814          | 0.84988           | 0.95254           | 1.02529         | 0.75701         | 1.01806         | 0.99467                   | 1.0000                 | 0.9950               | 0.68005                 | 2.80036     | 1.99467|
| train9        | 0.96814          | 0.84988           | 0.95254           | 1.02529         | 0.75701         | 1.01806         | 0.99467                   | 1.0000                 | 0.9950               | 0.68005                 | 2.80036     | 1.99467|
| train2        | 0.72048          | 0.55734           | 0.89550           | 0.94331         | 0.60031         | 0.96383         | 1.00000                   | 0.94927                | 0.99204              | 0.70802                 | 2.50745     | 1.94927|
| train6        | 0.72048          | 0.55734           | 0.89550           | 0.94331         | 0.60031         | 0.96383         | 1.00000                   | 0.94927                | 0.99204              | 0.70802                 | 2.50745     | 1.94927|
| train4        | 0.83696          | 1.04597           | 0.90473           | 0.85140         | 0.76191         | 0.94132         | 0.95481                   | 1.0000                 | 0.99192              | 0.76919                 | 2.55463     | 1.95481|
| train5        | 0.74647          | 0.73564           | 0.91225           | 0.91703         | 0.70769         | 0.97270         | 0.96803                   | 0.9600                 | 0.99071              | 0.74413                 | 2.59742     | 1.92803|
| train7        | 0.94845          | 2.28946           | 0.95819           | 0.91889         | 1.43075         | 1.05861         | 0.95651                   | 0.9600                 | 0.98724              | 0.72895                 | 3.40825     | 1.91651|

Como se observa, ** `train8` es el modelo recomendado para la detección de matrículas**, y los resultados obtenidos servirán como referencia para optimizar y ajustar futuras iteraciones del entrenamiento de YOLO.

<a name= "script"></a>
### Instrucciones para ejecutar un script  
> Abre tu terminal  
> Sitúate en la carpeta donde se encuentra el script. En mi caso: `cd "C:\Users\Laura\Desktop\VC\Practica 4"`  
> Ejecuta el script: `python <nombre_script>`


