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
- [Práctica 4b - ](#tarea2)
---

<a name= "librerias"></a>
## Librerías utilizadas

--- 

<a name="práctica4"></a>
## Práctica 4 - Detección de vehículos y matrículas
El objetivo de esta práctica es desarrollar un prototipo para detectar y seguir vehículos y personas, así como la localización y reconocimiento de las matrículas de dichos vehículos a partir de un vídeo. Para ello, se han empleado modelos de detección de objetos YOLO (You Only Look Once).

<a name="entorno"></a>
### Prearación del entorno
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
### Preparación del dataset para YOLO
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

Para crear esta estructura, se desarrolló, con ayuda de la IA, un [**script en Python**](https://github.com/lauraheerrera/VC/blob/P4/Practica4/repartir.py) que tomó todas las imágenes y etiquetas almacenadas inicialmente en la carpeta  
[_todo_](https://github.com/lauraheerrera/VC/tree/P4/Practica4/todo) y las dividió en tres subconjuntos de forma automática:
- **80%** del total del dataset se destinó a **entrenamiento y validación**.  
- **20%** restante se reservó para **pruebas (test)**.  
- Del **80% inicial**, se dividió de nuevo en:
  - **80%** para **entrenamiento (train)**
  - **20%** para **validación (val)**

De esta forma, se garantiza una distribución equilibrada y representativa del dataset, cumpliendo con las prácticas recomendadas para el entrenamiento de modelos de detección de objetos.

#### 4. De `json` a formato YOLO
Una vez creada la estructura de carpetas del dataset, es importante recordar  que las anotaciones generadas con **LabelMe** se guardan inicialmente en formato `.json`. Para que el modelo **YOLO** pueda utilizarlas, es necesario convertirlas al formato de etiquetas propio del framework.

Para ello, se desarrolló un [**script en Python**](https://github.com/lauraheerrera/VC/blob/P4/Practica4/script.py) que recorre todas las etiquetas en formato `.json` y las convierte en archivos `.txt` con la estructura estándar de YOLO:
`<class_id> <x_center> <y_center> <width> <height>`

Cada línea del archivo `.txt` corresponde a un objeto detectado dentro de la imagen y contiene la siguiente información:
- **class_id** → identificador numérico de la clase del objeto (por ejemplo, `0` para matrículas).  
- **x_center** → coordenada **x** del centro del contenedor delimitador.  
- **y_center** → coordenada **y** del centro del contenedor delimitador.  
- **width** → ancho del contenedor delimitador.  
- **height** → alto del contenedor delimitador.  

Las coordenadas del centro (`x_center`, `y_center`) y las dimensiones (`width`, `height`) se encuentran **normalizadas**, es decir, divididas por el ancho y alto total de la imagen, para que sus valores estén comprendidos entre 0 y 1.

De esta forma, las etiquetas resultantes son totalmente compatibles con los modelos **YOLO**, permitiendo entrenar el detector de matrículas de manera eficiente.

#### 5. Archivo de configuración del dataset

Se creó un archivo [`data.yaml`](https://github.com/lauraheerrera/VC/blob/P4/Practica4/data.yaml), que define las rutas del conjunto de datos utilizadas durante el entrenamiento, validación y prueba del modelo.  
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
### Proceso para el entrenamiento YOLO
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
### Resultados del entrenamiento
Tras ejecutar los distintos entrenamientos, YOLO genera automáticamente los resultados en la carpeta:
Dentro de esta carpeta, se crean subcarpetas por cada ejecución, por ejemplo `train1`, `train2`, etc. Cada subcarpeta contiene los siguientes elementos:
- **`weights/`** → Modelos entrenados:
  - **`best.pt`** → Modelo que obtuvo la mejor precisión durante el entrenamiento.  
  - **`last.pt`** → Modelo final después de completar todas las épocas, aunque no sea el más preciso.
- **`results.png`** → Gráfica que muestra la evolución de las métricas de entrenamiento: precisión, recall y loss.

