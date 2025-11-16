# Práctica 5

**Autores:**  
- Laura Herrera Negrín  
- Ayman Asbai Ghoudan

**Universidad:** Universidad de Las Palmas de Gran Canaria  
**Asignatura:** Visión por Computador  

---
## Contenidos
- [Librerías utilizadas](#librerias)
- 
 
---

<a name="librerias"></a>
## Librerías utilizadas

--- 

La detección y análisis de rostros humanos mediante herramientas como DeepFace permite extraer información biométrica y emocional de manera automática, abriendo posibilidades para aplicaciones interactivas y personalizadas. El objetivo de la práctica es explorar estas capacidades mediante el desarrollo de dos prototipos: uno que utilice un modelo entrenado para la extracción de información biométrica específica, y otro de temática libre que genere reacciones a partir de los datos faciales detectados.

<a name="entorno"></a>
## 🖥️ Preparación del entorno
Para garantizar la correcta ejecución de los prototipos, es necesario configurar un entorno de Python con las librerías requeridas. Este entorno incluye herramientas para procesamiento de imágenes, detección de rostros y análisis de información biométrica mediante DeepFace y modelos de TensorFlow.
Para ello, se creó un nuevo entorno de Conda con Python 3.11.5:

```bash
conda create --name VC_P5 python=3.11.5
conda activate VC_P5
pip install opencv-python matplotlib imutils mtcnn tensorflow deepface tf-keras cmake kagglehub  scikit-learn joblib numpy
```

Asimismo, la librería `dlib` se instalará mediante un binario para asi evitar conflictos:
  * Descargar el archivo [dlib-19.24.1-cp311-cp311-win_amd64.whl](https://github.com/ZeroReiNull/dlib-python/blob/main/dlib-19.24.1-cp311-cp311-win_amd64.whl)
  * Instalar con pip:
```bash
pip install path\to\dlib-19.24.1-cp311-cp311-win_amd64.whl
```
> Nota: Reemplazar `path\to\` con la ruta donde se descarga el archivo .whl.

<a name="filtro-1"></a>
## 🎭 Filtro 1 - Entrenamiento previo
Para el primer prototipo, se optó por desarrollar un filtro capaz de identificar si el usuario lleva gafas o no.

### 🖼️ Preparación del dataset 
Para este prototipo se emplea el dataset "People With and Without Glasses", disponible en Kaggle. El script utiliza la librería `kagglehub` para descargar y localizar automáticamente el dataset mediante el identificador "saramhai/people-with-and-without-glasses-dataset".

El dataset está organizado en dos carpetas que definen las clases: glasses (con gafas) y no_glasses (sin gafas). Asimismo, para agilizar el proceso y reducir el uso de memoria, el script selecciona aleatoriamente un tercio de las imágenes de cada clase para el entrenamiento.

### 🏋🏽 Extracción y entrenamiento
Una vez preparada la muestra de imágenes, el script procede a la extracción de embeddings de los rostros, iterando sobre cada imagen y utilizando `DeepFace.represent` para obtener un embedding representativo de cada rostro; en este proceso se emplea `detector_backend='skip'`, asumiendo que las imágenes ya contienen los rostros recortados, lo que permite acelerar significativamente el procedimiento al omitir la detección facial.

Posteriormente, los embeddings se dividen en conjuntos de entrenamiento (70%) y prueba (30%) para realizar una evaluación preliminar de la viabilidad del enfoque. Finalmente, se entrena un modelo SVC (Support Vector Classifier) con los embeddings y las etiquetas correspondientes, y tanto el modelo como la lista de clases se guardan en disco mediante `joblib` para su posterior uso en la aplicación del filtro.

### 📊 Resultados de la evaluación
El informe de clasificación generado durante la evaluación preliminar, muestra la calidad del clasificador. Un accuracy (exactitud) alto indica que los embeddings de ArcFace son muy efectivos para separar las dos clases (con y sin gafas) usando un simple SVC.

A continuación, se muestra un ejemplo de los resultados obtenidos en la consola durante la evaluación con el 30% de los datos:

| Clase        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| glasses      | 0.99      | 0.98   | 0.98     | 584     |
| no_glasses   | 0.98      | 0.99   | 0.98     | 583     |
| **accuracy** |           |        | 0.98     | 1167    |
| macro avg    | 0.98      | 0.98   | 0.98     | 1167    |
| weighted avg | 0.98      | 0.98   | 0.98     | 1167    |

Los resultados demuestran una precisión superior al 98%, validando que el modelo SVC entrenado sobre los embeddings de ArcFace es un método excelente para esta tarea de clasificación.
