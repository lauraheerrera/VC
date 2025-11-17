# Práctica 5

**Autores:**  
- Laura Herrera Negrín  
- Ayman Asbai Ghoudan

**Universidad:** Universidad de Las Palmas de Gran Canaria  
**Asignatura:** Visión por Computador  

---
## Contenidos
- [Librerías utilizadas](#librerias)
- [🖥️ Preparación del entorno](#entorno)
- [🎭 Filtro 1 - Entrenamiento previo](#filtro-1)
  - [Preparación del dataset](#dataset-f1)
  - [Extracción y entrenamiento](#entrenamiento-f1)
  - [Resultados de la evaluación](#resultados-f1)
  - [Aplicación del filtro](#aplicacion-f1)
- [🦄 Filtro 2 - Tema libre: Unicornio mágico](#filtro-2)
  - [Funcionalidad y detección facial](#deteccion-f2)
  - [Efectos aplicados](#efectos-f2)
  - [Configuración y ejecución](#configuracion-f2)

---

<a name="librerias"></a>
## Librerías utilizadas

[![OpenCV](https://img.shields.io/badge/OpenCV-%23007ACC?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)  
- Librería para procesamiento de imágenes y visión por computadora.  
- Permite lectura, transformación y visualización de imágenes y video.

[![Matplotlib](https://img.shields.io/badge/Matplotlib-%230077B5?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)  
- Librería para visualización de datos.  
- Permite crear gráficos, histogramas y mostrar imágenes de manera interactiva.

[![Imutils](https://img.shields.io/badge/Imutils-%23FF6F61?style=for-the-badge)](https://github.com/jrosebr1/imutils)  
- Utilidades complementarias para OpenCV.  
- Facilita la manipulación de imágenes, redimensionado y transformación de coordenadas.

[![MTCNN](https://img.shields.io/badge/MTCNN-%23FF6F00?style=for-the-badge)](https://github.com/ipazc/mtcnn)  
- Detector de rostros basado en redes neuronales convolucionales.  
- Permite localizar caras y puntos clave faciales en imágenes.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)  
- Framework de aprendizaje automático y deep learning.  
- Permite entrenar y utilizar modelos de redes neuronales, incluyendo extracción de embeddings faciales.

[![DeepFace](https://img.shields.io/badge/DeepFace-%234BBEFB?style=for-the-badge)](https://github.com/serengil/deepface)  
- Librería para análisis de rostros y biometría facial.  
- Proporciona modelos preentrenados para extracción de embeddings, reconocimiento de identidad y predicción de edad, género o emociones.

[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)  
- Librería para machine learning en Python.  
- Permite entrenar clasificadores, medir métricas de rendimiento y realizar evaluaciones de modelos.

[![Joblib](https://img.shields.io/badge/Joblib-%23E44D26?style=for-the-badge&logo=python&logoColor=white)](https://joblib.readthedocs.io/)  
- Librería para serializar objetos de Python.  
- Se utiliza para guardar y cargar modelos entrenados de manera eficiente.

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)  
- Librería para cálculo numérico y manejo de arrays multidimensionales.  
- Facilita operaciones matemáticas sobre los embeddings y matrices de datos.

[![Dlib](https://img.shields.io/badge/Dlib-%2300A6ED?style=for-the-badge&logo=dlib&logoColor=white)](http://dlib.net/)  
- Librería de aprendizaje automático y visión por computadora.  
- Utilizada principalmente para detección de rostros y alineación facial en datasets.

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

<a name="dataset-f1"></a>
### 🖼️ Preparación del dataset 
Para este prototipo se emplea el dataset "People With and Without Glasses", disponible en Kaggle. El script utiliza la librería `kagglehub` para descargar y localizar automáticamente el dataset mediante el identificador "saramhai/people-with-and-without-glasses-dataset".

El dataset está organizado en dos carpetas que definen las clases: glasses (con gafas) y no_glasses (sin gafas). Asimismo, para agilizar el proceso y reducir el uso de memoria, el script selecciona aleatoriamente un tercio de las imágenes de cada clase para el entrenamiento.

<a name="entrenamiento-f1"></a>
### 🏋🏽 Extracción y entrenamiento
Una vez preparada la muestra de imágenes, el script procede a la extracción de embeddings de los rostros, iterando sobre cada imagen y utilizando `DeepFace.represent` para obtener un embedding representativo de cada rostro; en este proceso se emplea `detector_backend='skip'`, asumiendo que las imágenes ya contienen los rostros recortados, lo que permite acelerar significativamente el procedimiento al omitir la detección facial.

Posteriormente, los embeddings se dividen en conjuntos de entrenamiento (70%) y prueba (30%) para realizar una evaluación preliminar de la viabilidad del enfoque. Finalmente, se entrena un modelo SVC (Support Vector Classifier) con los embeddings y las etiquetas correspondientes, y tanto el modelo como la lista de clases se guardan en disco mediante `joblib` para su posterior uso en la aplicación del filtro.

<a name="resultados-f1"></a>
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

---
<a name="aplicacion-f1"></a>
### Aplicación del filtro
Esta parte de la práctica consiste en utilizar el modelo previamente entrenado para crear el prototipo funcional. El objetivo es cargar el clasificador SVC guardado (`.pkl`) y usarlo para predecir si un rostro en una nueva imagen (que no estaba en el dataset) lleva gafas o no.

--- 
<a name="filtro-2"></a>
## 🦄 Filtro 2 - Tema libre: Unicornio mágico

El segundo prototipo se centra en un filtro interactivo y divertido que aplica efectos mágicos en tiempo real al detectar el rostro del usuario y la apertura de su boca. Cuando la boca se abre por encima de un umbral, el filtro superpone un **cuerno de unicornio** sobre la frente, un **arcoíris** que sale de la boca y un **fondo rosa translúcido con destellos**.

<a name="deteccion-f2"></a>
### 🖥️ Funcionalidad y detección facial
Para este prototipo se utiliza la librería **OpenCV** para captura y procesamiento de video, junto con utilidades de **imutils** y un detector de rostros personalizado (`FaceDetector`). Asimismo, se calculan los **landmarks faciales** usando 68 puntos clave estándar de `dlib`, con el fin de determinar:
- Ojos (puntos 36–45): permiten calcular el centro y la distancia interocular, utilizada como referencia para escalar y posicionar el cuerno.
- Boca (puntos 48–67): sus puntos internos permiten calcular el **Mouth Aspect Ratio (MAR)**.
- Debido a que los landmarks no incluyen la frente, la posición del cuerno se estima geométricamente desde el centro de los ojos con un desplazamiento vertical proporcional a la distancia interocular.

El cálculo del MAR sigue la fórmula: _MAR = (A + B) / 2C_​, 
donde A y B son distancias verticales internas de la boca y C la distancia horizontal.
El umbral elegido es 0.50, determinado experimentalmente por valores típicos de boca cerrada (0.20–0.35) y abierta (>0.55).

<a name="efectos-f2"></a>
### ✨ Efectos aplicados
Cuando el MAR supera el umbral definido (`MOUTH_AR_THRESH = 0.50`), se activan los efectos mágicos:
1. **Fondo mágico rosa y destellos**: Se genera un fondo translúcido con color rosa y se superponen destellos aleatorios sobre la imagen, entre 15 y 35 destellos por fotograma.
2. **Cuerno de unicornio**: Se dibuja un cuerno sobre la frente, centrado entre los ojos, con una altura aproximada de 1.8 x la distancia interocular y escala proporcional al rostro para asegurar realismo.
3. **Arcoíris**: Se dibuja un arcoíris que sale de la boca, entre los puntos 48 y 54, con ajustes de desplazamiento que garantizan un efecto natural. Su tamaño es aproximadamente 5 × el ancho de la boca.

La superposición de imágenes con transparencia se realiza mediante la función `overlay_transparent`, que maneja correctamente recortes y escalado para evitar errores de desbordamiento fuera de la imagen.

<a name="configuracion-f2"></a>
### 💻 Configuración y ejecución
Para ejecutar el filtro, se requiere tener en la misma carpeta las imágenes de efectos:

- `unicorn_horn.png` (cuerno de unicornio)
- `rainbow.png` (arcoíris)
- `sparkles.png` (destellos, con fondo transparente)

Al iniciar el script, se abre la cámara y se muestra el video en tiempo real. Un mensaje guía al usuario para abrir la boca y activar la magia. El filtro continúa hasta que se presiona la tecla `q`.

> Nota: Este prototipo combina técnicas de visión por computadora, geometría facial y superposición de imágenes con transparencia para crear una experiencia interactiva y visualmente atractiva.
