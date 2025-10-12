# Práctica 3

**Autores:**  
- Laura Herrera Negrín  
- Ayman Asbai Ghoudan

**Universidad:** Universidad de Las Palmas de Gran Canaria  
**Asignatura:** Visión por Computador  

---
## Contenidos
- [Librerías utilizadas](#librerias)
- [Tarea 1 - Conteo de monedas](#tarea1)
- [Tarea 2 - Identificación de microplásticos](#tarea2)
---

<a name= "librerias"></a>
## Librerías utilizadas

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)  
- Manipulación eficiente de arreglos y matrices.  
- Cálculos estadísticos y vectoriales sobre las características extraídas.  
- Operaciones matemáticas para medir distancias entre vectores de características.  

 [![Matplotlib](https://img.shields.io/badge/Matplotlib-%23006DBA?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)  
- Visualización de resultados de clasificación (imágenes reales vs predichas).  
- Presentación de comparativas gráficas y figuras informativas.  

[![Seaborn](https://img.shields.io/badge/Seaborn-%232E8B57?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)  
- Visualización avanzada de datos.  
- Creación de la **matriz de confusión** mediante mapas de calor con estilo mejorado.  

[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/stable/)  
- Evaluación del rendimiento del clasificador (exactitud, precisión, recall, F1-score).  
- Normalización de vectores de características con `StandardScaler`.  
- Generación y análisis de la matriz de confusión.  

[![CSV](https://img.shields.io/badge/CSV-%2300BFAE?style=for-the-badge&logo=csv&logoColor=white)](https://docs.python.org/3/library/csv.html)  
- Lectura y manejo de anotaciones desde archivos CSV.  
- Carga de etiquetas reales y coordenadas de bounding boxes para validar la clasificación.  

--- 
<a name="tarea1"></a>
## TAREA 1 
**Los ejemplos ilustrativos anteriores permiten saber el número de monedas presentes en la imagen. ¿Cómo saber la cantidad de dinero presente en ella? Sugerimos identificar de forma interactiva (por ejemplo haciendo clic en la imagen) una moneda de un valor determinado en la imagen (por ejemplo de 1€). Tras obtener esa información y las dimensiones en milímetros de las distintas monedas, realiza una propuesta para estimar la cantidad de dinero en la imagen. Muestra la cuenta de monedasW y dinero sobre la imagen. No hay restricciones sobre utilizar medidas geométricas o de color.**
- **Salida:**
  - [`salidas/monedas_ideal_resultado.jpg`](salidas/monedas_ideal_resultado.jpg)
  - [`salidas/Monedas1_resultado.jpg`](salidas/Monedas1_resultado.jpg)
  - [`salidas/monedas2_resultado.jpg`](salidas/monedas2_resultado.jpg)
  - [`salidas/monedas3_resultado.jpg`](salidas/monedas3_resultado.jpg)

Esta tarea tiene como principal objetivo detectar monedas en una imagen y estimar la cantidad total de dinero presente. Para ello, se ha seguido la sugerencia planteada: el programa permite al usuario **seleccionar interactivamente una moneda de referencia** (haciendo clic en ella) e indicar su **valor en euros**.
Con esta información y las dimensiones reales de monedas en milímetros, se calcula la **escala milímetro - píxel** y se determina el valor de todas las monedas detectadas en la imagen. 

El resultado final mostrará:
* Las monedas detectadas
* El valor estimado de cada moneda
* El total dinero presente en la imagen


### ⚙️ Funciones principales

A continuación se describen las principales funciones implementadas para llevar a cabo este proceso:
```py 
cargar_y_preprocesar(ruta_img, metodo='gris')
```
- Carga la imagen desde disco y aplica un preprocesamiento para mejorar la detección:
  - Si se usa el método `gris`, convierte a escala de grises y aplica un desenfoque mediano.
  - Si se usa `threshold`, convierte a gris y aplica binarización con Otsu para segmentar las monedas.
---
```py 
detectar_monedas(img, metodo='hough', radio_min=40, radio_max=160, area_min=200)
```
- Detecta las monedas presentes en la imagen:
 - Con `metodo=hough`, usa **Transformada de Hough** para detectar círculos.
 - Con `metodo=contours`, usa **contornos y círculos mínimos envolventes.**
Cada moneda detectada se almacena con su centro y su radio en píxeles.
---
```py 
seleccionar_moneda_referencia(img, monedas)
```
- Permite al usuario hacer **clic sobre una moneda** en la imagen para seleccionarla como **referencia**.
- Guarda sus coordenadas y radio, que se usarán para **calcular la escala**.
---
```py 
calcular_escala(ref_moneda)
```
- Solicita al usuario el valor de la moneda seleccionada (por ejemplo, 1€ o 0.10€) y calcula la **escala miolímetro - píxel**, comparando el radio detectado con el radio real de esa moneda.
- Esta escala se usará para estimar el tamaño de las demás monedas.
---
```py 
clasificar_monedas(monedas, escala, rel_tol=0.12, abs_tol_mm=1.5)
```
- Convierte el radio de cada moneda de píxeles a milímetros y lo compara con los radios reales de las monedas de euro.
- Asigna a cada moneda el valor más probable (0.01€, 0.02€, 0.05€, 0.10€, 0.20€, 0.50€, 1€, 2€) y calcula el total acumulado.
---
```py 
crear_rellenos(img, monedas)
```
- Genera una imagen en blanco y negro donde las monedas detectadas aparecen como círculos blancos rellenos.
---
```py 
mostrar_resultados(img, img_rellenos, resultados, total, ruta_salida)
```
Muestra los resultados de forma visual:
- Imagen original.
- Imagen de rellenos.
- Imagen final con monedas detectadas, su valor y el total.
Asimismo, guarda la imagen final con las monedas y sus valores en la ruta indicada: `salidas/..._resultado.jpg`
---
```py 
contar_monedas(ruta_img, metodo='hough')
```
Integra todas las funciones anteriores:
1. Carga y preprocesa la imagen.
2. Detecta las monedas.
3. Permite seleccionar la moneda de referencia.
4. Calcula la escala y clasifica las monedas.
5. Muestra y guarda los resultados finales.
Devuelve el total detectado y una lista de los resultados individuales.
---

El programa implementa dos métodos para adaptarse a distintos tipos de imágenes:
| **Método** | **Descripción** | **Ventajas** | **Inconvenientes** |
|--------------|------------------|--------------|----------------|
| **`threshold/contours`** | Segmenta la imagen mediante binarización (umbral) y detecta contornos circulares. | Ideal para imágenes limpias o sintéticas (“imagen ideal”). | En imágenes reales con sombras o brillos, puede fallar o detectar menos monedas. |
| **`hough`** | Utiliza la Transformada de Hough para detectar círculos directamente sobre la imagen en escala de grises. | Más robusto ante variaciones de iluminación o fondos complejos. | En imágenes ideales, puede dar errores en el cálculo de los valores de las monedas. |

Ya que no se ha sido capaz de obtener buenos resultados para ambas situaciones, se ha desarrollado una versión con ambos métodos:
- En la imagen ideal, el método por umbral `(threshold/contours)` ofrece mejores resultados.
- En imágenes reales o no ideales, el método de Hough detecta mejor las monedas.

#### 🔍 Resultados obtenidos

**Imagen ideal**

En esta situación, se trabaja con un entorno ideal, donde las condiciones son óptimas para la detección de monedas. Una **imagen ideal** se caracteriza por:
- Iluminación uniforme, sin sombras ni reflejos.
- Fondo liso y homogéneo, que contrasta claramente con las monedas.
- Monedas bien separadas, sin solapamientos ni oclusiones.
- Enfoque nítido y sin ruido, lo que facilita la detección de bordes y contornos circulares.
- Escala constante y sin deformaciones de perspectiva.

Gracias a estas condiciones, la detección mediante el método de **umbral y contornos** resulta precisa, permitiendo identificar correctamente el número y el valor de las monedas presentes, independientemente de la moneda de referencia seleccionada.

Esto ocurre porque la relación entre los radios de las monedas y los reales, al no existir distorsiones, se mantienen constantes y proporcionales.

<div align="center">
  <img src="salidas/monedas_ideal_resultado.jpg" width="50%">
</div>

**Imagen no ideal**

<a name="ejemplo1"></a>
_Ejemplo 1_
<div align="center">
  <img src="salidas/Monedas1_resultado.jpg" width="75%">
</div>

<a name="ejemplo2"></a>
_Ejemplo 2_
<div align="center">
  <img src="salidas/monedas2_resultado.jpg" width="75%">
</div>

<a name="ejemplo3"></a>
_Ejemplo 3_
<div align="center">
  <img src="salidas/monedas3_resultado.jpg" width="75%">
</div>

En las anteriores situacciones, se ha trabajado con imágenes **no ideales**, donde las condiciones son más complejas y no garantizan una detección perfecta, como en la imagen ideal.
Algunas características de las imágenes no ideales son:
- Iluminación irregular, con sombras y reflejos que dificultan la segmentación.
- Distorsiones de perspectiva, que alteran la proporción entre el radio detectado y el real.
- Objetos que no son monedas presentes en la escena, aumentando la probabilidad de falsos positivos.

Como resultado, algunas monedas no se detectan correctamente. Por ejemplo, en el [Ejemplo 1](#ejemplo1), monedas de 1€ pueden detectarse únicamente por la parte plateada, mientras que la parte dorada no se reconoce.

Asimismo, algunos valores **no se reconocen correctamente** debido a pequeñas variaciones en el tamaño detectado de las monedas o al solapamiento parcial con otras monedas u objetos. Esto puede provocar que algunas monedas sean clasificadas erróneamente, hecho que ocurre en todos los ejemplos. 

[Véase:  [Ejemplo 1](#ejemplo1),  [Ejemplo 2](#ejemplo2),  [Ejemplo 3](#ejemplo3) ]

Asimismo, algunos valores **no se reconocen correctamente** debido a pequeñas variaciones en el tamaño detectado de las monedas o al solapamiento parcial con otras monedas u objetos. Esto puede provocar que monedas de 0.10€, 0.20€ o 0.50€ sean clasificadas erróneamente o marcadas como “no confiables”.


Además, el **valor detectado** puede depender de la **moneda de referencia seleccionada**, pues las proporciones reales entre radios se ven afectadas por sombras o deformaciones, lo que provoca variaciones en la estimación final.

Tal y como se comentó anteriormente, el método de **HoughCircles** resulta más robusto frente a variaciones de iluminación o fondos complejos

---
<a name="tarea2"></a>
## TAREA 2: La tarea consiste en extraer características (geométricas y/o visuales) de las tres imágenes completas de partida, y *aprender* patrones que permitan identificar las partículas en nuevas imágenes. 
- **Salida:**
  - [`salidas/comparacion_real_predicha.jpg`](salidas/comparacion_real_predicha.jpg)
  - [`salidas/matriz_confusion.jpg`](salidas/matriz_confusion.jpg)

Esta tarea implementa un **sistema de clasificación de microplásticos** en imágenes, utilizando técnicas de **visión por computador**. A partir de imágenes de referencia, el sistema **extrae características geométricas y de color** de cada partícula, entrena un clasificador simple basado en distancia euclidiana ponderada, y evalúa su rendimiento en una imagen de prueba con anotaciones.

### ⚙️ Funciones principales

A continuación se describen las principales funciones implementadas para llevar a cabo este proceso:
```py 
detectar_caracteristicas(contorno, imagen_hsv)
```
Extrae un conjunto de **características geométricas y de color** a partir de un contorno detectado:  
- Área, perímetro, compacidad, excentricidad.  
- Relaciones de forma (ancho/alto, área relativa, distancias al centroide).  
- Estadísticas del color en HSV (media y desviación).
---

```py 
extraer_caracteristicas_imagen(imagen_path)
```
Procesa una imagen completa:  
- Convierte a escala de grises.  
- Aplica desenfoque y umbral adaptativo para separar objetos del fondo.  
- Encuentra contornos y calcula sus características con la función anterior.
---

```py
vector_caracteristicas_medio(imagen_path)
```
- Obtiene el **vector medio de características** de todos los objetos de una imagen.
- Se usa para representar cada clase (tipo de microplástico) de forma promedio.
---

```py
entrenar_clasificador(imagenes_referencia)
```
- Calcula el vector de características medio para cada clase (por ejemplo, FRA, PEL, TAR) usando imágenes de referencia. Estos serán la **base del clasificador**.
---

```py 
preparar_referencias(mean_vectors)
```
- Normaliza los vectores de referencia con `StandardScaler` y define un **vector de pesos** que ajusta la importancia de cada característica (por ejemplo, más peso al color o a la forma).  

La normalización es importante porque las características pueden tener escalas muy diferentes (por ejemplo, el área puede tener valores miles de veces mayores que la compacidad o el color), lo que haría que unas dominen sobre otras al calcular distancias o similitudes. 
Al usar `StandardScaler`, todas las características se transforman para tener **media cero** y **varianza unitaria**, garantizando que ninguna domine sobre las demás.
Luego, los pesos permiten dar más relevancia a las características más discriminativas según el contexto, logrando comparaciones más justas y representativas.

---

```py 
clasificar_contorno(...)
```
- Dada una región de interés (bounding box), calcula las características del contorno y las compara con las referencias.
- Clasifica el objeto según la **distancia euclidiana ponderada más corta**.
---

```py 
clasificar_imagen_con_anotaciones(...)
```
Procesa una imagen completa y sus anotaciones (desde un CSV):  
- Carga la imagen de test (`MPs_test.jpg`) y sus anotaciones (`MPs_test_bbs.csv`).
- Detecta los contornos y clasifica cada uno de los objetos dentro de las regiones anotadas.
- Devuelve las etiquetas reales (`y_true`), las predichas (`y_pred`) y la imagen combinada con ambas visualizaciones.
---

```py 
mostrar_resultado_visual(y_true, y_pred, imagen_combined)
```
- Muestra visualmente los resultados de clasificación (reales vs predichos) y calcula la precisión global del modelo.

<div align="center">
  <img src="salidas/comparacion_real_predicha.jpg" width="80%">
</div>

Tal y como se observa en la imagen, la parte izquierda muestra la clasificación real (según las anotaciones del archivo CSV) y la parte derecha muestra la clasificación predicha por el modelo.

Cada tipo de microplástico está representado con un color diferente:
* 🟥 Fragmentos (FRA): contornos y etiquetas en rojo.
* 🟩 Pellets (PEL): contornos y etiquetas en verde.
* 🟦 Alquitrán (TAR): contornos y etiquetas en azul.

De esta forma, es posible comparar visualmente de un vistazo las predicciones y los aciertos.
En general, la mayoría de los objetos coinciden correctamente entre ambas imágenes, aunque se aprecian algunas discrepancias: 
- Algunos fragmentos rojos fueron clasificados erróneamente como pellets verdes, lo que coincide con la confusión detectada en la matriz de confusión.
- Las partículas de alquitrán azul suelen estar correctamente clasificadas, aunque en ciertos casos con forma irregular o bordes difusos el modelo las confundió con fragmentos.
- La distribución general de colores muestra una buena correspondencia global entre las clasificaciones reales y las predichas, lo que respalda el resultado cuantitativo obtenido (accuracy ≈ 72 %).

---
```py 
mostrar_matriz_confusion(y_true, y_pred, clases)
```
- Genera un **heatmap de la matriz de confusión** con `Seaborn` para visualizar los aciertos y errores del clasificador.

La matriz de confusión es una herramienta que muestra, para cada clase real, cómo el clasificador asignó sus **predicciones**. En ella, cada **fila** representa las **etiquetas reales** y cada **columna** las **etiquetas predichas**.
De este modo, la matriz permite visualizar para cada clase el número de muestras clasificadas correctamente (cuando la predicción coincide con la etiqueta real) y el número de muestras clasificadas incorrectamente como pertenecientes a alguna de las otras clases.

La matriz de confusión obtenida es: 

<div align="center">
  <img src="salidas/matriz_confusion.jpg" width="50%">
</div>

Interpretación:
- **Diagonal principal (37, 24, 9):** son los aciertos del modelo → el objeto se clasificó correctamente.  
- **Fuera de la diagonal:** representan errores de clasificación (confusiones entre clases).  

#### 🔍 Análisis detallado
- La clase **FRA** (fragmentos) tuvo **37 aciertos**, pero fue confundida **7 veces con PEL** (pellets) y **3 veces con TAR** (tiras).  
  Esto sugiere que algunos fragmentos comparten **formas o colores similares a los pellets**, lo que genera confusión.
  
- La clase **PEL** tuvo un rendimiento sólido (**24 aciertos**, 7 errores hacia FRA).  
  Nuevamente, la mayor confusión se da entre **FRA ↔ PEL**, lo que indica que ambas clases tienen **características geométricas o cromáticas parecidas**.

- La clase **TAR** obtuvo **9 aciertos**, pero también se confundió en **3 casos como FRA** y **2 como PEL**.  
  Al ser posiblemente más oscura o de forma irregular, puede haberse interpretado erróneamente según la iluminación o textura.

En conjunto, el **patrón de confusión dominante** es entre **fragmentos (FRA)** y **pellets (PEL)**, lo que sugiere que el modelo podría beneficiarse de:
- Aumentar el peso de las características **de color** (HSV).  
- Añadir más imágenes de entrenamiento para ambos tipos.  
- Aplicar segmentación más precisa para evitar que fragmentos incompletos afecten el cálculo de características.  

Sin embargo, se intentó cambiar el peso de las distintas características pero, el resultado obtenido era peor que este pues, a la vez que mejoraba un elemento en concreto, los otros dos empeoraban.

---

```py 
imprimir_metricas(y_true, y_pred)
```
Esta función calcula las métricas del clasificador:
| **Métrica** | **Descripción** | **Fórmula** | **Valor (%)** |
|--------------|------------------|--------------|----------------|
| **Exactitud (Accuracy)** | Porcentaje total de clasificaciones correctas sobre todas las predicciones. | `Accuracy = (TP + TN) / (TP + TN + FP + FN)` | **72.16** |
| **Precisión (Precision)** | Qué tan precisas son las predicciones positivas (por clase). Indica el nivel de “falsos positivos” cometidos. | `Precision = TP / (TP + FP)` | **75.25** |
| **Sensibilidad (Recall)** | Mide la capacidad del modelo para detectar correctamente todos los objetos de una clase (minimiza los falsos negativos). | `Recall = TP / (TP + FN)` | **72.16** |
| **F1-Score** | Promedio armónico entre precisión y recall; balance entre exactitud y cobertura. | `F1 = 2 * (Precision * Recall) / (Precision + Recall)` | **73.67** |

> **Leyenda de términos:**  
> - **TP:** Verdaderos Positivos  
> - **TN:** Verdaderos Negativos  
> - **FP:** Falsos Positivos  
> - **FN:** Falsos Negativos  

### 📈 Interpretación de las métricas
- El modelo logra un rendimiento moderado-alto, identificando correctamente alrededor del 72 % de los microplásticos.
- La precisión del 75 % indica que la mayoría de las predicciones son correctas, mientras que un recall similar muestra que el sistema detecta bien las clases, aunque aún pierde algunos objetos.
- El F1-score de 73.67 % refleja un equilibrio adecuado entre precisión y cobertura.

En conjunto, los resultados son satisfactorios considerando la simplicidad del clasificador y la variabilidad visual de las muestras.


> Uso de la IA:
- Explicación de algunas funciones de las librerías OpenCV y MatplotLib
- Refactorización del código para hacerlo modular
- Redacción y mejora de docstrings
- Estructura y redacción del Readme



