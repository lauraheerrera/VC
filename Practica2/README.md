# Práctica 2

**Autores:**  
- Laura Herrera Negrín  
- Ayman Asbai Ghoudan

**Universidad:** Universidad de Las Palmas de Gran Canaria  
**Asignatura:** Visión por Computador  

---
## Contenidos
- [Librerías utilizadas](#librerias)
- [Tarea 1 - Conteo por filas (Canny)](#tarea1)
- [Tarea 2 - Sobel + Umbralizado](#tarea2)
- [Tarea 3 - Demostrador en tiempo real](#tarea3)
- [Tarea 4 - Interactivo inspirado en vídeos](#tarea4)

---

<a name= "librerias"></a>
## Librerías utilizadas
[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)  
- Procesamiento de imágenes y vídeo en tiempo real.  
- Detección de bordes (Canny, Sobel).  
- Aplicación de filtros.  

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)  
- Operaciones matriciales.  
- Reducción, conteo y manipulación de píxeles en imágenes.  
- Transformaciones y cálculos estadísticos sobre datos visuales.  

[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23006DBA?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)  
- Visualización de gráficos y resultados de procesamiento.  
- Histogramas y distribuciones de píxeles.  
- Representación por filas y columnas de intensidad.  

[![OS](https://img.shields.io/badge/OS-%2300A300?style=for-the-badge&logo=windows-terminal&logoColor=white)](https://docs.python.org/3/library/os.html)  
- Gestión de directorios y rutas.  
- Automatización de guardado y carga de imágenes de salida.  
---

<a name="tarea1"></a>
## Tarea 1: Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil.
- **Salida:** Imagen [`salidas/filas_canny.png`](salidas/filas_canny.png) mostrando las filas destacadas.

El objetivo de esta tarea es realizar la **cuenta de píxeles blancos por filas** en una imagen, en lugar de hacerlo por columnas. Se busca determinar el valor máximo de píxeles blancos por fila (`maxfil`) y mostrar las filas que tengan un número de píxeles blancos mayor o igual a 0.90 * `maxfil`.

Para ello, primero se debe **cargar la imagen** [`mandril.jpg`](recursos/mandril.jpg) desde el disco y **convertirla a escala de grises**, lo que permite detectar bordes utilizando un único canal. A continuación, se aplica el **detector de bordes Canny** (`cv2.Canny(img, threshold1, threshold2)`), en el que `threshold1` y `threshold2` definen los límites para detectar bordes débiles y fuertes, respectivamente. Los píxeles detectados como bordes se representan en blanco (255) y el resto en negro (0).

Una vez obtenida la imagen de bordes, se realiza el **conteo de píxeles blancos por fila** utilizando `cv2.reduce` con la operación de suma. Este valor se normaliza dividiendo por el número de columnas y el valor máximo del píxel (255), obteniendo así el **porcentaje de píxeles blancos por fila**.  

Posteriormente, se determina el **valor máximo de píxeles blancos por fila (`maxfil`)** y se define un **umbral del 90%** de este valor para identificar las filas más relevantes. Se recorren todas las filas y se seleccionan aquellas cuyo porcentaje de píxeles blancos es mayor o igual al umbral. Los resultados se muestran en consola indicando el número de filas que cumplen la condición y sus índices.

Finalmente, se realiza una **visualización gráfica** que incluye:  
1. La imagen resultante del detector de bordes Canny.  
2. Un gráfico del porcentaje de píxeles blancos por fila, con una línea que indica el umbral del 90% de `maxfil`.

![Resultado Canny por filas](salidas/filas_canny.png)

Esta visualización permite identificar de manera clara qué filas tienen la mayor concentración de bordes y evaluar la distribución vertical de los contornos en la imagen.

El análisis mostró que la **fila 12** tiene la mayor concentración de bordes en la imagen, alcanzando aproximadamente el **43% de píxeles blancos**. Además, se identificaron otras **6 filas** que superan el 90% del valor máximo, lo que indica que existen varias zonas horizontales con alta densidad de bordes. 

En el resultado gráfico, se puede apreciar que en la imagen se remarcan en color rojo las filas que superan el 90% del número máximo de píxeles no nulos. Se aprecia que estas líneas se concentran principalmente en la parte alta de la cara del mandril, coincidiendo con las zonas de mayor contraste.

<a name="tarea2"></a>
## Tarea 2: Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. 
- **Salida:** Imágenes comparativas:
  - [`salidas/sobel_umbralizado.png`](salidas/sobel_umbralizado.png)
  - [`salidas/comparativa_sobel_canny.png`](salidas/comparativa_sobel_canny.png)
  - [`salidas/sobel_vs_canny.png`](salidas/sobel_vs_canny.png)

El objetivo de esta tarea es **aplicar un umbral a la imagen resultante del filtro Sobel** y posteriormente realizar el **conteo de píxeles blancos por filas y columnas**, similar al análisis realizado previamente con la salida de Canny. A partir de ello,, busca determinar los valores máximos de píxeles blancos por fila y columna, y resaltar las filas y columnas que superen el **90% de este máximo**.

Para ello, al igual que se hizo anteriormente, se cargará la imagen [`mandril.jpg`](recursos/mandril.jpg) desde disco y se convierte a escala de grises para el filtro de Canny y se aplica un **desenfoque gaussiano** para suavizar el ruido antes del cálculo del gradiente para Sobel.

A continuación para detectar los bordes con Sobel, se calculan los gradientes horizontales (`sobelx`) y verticales (`sobely`) usando el operador Sobel y se obtiene la magnitud del gradiente, combinando ambos gradientes y convirtiéndolo a 8 bits (`sobel8u`).  

Ahora, se aplica un **umbral fijo** a la imagen de Sobel para obtener una imagen binaria (`sobel_umbral`), donde los píxeles de borde se representan en blanco (255).

Con la imagen umbralizada, se realiza el **conteo de píxeles blancos por filas y columnas**, similar al procedimiento realizado en los ejemplos con Canny. Esto permite determinar qué filas y columnas contienen la mayor concentración de bordes. Con ello, se calcula el **valor máximo por fila y por columna**, y se seleccionan aquellas filas y columnas cuyo número de píxeles blancos supera el **90% de este máximo**.
Finalmente, se visualiza en un gráfico el resultado:
   - Las filas que superan el 90% del máximo se remarcan en **rojo** 
   - Las columnas que superan el 90% del máximo se remarcan en **azul**.

<div align="center">
  <img src="salidas/sobel_umbralizado.png" width="50%">
</div>

Se han remarcado más filas que columnas, lo que se puede concluir que, tras umbralizar la imagen, los bordes horizontales son más predominantes que los verticales. Esto indica que hay más cambios de intensidad a lo largo de la dirección vertical (generando bordes horizontales) que a lo largo de la dirección horizontal. En la imagen del mandril, esto se traduce en que los rasgos faciales y las zonas de contraste, como la frente, los ojos o el pelaje, presentan transiciones de intensidad más marcadas horizontalmente.

### ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?
Para responder a esta pregunta, se ha seguido un procedimiento similar al que se hizo con Sobel para Canny:
1. Convertir la imagen a una escala de grises para detectar los bordes.
2. Aplicar el algoritmo de Canny para detectar los bordes de la imagen.
3. Calcular los valores máximos de los bordes a lo olargo de cada columna y fila.
4. Aplicar un umbral del 90% del valor mázimo para identificar qué filas y columnas son lass más significaticas.

Con ello, se obtuvo una visualización comparativa que permite apreciar las diferencias en la detección de bordes entre ambos métodos:

![Comparativa Sobel y Canny](salidas/sobel_vs_canny.png)

Numéricamente, los resultados son los siguientes:

| Método | Max píxeles por fila | Filas ≥ 0.9·max | Max píxeles por columna | Columnas ≥ 0.9·max |
| ------ | -------------------- | --------------- | ----------------------- | ------------------ |
| Sobel  | 0.3320               | 13              | 0.2969                  | 3                  |
| Canny  | 0.4297               | 7               | 0.3652                  | 19                 |

**Distribución y calidad de los bordes**
* Sobel detecta principalmente bordes horizontales, generando contornos gruesos y dispersos, con más ruido y redundancia.
* Canny resalta bordes finos, continuos y precisos, especialmente verticales, centrados en los contornos relevantes (ojos, hocico, contorno facial).

**Interpretación visual**
* En las imágenes con líneas, se observa que Sobel marca más filas dispersas, mientras que Canny resalta las columnas principales y los contornos importantes.
* Esto refleja la mayor selectividad y robustez del detector de Canny frente a Sobel umbralizado.

**Conclusión final**
* Sobel ofrece un panorama amplio de cambios de intensidad, marcando más filas dispersas.
* Canny es más selectivo, resaltando columnas principales y contornos estructurales importantes.
  
Se puede concluir que Sobel proporciona un mapeo más amplio de bordes, especialmente horizontales aunque con algo ruido, mientras que Canny concentra su detección en bordes significativos y continuos, ofreciendo una visión más limpia y estructural de la imagen.

<a name="tarea3"></a>
## Tarea 3: Proponer un demostrador que capture las imágenes de la cámara, y les permita exhibir lo aprendido en estas dos prácticas ante quienes no cursen la asignatura :). Es por ello que además de poder mostrar la imagen original de la webcam, permita cambiar de modo, incluyendo al menos dos procesamientos diferentes como resultado de aplicar las funciones de OpenCV trabajadas hasta ahora.
- **Modo de uso:** Teclas `1`, `2`, `3` para cambiar de modo; `ESC` para salir.

El objetivo de esta tarea es aplicar los conocimientos adquiridos en las prácticas anteriores. Para ello, se capturará vídeo desde la webcam y se aplicarán distintos efectos visuales en tiempo real.

El programa permitirá alternar entre diferentes modos de visualización utilizando el teclado:
* `1` → Modo original: muestra la imagen de la webcam sin ningún efecto, tal como se captura.
* `2` → Modo pixelado con mapa de colores: reduce la resolución de la imagen para crear un efecto de pixelado y, posteriormente, aplica un mapa de colores para resaltar las diferentes intensidades.
* `3` → Modo bordes en color: detecta los bordes de la imagen utilizando el algoritmo de Canny con distintos umbrales y los colorea en azul, verde y rojo, superponiéndolos sobre la imagen original.

Además, en la parte inferior de la ventana se mostrarán instrucciones sobre cómo cambiar de modo y cómo salir del programa.
> Para **salir del programa**, se debe presionar la `tecla ESC`.

#### Modos de visualización

**Modo 1: Original**

Este modo muestra el vídeo capturado por la webcam sin aplicar ningún efecto, simplemente mostrando la imagen tal como se recibe.
```python
vista = frame.copy()
```
<div align="center">
<img width="395" height="295" alt="image" src="https://github.com/user-attachments/assets/700b71f8-3529-44d0-a91a-24f2def55f65" />
</div>

**Modo 2: Pixelado + color**
1. Se reduce la resolución de la imagen para crear un efecto pixelado.
2. Luego se escala de nuevo al tamaño original.
3. Finalmente se aplica un mapa de color JET, que colorea la imagen según intensidad.
  ```python
   cv2.applyColorMap(imagen, cv2.COLORMAP_JET)
   ```
<div align="center">
<img width="395" height="295" alt="image" src="https://github.com/user-attachments/assets/b5211526-bbe3-4aec-ae82-3be7479adcba" />
</div>


**Modo 3: Bordes de movimiento en color**
1. Convierte la imagen a escala de grises.
2. Aplica un desenfoque gaussiano para reducir ruido.
3. Detecta bordes usando Canny con diferentes umbrales:   
   - Azul: 30–60
   - Verde: 60–120
   - Rojo: 120–240
  ```python
   cv2.Canny(imagen, minVal, maxVal)
   ```
4. Superpone los bordes coloreados sobre la imagen original con distintas transparencias.
  ```python
   cv2.addWeighted(img1, alpha, img2, beta, gamma)
   ```
<div align="center">
<img width="395" height="295" alt="image" src="https://github.com/user-attachments/assets/8fa73f5b-5220-4121-9dbd-4f6d661838ec" />
</div>

<a name="tarea4"></a>
## Tarea 4: Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.
El objetivo de esta práctica es explorar cómo el movimiento del usuario frente a la cámara puede transformarse en elementos visuales dinámicos, creando una experiencia interactiva en tiempo real. Para ello, tomando como inspiración los vídeos de referencia, el demostrador:  
- Detecta **objetos en movimiento** mediante sustracción de fondo.  
- Genera **elementos gráficos dinámicos** (círculos de colores) en la posición del movimiento detectado.  
- Permite una **visualización interactiva** mostrando tanto la máscara de movimiento como la imagen real con los elementos gráficos superpuestos.  

Para comenzar, se captura video desde la **webcam** y se aplica un **sustractor de fondo** basado en mezcla de gaussianas (`cv2.createBackgroundSubtractorMOG2`). Este genera una **máscara binaria** que resalta los cambios entre frames, detectando objetos en movimiento. La configuración utilizada (`history=100, varThreshold=80, detectShadows=False`) permite un equilibrio entre sensibilidad y estabilidad, evitando que pequeños ruidos generen falsas detecciones.

A partir de la **máscara de movimiento**, se buscan contornos que representen las **áreas de mayor actividad**. Solo se consideran aquellos contornos cuyo área supere un **umbral mínimo** (1000 píxeles), descartando pequeñas variaciones que podrían interferir con la interacción visual.

Cada contorno identificado genera un círculo de color en la posición central del contorno. Estos tienen un tamaño fijo (`RADIUS = 15`) y un tiempo de vida limitado (`LIFETIME = 50 frames`), lo que crea un efecto visual dinámico: los elementos aparecen cuando hay movimiento y desaparecen progresivamente, generando un rastro visual según la actividad del usuario.

Cada círculo se almacena en la **lista `circles`**, donde se guarda toda la información necesaria para su visualización y gestión: la posición `(x, y)` del centro, el **radio** del círculo, su **color** y el **tiempo de vida restante** (en frames). En cada iteración del bucle principal, se dibujan todos los círculos presentes en la lista y se decrementa su tiempo de vida. Cuando un círculo alcanza cero, se elimina de la lista, lo que permite que los elementos visuales **aparezcan y desaparezcan de forma fluida** manteniendo la interactividad.  

La visualización del demostrador se realiza de manera interactiva: se muestra lado a lado la máscara de movimiento con las zonas que cambian respecto al fondo y la imagen real con los círculos superpuestos. Esto permite al usuario observar simultáneamente cómo se detecta el movimiento y cómo este se traduce en elementos visuales, cerrando el ciclo de interacción.

<div align="center">
<img width="1583" height="598" alt="image" src="https://github.com/user-attachments/assets/51573321-ce9a-4ce9-bbf8-b576770baf28" />
</div>


En la imagen puede observarse cómo, en el panel izquierdo, la máscara de movimiento resalta en blanco las regiones que presentan cambios respecto al fondo. Paralelamente, en el panel derecho se muestra la imagen original, sobre la cual se superponen círculos que marcan esas mismas zonas detectadas, permitiendo visualizar de forma directa la correspondencia entre la detección y su representación en la escena.

> Uso de la IA:
- Explicación de algunas funciones de las librerías **OpenCV** y **MatplotLib**
- Estructura y redacción del Readme
