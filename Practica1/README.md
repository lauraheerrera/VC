# Práctica 1

Para llevar a cabo esta práctica, lo primero ha sido aprender a configurar el entorno de desarrollo, siguiendo las indicaciones explicadas en el guión. Para ello, se ha creado un entorno virtual con **Anaconda Prompt**, el cual permite instalar y utilizar ciertas librerías para poder trabajar con ellas en **Visual Studio Code**.

---

## Tarea 1: Crea una imagen, p.e. de 800x800 píxeles, con la textura del tablero de ajedrez

Antes de implementar esta tarea, fue necesario investigar cómo es un [tablero de ajedrez](https://image.freepik.com/vector-gratis/tablero-ajedrez-figuras-ajedrez-blanco-negro-ilustracion-vectorial_125869-1622.jpg) : se trata de un cuadrado dividido en **ocho filas** y **ocho columnas**, donde los cuadros se alternan entre color blanco y negro.

Para construir el tablero, se emplea la función **`np.zeros()`** de la librería **NumPy**, que permite crear una matriz rellena de ceros. Se define con la forma **(800, 800, 1)**, lo que indica que la imagen tendrá un alto de 800 píxeles, un ancho de 800 píxeles y un único canal, que corresponde a una imagen en escala de grises.  

Además, se indica que el tipo de datos es **uint8**, que limita los valores de cada píxel a un rango entre 0 y 255. Ya que solo hay un canal:  
- El valor **0** representará al negro.  
- El valor **255** al blanco.  
- Los valores intermedios a diferentes tonos de grises.  

Con esta imagen totalmente en negro creada, el siguiente paso consiste en pintar los cuadros blancos que forman el tablero de ajedrez.  

En un primer intento, se ha hecho utilizando una estrategia de ***fuerza bruta***: se pintaba manualmente cada cuadro indicando sus coordenadas. Sin embargo, este método resultó ser poco práctico ya que obligaba a definir individualmente los 32 cuadros blancos.  

Para optimizar el proceso, se ha optado por ***dos bucles anidados***. En cada posición, se comprueba si la suma de los índices de fila y columna es par. Si esto ocurre, corresponderá a un cuadro blanco en esa posición, asignándole el valor **255**.

Finalmente, con la función **`plt.imshow()`** de librería **Matplotlib**, se muestra la imagen generada.

## Tarea 2: Crear una imagen estilo Mondrian con las funciones de dibujo de OpenCV

El objetivo de esta tarea es generar una imagen inspirada en las obras del pintor **Piet Mondrian**, quien utilizaba rectángulos de colores primarios (**rojo, azul y amarillo**) y líneas negras gruesas para dividir el espacio.

Para construir la imagen, se ha empleado la librería **OpenCV**, que ofrece distintas funciones de dibujo sobre matrices de píxeles.

1. **Definición de parámetros**  
   - Dimensiones de la imagen: **225x330 píxeles**.  
   - Colores que se utilizarán en formato **RGB**: rojo, azul, amarillo y negro.  
   - Grosor fijo de las líneas negras que delimitan los rectángulos: **5 píxeles**.

2. **Creación del lienzo**  
   Se genera la imagen con la función **`np.ones()`** de NumPy, que permite crear una matriz rellena de unos, para obtener un **lienzo en blanco**.  
   Se define con la forma `(alto, ancho, 3)`, indicando que es una imagen a color con tres canales (RGB).

3. **Dibujo de rectángulos y líneas**  
   - Para los rectángulos se utiliza la función:  
     ```python
     cv2.rectangle(imagen, pt1, pt2, color, grosor)
     ```  
     Donde:  
     - `pt1` → esquina superior izquierda del rectángulo  
     - `pt2` → esquina inferior derecha  
     - `color` → color en formato **BGR**  
     - `grosor` → si se pasa `-1`, el rectángulo queda relleno  

   - Para las líneas se usa la función:  
     ```python
     cv2.line(imagen, pt1, pt2, color, grosor)
     ```  
     Donde `pt1` y `pt2` son los puntos inicial y final de la línea.

De esta forma, combinando rectángulos de colores primarios y líneas negras, se construye una **imagen digital con el estilo característico de Mondrian**.

## TAREA 3: Modificación de los planos de color en una imagen con OpenCV

El objetivo de esta tarea es realizar diferentes cambios visuales sobre los planos de color **(R, G, B)** de una imagen.  

Para ello, se emplea la función `cv2.VideoCapture(0)` de la librería **OpenCV**, que permite acceder a la cámara y capturar fotogramas de en tiempo real.  
En cada lectura:  
- `ret` indica si la captura fue exitosa.  
- `frame` almacena la imagen en formato de matriz con los valores de color en **BGR**, lista para su procesamiento.  

Una vez verificada la lectura correcta del fotograma:

1. **Obtención de dimensiones**: Con `frame.shape` se obtienen la altura (`h`), el ancho (`w`) y el número de canales de color (`c`).  

2. **Separación de canales**: Se dividen los planos en **rojo (r)**, **verde (g)** y **azul (b)** para trabajar de forma independiente sobre cada uno.  

3. **Transformaciones realizadas**  
   - **Mostrando únicamente un color (rojo, verde o azul):** Se utiliza `cv2.merge` para combinar el canal deseado con matrices de ceros en los demás.  
     Ejemplo: `[0, 0, r]` deja visibles únicamente los valores del plano rojo.  
   - **Invirtiendo los colores (rojo, verde o azul):** Se calcula el complemento restando cada canal a 255 (`255 - r`, `255 - g`, `255 - b`) y se combinan los resultados con ceros en los otros planos.  
   - **Añadiendo ruido en el canal rojo:** Se genera una matriz de valores aleatorios con `np.random.randint(0, 200)` y se suma al canal rojo con `cv2.add`. Luego, se reconstruye la imagen con `cv2.merge([0, 0, r_ruido])`, produciendo un efecto de puntos brillantes.  
   - **Añadiendo franjas blancas en el canal verde:** Se modifica directamente la matriz del canal verde asignando 255 en ciertas filas y columnas (`g[100:120, :] = 255`, `g[:, 150:170] = 255`). Finalmente, se combina con ceros en los otros canales: `cv2.merge([0, g, 0])`.  
   - **Aumentando la saturación en el canal azul:** Se multiplica el plano azul por un factor (`b * 4`) y se limita el rango con `np.clip`. Después, se reconstruye la imagen con `cv2.merge([b_sat, 0, 0])`, intensificando únicamente el azul.
  
4. **Creación del collage 3x3**  
   Para mostrar todas las versiones, se emplean las funciones:  
   - `np.hstack()` → apila imágenes horizontalmente.  
   - `np.vstack()` → apila imágenes verticalmente.  
   De esta manera, se genera una cuadrícula 3x3 con los canales originales, invertidos y modificados.  

5. **Visualización**  
   - Con `cv2.resize`, el collage se redimensiona a un tamaño mayor para visualizarlo correctamente. 
   - `cv2.imshow()` abre una ventana en la que se muestran los fotogramas en tiempo real.  
   - El programa se mantiene en ejecución hasta detectar la tecla **ESC** (`cv2.waitKey(20) == 27`).
     
6. **Liberación de recursos**  
   Al finalizar, se liberan los recursos con:  
   - `vid.release()` → libera la cámara.  
   - `cv2.destroyAllWindows()` → cierra todas las ventanas abiertas por OpenCV.
  
## Tarea 4: Pintar círculos en las posiciones del píxel más claro y oscuro de la imagen

Para llevar a cabo esta tarea, se utilizó una imagen con una gran variedad de colores, con el fin de localizar y marcar los píxeles de mayor y menor intensidad.  
<div align="center">
  <img src="https://github.com/user-attachments/assets/16a632b2-da61-44b2-b673-20500fae632f" width="50%">
</div>

En primer lugar, se lee la imagen en color con la función `cv2.imread()` de **OpenCV** y posteriormente se convierte a escalas de grises con `cv2.cvtColor()`. Esto permitirá calcular los valores mínimo y máximo de intensidad, junto con sus posiciones dentro de la imagen, utilizando la función `cv2.minMaxLoc()`.  

Con las coordenadas del píxel más oscuro (`min_loc`) y del más claro (`max_loc`), devueltas por la función, se dibujan dos círculos:  
- Un círculo negro para marcar el píxel más oscuro.  
- Un círculo blanco para señalar el píxel más claro.  

Estos han sido generados con la función `cv2.circle()`, que toma como parámetros el centro, el radio, el color y el grosor del trazo. Además, a cada uno de ellos se le añade un texto descriptivo mediante la función `cv2.putText()`, con el fin de identificarlos claramente.  

Finalmente, la imagen modificada se convierte a su formato original RGB para poder mostrarse correctamente con **Matplotlib**, eliminando los ejes para una presentación más limpia.  

De esta forma, se identificaron visualmente los puntos extremos de luminosidad dentro de la imagen, facilitando el análisis de contraste entre distintas zonas.  

### ¿Si quisieras hacerlo sobre la zona 8x8 más clara/oscura?
En este caso, en lugar de localizar **píxeles individuales** de máxima y mínima intensidad, se busca la **zona más clara y la más oscura de la imagen** dentro de bloques de **8 píxeles x 8 píxeles**.

**Procedimiento**
- Se definen variables que almacenan los valores de intensidad media mínima y máxima (`min_mean`, `max_mean`), junto con las posiciones de sus bloques.
- Mediante **dos bucles anidados**, se recorre la imagen en pasos de 8 píxeles tanto en ancho como en alto.
- Para cada bloque, se calcula la media de intensidad con `np.mean()`.
- Cada vez que se encuentra un bloque más oscuro o más claro que los registrados previamente, se actualizan las variables de referencia junto con sus coordenadas.  

Una vez localizadas las zonas extremas:
   - Se dibuja un rectángulo negro en el bloque más oscuro.
   - Se marca con un rectángulo verde el bloque más claro.

En lugar de utilizar **círculos sobre píxeles concretos**, aquí se destacan **áreas completas de 8x8 píxeles**, lo que permite un análisis más global de regiones homogéneas de luminosidad.  

> 💡 Se ha elegido el color **verde** para el bloque más claro ya que, al coincidir la posición del bloque de intensidad mínima con un borde de la imagen, el color blanco no se apreciaba correctamente.

## Tarea 5: LLevar a cabo una propuesta propia de Pop Art
En esta ocasión, dada la posibilidad de desarrollar una idea por libre, se tomó como inspiración parte de lo ya implementado en los ejemplos expuestos en el cuaderno, 
además de indagar un poco el potencial de las librerías que se han empleado hasta ahora.  
Lo que se ha decidido para formar este *Pop Art*, es fragmentar el marco de cámara en cuatro paneles, cada uno de los cuales presentará un “filtro” distinto. A continuación, se procederán a dar los detalles de la idea tomada.  
Primeramente, se especificó con **`cv2.VideoCapture(0)`** que se leerá la imagen que proviene de la webcam, así como indicar el ancho y alto del que dispondrá cada uno de los 
fotogramas captados por la cámara. Siempre y cuando la lectura de estos frames sea correcta, se permitirá la ejecución del programa. Seguidamente, con ayuda del método **`cv2.resize()`**, 
se le ha asignado al propio frame que se tiene en lectura, las medidas ya comentadas: *ancho* y *alto*.  
Tal y como se introdujo previamente, esta tarea consta de cuatro paneles. El primero y el cuarto de ellos, vienen dados por la función **`cv2.applyColorMap()`**, que según el argumento 
que se indique como parámetro, establecerá una paleta de colores u otra a la imagen en formato BGR. Respectivamente, se escogieron los filtros **`cv2.COLORMAP_JET`**, que representa una 
especie de mapa de calor, y **`cv2.COLOMAP_INFERNO`**.  
Sin embargo, para el segundo y tercer panel, se llevó a cabo una lógica con el fin de pixelar la imagen. Dicha lógica, consiste en encoger cada frame con una proporción determinada 
(para el segundo panel fue un factor de ocho, y para el tercero uno de 12). Esto es debido a que cada imagen contiene numerosos píxeles, y para que se logre ver pixelada, cada bloque 
ha de abarcar más espacio, evitando el suavizado de la imagen. Es decir, exponiendo el caso del segundo panel, los píxeles se encogen en una proporción de ocho, de forma que al 
agrandarlos nuevamente, simplemente se estiran y acabarán representando un espacio de 8x8. En cuanto al color que tomarán estos frames, los argumentos **`cv2.INTER_AREA`** y **`cv2.INTER_NEAREST`**
se encargan de mezclar y promediar los colores en el proceso de encoger la imagen, y duplicarlos en el momento de la redimensión.  
Finalmente, se apilan los diferentes marcos dentro de la ventana de ejecución por medio de los métodos **`np.hstack()`** y **`np.vstack()`**.

> Uso de la IA:
- Explicación de algunas funciones de las librerías **OpenCV** y **MatplotLib**
- Estructura y redacción del Readme
- Obtención de ideas originales para las tareas 3 y 5
