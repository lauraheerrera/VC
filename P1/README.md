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
