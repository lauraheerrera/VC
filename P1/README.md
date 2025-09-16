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

