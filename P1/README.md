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
