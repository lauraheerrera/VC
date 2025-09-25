# Visión por Computador - Prácticas

**Autores:**  
- Laura Herrera Negrín  
- Ayman Asbai Ghoudan

**Universidad:** Universidad de Las Palmas de Gran Canaria  
**Asignatura:** Visión por Computador  

---

## Descripción
Este repositorio contiene todas las prácticas realizadas durante la asignatura de Visión por Computador. Incluye implementaciones en Python, procesamiento de imágenes, detección de objetos, análisis de vídeo y otros proyectos prácticos relacionados con la materia.  

---
## Estructura del repositorio
Cada carpeta de práctica contiene:
- Notebook de la práctica.  
- Los recursos necesarios (imágenes, vídeos, etc.).  
- Un README específico explicando la práctica y los resultados obtenidos.  La estructura típica es:

```bash
.
├── Practica1/
│   ├── VC_P1.ipynb       # Notebook con el código y explicación
│   ├── recursos/             # Imágenes, vídeos y otros archivos necesarios
│   └── README.md             # Explicación de la práctica y resultados
├── Practica2/
│   ├── VC_P2.ipynb
│   ├── recursos/
│   └── README.md
├── Practica3/
│   ├── VC_P3.ipynb
│   ├── recursos/
│   └── README.md
└── README.md                 # Este README general

```

---
## Requisitos
- Python >= 3.11.5 
- Librerías: `OpenCV`, `NumPy`, `Matplotlib`
- Jupyter Notebook

Se recomienda crear un entorno virtual para instalar las dependencias.

---

## Preparar el entorno
Para configurar el entorno y asegurarte de que todas las librerías están disponibles:

1. **Instalar Anaconda o Miniconda** (si no lo tienes) desde [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
2. **Crear un entorno virtual** llamado `vision`:
```bash
conda create -n vision python=3.11
conda activate vision
```
3. Instalar las librerías necesarias:
```bash
pip install numpy opencv-python matplotlib jupyter
```

---
## Uso
1. Clonar el repositorio:  
```bash
git clone https://github.com/tu_usuario/repositorio_vision_computador.git
```
2. Entrar en la carpeta de la práctica deseada:
```bash
cd <carpeta_del_repositorio>
```
3. Abrir Jupyter Notebook
```bash
jupyter notebook <Cuaderno>
```

## Objetivos
- Aplicar técnicas de visión por computador en problemas prácticos.
- Aprender a utilizar librerías como OpenCV, NumPy y Matplotlib en proyectos reales.
- Desarrollar habilidades de análisis de imágenes y vídeos, y presentación de resultados de manera visual.
