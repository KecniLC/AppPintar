# 🎨 AppPintar

¡Hola! Este es un proyecto de visión computacional e inteligencia artificial desarrollado en Python utilizando **OpenCV** y **MediaPipe**. Permite dibujar en el aire en tiempo real utilizando la cámara web y gestos de la mano.

El proyecto nació como una forma de reforzar mi lógica de programación y explorar el procesamiento de imágenes en tiempo real. 🚀

---

## 📺 Demostración en Vivo 

👉 **[Ver Reel del proyecto funcionando en Instagram](https://www.instagram.com/reel/DYM3FLKvcqL/?igsh=MTEwcDc3NGkybWZmdA==)**

---

## ✨ Características y Gestos
El sistema reconoce la mano y activa diferentes funciones según la combinación de dedos levantados:

* **☝️ Dibujar (Solo dedo índice):** Dibuja una línea continua en la pantalla con el color seleccionado.
* **✌️ Cambiar Color (Índice y Medio):** Cambia cíclicamente entre Rojo, Verde, Azul y Amarillo (cuenta con un *cooldown* de 0.8s para evitar cambios bruscos).
* **🖐️ Borrar Lienzo (Todos los dedos):** Limpia por completo el espacio de dibujo de forma instantánea.

---

## 🛠️ Tecnologías Utilizadas
* **Python 3**
* **OpenCV** (Procesamiento de video y renderizado de gráficos)
* **MediaPipe Tasks (Vision)** (Modelo Hand Landmarker para la detección de nodos de la mano)
* **NumPy** (Manipulación del lienzo como matriz de píxeles)

---

## 📦 Requisitos e Instalación

Sigue estos pasos para clonar el proyecto y configurar tu entorno de desarrollo de forma aislada y limpia:

### 1. Crea y activa un entorno virtual (Virtual Environment)
En Windows: python -m venv venv 
.\venv\Scripts\activate

### 2.  Instala las dependencias (MediaPipe y OpenCV)
pip install opencv-python mediapipe numpy

### 3. Ejecuta la app
python dibujo.py
