# 🚗 Detección de Placas de Vehículos con YOLOv8n, ExecuTorch y Hugging Face Spaces
 
---
 **Proyecto Final** - Procesamiento digital de imagenes **PDI**
 
**Autores**
 
- *Wilmer Sebastian Perez Cuastumal* - wiperezc@unal.edu.co
- *Daniel Mauricio Mejia Hoyos* - dmejiaho@unal.edu.co
    
**Profesor:** Lucas Iturriago - liturriago@unal.edu.co

**Monitora:** Isabella Valero Mora - lvalerom@unal.edu.co


Facultad de ingeniería y arquitectura

Departamento de ingeniería eléctrica, electrónica y computación
 
2026
 
---
 
## Descripción General
 
Este proyecto implementa un sistema de detección automática de placas de vehículos (automóviles y motocicletas) utilizando técnicas de Visión por Computador basadas en Deep Learning.
 
El modelo fue entrenado utilizando **YOLOv8n (You Only Look Once v8 Nano)** sobre un conjunto de datos especializado en detección de matrículas vehiculares y posteriormente exportado al formato **ExecuTorch (.pte)** para optimizar la inferencia y facilitar su despliegue en entornos ligeros.
 
Finalmente, se desarrolló una aplicación web interactiva mediante **Gradio**, desplegada en **Hugging Face Spaces** utilizando contenedores Docker. La aplicación integra un módulo de **OCR automático (EasyOCR)** que extrae y muestra el texto de cada placa detectada.
 
---
 
## Objetivo
 
Desarrollar una solución completa capaz de:
 
- Detectar placas vehiculares en imágenes cargadas o capturadas con cámara.
- Extraer automáticamente el texto de cada placa detectada mediante OCR.
- Ejecutar inferencias optimizadas mediante ExecuTorch.
- Desplegar el modelo como servicio web.
- Proporcionar una interfaz amigable para usuarios finales.
- Mantener un historial acumulado de inferencias por sesión de usuario.
- Documentar todo el flujo de entrenamiento, exportación y despliegue en Hugging Face Spaces mediante Docker.
---
 
## 📂 Estructura del Repositorio
 
```text
4101134-Procesamiento-Digital-de-Imagenes/
│
└── Proyecto Final/
    │
    ├── License_plate_detection_HF_Space.ipynb
    ├── best.pt
    └── yolo8nbest.pte
```
 
### Archivos
 
| Archivo | Descripción |
|----------|------------|
| `License_plate_detection_HF_Space.ipynb` | Notebook principal para exportación y generación del Space de Hugging Face |
| `best.pt` | Modelo YOLOv8n entrenado en formato PyTorch |
| `yolo8nbest.pte` | Modelo exportado al formato ExecuTorch |
| `app.py` | Aplicación Gradio generada dinámicamente con detección + OCR |
| `Dockerfile` | Configuración del contenedor Docker |
| `README.md` | Configuración y documentación del Space |
 
---
 
# Descripción de la Base de Datos
 
Dataset disponible en Roboflow:
 
https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/1
 
Para el entrenamiento del modelo se utilizó el conjunto de datos **Vehicle Registration Plates Dataset**, desarrollado por **Augmented Startups** y publicado en **Roboflow Universe**. El dataset está diseñado específicamente para tareas de detección de placas vehiculares mediante técnicas de Visión por Computador y Aprendizaje Profundo.
 
## Información General
 
| Característica | Valor |
|----------------|--------|
| Nombre | Vehicle Registration Plates Dataset |
| Autor | Augmented Startups |
| Tipo de tarea | Object Detection |
| Licencia | CC BY 4.0 |
| Número de clases | 1 |
| Clase detectada | License_Plate |
| Plataforma | Roboflow Universe |
 
## Tamaño del Dataset
 
La versión utilizada para este proyecto corresponde a la versión **v1 (original-images)** del dataset, compuesta por un total de **8.823 imágenes anotadas**.
 
### Distribución de los Datos
 
| Conjunto | Cantidad de imágenes | Porcentaje |
|-----------|---------------------|------------|
| Entrenamiento | 6.176 | 70 % |
| Validación | 1.765 | 20 % |
| Prueba | 882 | 10 % |
| **Total** | **8.823** | **100 %** |
 
Esta división permitió entrenar el modelo, ajustar hiperparámetros durante la validación y evaluar el rendimiento final sobre datos no observados previamente.
 
## Formato de Anotación
 
El dataset puede exportarse en múltiples formatos compatibles con los principales frameworks de detección de objetos:
 
- YOLOv8
- YOLOv5
- YOLOv7
- YOLOv11
- YOLOv12
- COCO JSON
- Pascal VOC XML
- TFRecord
- YOLO Darknet
Para este proyecto se utilizó el formato **YOLOv8**, compatible con la librería **Ultralytics** empleada durante el entrenamiento.
 
## Características de las Imágenes
 
Las imágenes contienen escenarios reales de tráfico donde aparecen:
 
- Automóviles particulares.
- Motocicletas.
- Vehículos en movimiento.
- Vehículos estacionados.
- Diferentes condiciones de iluminación.
- Diferentes ángulos de captura.
- Oclusiones parciales de las placas.
Estas características permiten entrenar modelos con una mayor capacidad de generalización frente a situaciones reales de operación.
 
## Justificación de la Selección del Dataset
 
Este conjunto de datos fue seleccionado debido a:
 
- Su amplio número de imágenes anotadas.
- La disponibilidad pública para investigación académica.
- La compatibilidad con YOLOv8.
- La presencia de placas vehiculares en diferentes condiciones ambientales.
- Su amplia utilización en proyectos de *Automatic Number Plate Recognition (ANPR)*.
---
 
# Entrenamiento del Modelo
 
El entrenamiento se realizó en Kaggle mediante el notebook:
 
### Kaggle Notebook
 
https://www.kaggle.com/code/pereztatan/license-plate-detection
 
Modelo utilizado:
 
```python
YOLOv8n
```
 
Configuración principal:
 
```python
Image Size = 640x640
Task = Object Detection
Classes = 1
Class = License Plate
```
 
---
 
## Resultados del Entrenamiento
 
El modelo YOLOv8n fue entrenado durante 100 épocas utilizando el dataset Vehicle Registration Plates de Roboflow.
 
### Métricas finales
 
| Métrica | Resultado |
|----------|----------|
| Precision | 98.8 % |
| Recall | 96.7 % |
| mAP@50 | 98.4 % |
| mAP@50-95 | 70.2 % |
 
### Pérdidas finales
 
| Métrica | Valor |
|----------|----------|
| Train Box Loss | 0.93 |
| Train Classification Loss | 0.34 |
| Train DFL Loss | 1.05 |
| Validation Box Loss | 1.05 |
| Validation Classification Loss | 0.35 |
| Validation DFL Loss | 1.09 |
 
Estos resultados demuestran una alta capacidad del modelo para detectar placas vehiculares tanto en automóviles como en motocicletas, manteniendo un equilibrio adecuado entre precisión y capacidad de generalización.
 
---
 
# Exportación del Modelo
 
Una vez completado el entrenamiento se obtuvo:
 
```text
best.pt
```
 
Posteriormente el modelo fue exportado a ExecuTorch:
 
```python
best.pt
        ↓
ExecuTorch Export
        ↓
yolo8nbest.pte
```
 
Formato generado:
 
```text
.pte
```
 
Beneficios:
 
- Menor consumo de recursos.
- Inferencia optimizada.
- Compatibilidad con dispositivos Edge.
- Integración sencilla con Hugging Face Spaces.
---
 
## ExecuTorch
 
ExecuTorch es el framework de inferencia ligera desarrollado por Meta para desplegar modelos PyTorch en dispositivos con recursos limitados.
 
Ventajas:
 
- Baja latencia.
- Menor uso de memoria.
- Compatibilidad con CPU.
- Optimización para aplicaciones embebidas.
En este proyecto se utiliza:
 
```text
Runtime: ExecuTorch
Formato: .pte
Backend: CPU
```
 
---
 
# OCR Automático de Placas (EasyOCR)
 
Una vez que el modelo detecta una placa, la aplicación extrae automáticamente el texto visible en ella mediante **EasyOCR**.
 
## Pipeline de reconocimiento
 
```text
Imagen de entrada
        ↓
Modelo ExecuTorch (detección)
        ↓
Bounding Box de la placa
        ↓
Recorte de la región detectada
        ↓
EasyOCR (reconocimiento de texto)
        ↓
Texto de la placa mostrado en pantalla
y guardado en el historial
```
 
## Características del módulo OCR
 
- Biblioteca utilizada: **EasyOCR**
- Idioma configurado: inglés (`en`), compatible con el formato alfanumérico de placas colombianas.
- Ejecución en **CPU** (sin GPU requerida).
- Limpieza automática del texto: se conservan únicamente caracteres alfanuméricos y guiones.
- El texto reconocido se superpone en **color verde** sobre la imagen resultado, diferenciándose del label de detección.
- Si no se reconoce texto en un recorte, se registra `???` en el historial.
## Dependencia en el Dockerfile
 
```dockerfile
RUN pip install --no-cache-dir easyocr
```
 
Adicionalmente se requiere la librería del sistema:
 
```dockerfile
RUN apt-get install -y libgomp1
```
 
> **Nota:** EasyOCR descarga sus modelos de reconocimiento la primera vez que se inicializa el contenedor. Los arranques posteriores utilizan la caché local.
 
---
 
# Aplicación Web
 
La aplicación fue desarrollada utilizando **Gradio 6** y desplegada en Hugging Face Spaces con Docker.
 
## Características
 
- Carga de imágenes desde archivo o cámara web.
- Configuración dinámica del Confidence Threshold.
- Configuración dinámica del IoU Threshold.
- Visualización de detecciones con bounding boxes.
- **Texto OCR de la placa superpuesto en la imagen resultado.**
- **Historial acumulado de inferencias por sesión de usuario.**
- Indicador de estado del modelo (ExecuTorch / PyTorch fallback).
- Indicador de estado del módulo OCR.
## Historial de Inferencias
 
El historial se gestiona mediante `gr.State`, lo que garantiza que cada usuario mantiene su propio registro independiente durante la sesión. Cada fila del historial contiene:
 
| Campo | Descripción |
|-------|-------------|
| Fecha | Marca de tiempo de la inferencia |
| Imagen | Identificador secuencial de la imagen procesada |
| Confidence | Umbral de confianza utilizado |
| IoU | Umbral de IoU utilizado |
| Detecciones | Número de placas detectadas |
| **Placas** | **Texto OCR extraído de cada placa detectada** |
 
Cuando no se detecta ninguna placa, el historial registra `Sin detecciones` en la columna Placas, y la imagen de entrada se muestra en el panel de resultado para confirmar que el frame sí fue procesado.
 
## Compatibilidad con cámara web
 
La aplicación acepta imágenes tanto desde **archivo subido** como desde **cámara web** del dispositivo, habilitadas mediante:
 
```python
gr.Image(
    type="pil",
    sources=["upload", "webcam"]
)
```
 
---
 
# Despliegue en Hugging Face Spaces
 
El notebook genera automáticamente los tres archivos requeridos por Hugging Face:
 
## 1. app.py
 
Implementa:
 
- Carga del modelo ExecuTorch (con fallback a PyTorch/Ultralytics).
- Preprocesamiento de imagen a tensor 640×640.
- Inferencia y postprocesamiento con NMS.
- **Recorte de regiones de placa y extracción de texto con EasyOCR.**
- Superposición de bounding boxes y texto OCR sobre la imagen.
- **Gestión del historial de sesión con `gr.State`.**
- Interfaz Gradio compatible con Gradio 6.
---
 
## 2. Dockerfile 🐳
 
Se creó automáticamente para construir el contenedor del Space.
 
### Dependencias principales
 
- Python 3.10
- ExecuTorch
- Torch 2.11
- TorchVision
- OpenCV (headless)
- NumPy
- Gradio
- Ultralytics
- **EasyOCR**
### Dependencias del sistema
 
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    libgomp1
```
 
### Configuración principal
 
```dockerfile
FROM python:3.10-slim
```
 
El contenedor expone el puerto **7860**, utilizado por Gradio.
 
---
 
## 3. README.md
 
Contiene los metadatos utilizados por Hugging Face para construir el contenedor.
 
```yaml
---
title: Car and Motorcycle License Plate Detection
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---
```
 
---
 
# Space de Hugging Face
 
Aplicación desplegada:
 
https://huggingface.co/spaces/TatanPerez/Car_and_motorcycle_license_plate_detection
 
Funcionalidades:
 
- Detección de placas vehiculares en imágenes o cámara web.
- **Extracción automática del texto de la placa con EasyOCR.**
- Ajuste dinámico de umbrales de confianza e IoU.
- Inferencia mediante ExecuTorch con fallback a PyTorch.
- Visualización inmediata de resultados con texto OCR superpuesto.
- **Historial acumulado de inferencias por sesión.**
---
 
# Flujo Completo del Proyecto
 
```text
Roboflow Dataset
        ↓
Entrenamiento YOLOv8n
        ↓
best.pt
        ↓
Exportación ExecuTorch
        ↓
yolo8nbest.pte
        ↓
Notebook HF Space
        ↓
app.py
Dockerfile
README.md
        ↓
Docker Build
        ↓
Hugging Face Spaces
        ↓
Aplicación Web
        ↓
Detección de placa  ──→  Recorte de región  ──→  OCR (EasyOCR)  ──→  Texto extraído
```
 
---
 
# Tecnologías Utilizadas
 
- Python
- YOLOv8
- Ultralytics
- PyTorch
- ExecuTorch
- OpenCV
- NumPy
- Pandas
- Gradio 6
- **EasyOCR**
- Docker
- Hugging Face Spaces
- Kaggle
- Roboflow
---
 
# Referencias
 
### Roboflow Dataset
 
https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/1
 
### Kaggle Notebook
 
https://www.kaggle.com/code/pereztatan/license-plate-detection
 
### YOLOv8 Documentation
 
https://docs.ultralytics.com
 
### ExecuTorch
 
https://pytorch.org/executorch
 
### EasyOCR
 
https://github.com/JaidedAI/EasyOCR
 
### Hugging Face Spaces
 
https://huggingface.co/spaces
 
---
