# 🚗 Detección de Placas de Vehículos con YOLOv8n, ExecuTorch y Hugging Face Spaces

---

# 👨‍💻 Autores

- **Wilmer Sebastian Perez Cuastumal** - wiperezc@unal.edu.co
- **Daniel Mauricio Mejia Hoyos** - dmejiaho@unal.edu.co

Proyecto desarrollado para la asignatura:

**Procesamiento Digital de Imágenes - PDI**

Universidad Nacional de Colombia

2026

---

## 📖 Descripción General

Este proyecto implementa un sistema de detección automática de placas de vehículos (automóviles y motocicletas) utilizando técnicas de Visión por Computador basadas en Deep Learning.

El modelo fue entrenado utilizando **YOLOv8n (You Only Look Once v8 Nano)** sobre un conjunto de datos especializado en detección de matrículas vehiculares y posteriormente exportado al formato **ExecuTorch (.pte)** para optimizar la inferencia y facilitar su despliegue en entornos ligeros.

Finalmente, se desarrolló una aplicación web interactiva mediante **Gradio**, desplegada en **Hugging Face Spaces** utilizando contenedores Docker.

---

# 🎯 Objetivo

Desarrollar una solución completa capaz de:

- Detectar placas vehiculares en imágenes.
- Ejecutar inferencias optimizadas mediante ExecuTorch.
- Desplegar el modelo como servicio web.
- Proporcionar una interfaz amigable para usuarios finales.
- Documentar todo el flujo de entrenamiento, exportación y despliegue en Hugging Face Spaces mediante Docke.

---


# 📂 Estructura del Repositorio

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
| `app.py` | Aplicación Gradio generada dinámicamente |
| `Dockerfile` | Configuración del contenedor Docker |
| `README.md` | Configuración y documentación del Space |

---

# 📊 Descripción de la Base de Datos 

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

# 🏋️ Entrenamiento del Modelo

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

# 📈 Resultados del Entrenamiento

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

# 💾 Exportación del Modelo

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

# ⚡ ExecuTorch

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

# 🤖 Aplicación Web

La aplicación fue desarrollada utilizando:

```text
Gradio
```

Características:

- Carga de imágenes.
- Configuración del Confidence Threshold.
- Configuración del IoU Threshold.
- Visualización de detecciones.
- Tabla de resultados.
- Histograma de confianza.
- Métricas de inferencia.
- Historial de inferencias.

---

# 🐳 Despliegue en Hugging Face Spaces

El notebook genera automáticamente los tres archivos requeridos por Hugging Face:

## 1. app.py

Implementa:

- Carga del modelo ExecuTorch.
- Inferencia.
- Postprocesamiento.
- Interfaz Gradio.

---

## 2. Dockerfile

Se creó automáticamente para construir el contenedor del Space.

#### Dependencias principales:

- Python 3.10
- ExecuTorch
- Torch 2.11
- TorchVision
- OpenCV
- NumPy
- Gradio
- Ultralytics

#### Configuración principal:

- FROM python:3.10-slim

El contenedor expone:

Puerto 7860

utilizado por Gradio

---

## 3. README.md

Contiene los metadatos utilizados por Hugging Face para construir el contenedor.

Ejemplo:

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

# 🌐 Space de Hugging Face

Aplicación desplegada:

https://huggingface.co/spaces/TatanPerez/Car_and_motorcycle_license_plate_detection

Funcionalidades:

- Detección de placas vehiculares.
- Ajuste dinámico de umbrales.
- Inferencia mediante ExecuTorch.
- Visualización inmediata de resultados.

---

# 🔄 Flujo Completo del Proyecto

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
```

---

# 🛠 Tecnologías Utilizadas

- Python
- YOLOv8
- Ultralytics
- PyTorch
- ExecuTorch
- OpenCV
- NumPy
- Pandas
- Plotly
- Gradio
- Docker
- Hugging Face Spaces
- Kaggle
- Roboflow

---

# 📚 Referencias

### Roboflow Dataset

https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/1

### Kaggle Notebook

https://www.kaggle.com/code/pereztatan/license-plate-detection

### YOLOv8 Documentation

https://docs.ultralytics.com

### ExecuTorch

https://pytorch.org/executorch

### Hugging Face Spaces

https://huggingface.co/spaces

---
