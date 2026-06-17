# 🚗 Detección de Placas de Vehículos con YOLOv8n, ExecuTorch y Hugging Face Spaces

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
- Documentar todo el flujo de entrenamiento, exportación y despliegue.

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

# 📊 Dataset Utilizado

El entrenamiento se realizó utilizando el dataset público:

### Vehicle Registration Plates Dataset

Dataset disponible en Roboflow:

https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/1

Características:

- Detección de placas vehiculares.
- Automóviles.
- Motocicletas.
- Anotaciones en formato YOLO.
- Compatible con Ultralytics YOLOv8.

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

> Sustituir los siguientes valores por los obtenidos en la ejecución final del notebook.

| Métrica | Valor |
|----------|--------|
| Precision | XX.XX |
| Recall | XX.XX |
| mAP50 | XX.XX |
| mAP50-95 | XX.XX |

Estas métricas permiten evaluar la capacidad del modelo para localizar correctamente placas vehiculares en imágenes no vistas durante el entrenamiento.

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

Define:

- Imagen base.
- Dependencias.
- Librerías requeridas.
- Comando de ejecución.

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

# 👨‍💻 Autor

**Tatan Pérez**

Proyecto desarrollado para la asignatura:

**Procesamiento Digital de Imágenes**

Universidad de Caldas

2026
