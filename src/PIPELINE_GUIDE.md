# Guia Completa del Pipeline: Deteccion de Alzheimer con IA

## Indice

1. [Que hace este proyecto](#1-que-hace-este-proyecto)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [Requisitos de hardware y software](#3-requisitos-de-hardware-y-software)
4. [Instalacion paso a paso](#4-instalacion-paso-a-paso)
5. [Organizar las imagenes ADNI](#5-organizar-las-imagenes-adni)
6. [Convertir DICOM a NIfTI](#6-convertir-dicom-a-nifti)
7. [Preprocesar con FastSurfer](#7-preprocesar-con-fastsurfer)
8. [Verificar los datos](#8-verificar-los-datos)
9. [Entrenar los modelos](#9-entrenar-los-modelos)
10. [Diagnosticar un paciente nuevo](#10-diagnosticar-un-paciente-nuevo)
11. [Interpretar el resultado](#11-interpretar-el-resultado)
12. [Advertencias importantes](#12-advertencias-importantes)
13. [Preguntas frecuentes](#13-preguntas-frecuentes)

---

## 1. Que hace este proyecto

Este proyecto implementa un sistema de deteccion automatica de Alzheimer usando imagenes de resonancia magnetica (MRI) 3D del cerebro.

### Como funciona

El sistema analiza la imagen MRI de un paciente y determina si pertenece a una de tres categorias:

- **AD** (Alzheimer's Disease): el paciente tiene Alzheimer
- **MCI** (Mild Cognitive Impairment): el paciente tiene deterioro cognitivo leve, que es la etapa previa al Alzheimer
- **CN** (Cognitively Normal): el paciente es cognitivamente normal

### Arquitectura del modelo

Se usa una red neuronal 3D llamada **ResNet152 con CBAM** (Convolutional Block Attention Module). Esta red:
- Procesa el volumen 3D completo del cerebro (no cortes 2D)
- Tiene un mecanismo de atencion que aprende a enfocarse en las regiones mas afectadas por el Alzheimer: hipocampo, sustancia blanca cerebral y sustancia blanca cerebelar
- Esta basada en el paper: *"Enhanced ROI guided deep learning model for Alzheimer's detection using 3D MRI images"*

### Estrategia One-vs-Rest (OvR)

En lugar de un solo modelo que clasifica en 3 clases a la vez, este proyecto entrena **3 modelos especializados**:

| Modelo | Aprende a detectar |
|--------|--------------------|
| `model_ADNI_AD_vs_rest.h5` | Si una imagen es AD vs (MCI o CN) |
| `model_ADNI_MCI_vs_rest.h5` | Si una imagen es MCI vs (AD o CN) |
| `model_ADNI_CN_vs_rest.h5` | Si una imagen es CN vs (AD o MCI) |

Para diagnosticar, se pasa la imagen por los 3 modelos y se elige la clase con mayor probabilidad. Esta estrategia da mejores resultados que un solo modelo multiclase.

### Dataset

Se usa el dataset **ADNI** (Alzheimer's Disease Neuroimaging Initiative) que contiene imagenes MRI T1 3D de pacientes con AD, MCI y CN. Las imagenes originales vienen en formato DICOM (.dcm).

---

## 2. Estructura de carpetas

```
tu_repo/
│
├── preprocessing/              <- Scripts para preparar los datos
│   ├── 01_run_fastsurfer.bat   <- Corre FastSurfer en todas las imagenes (Windows)
│   ├── 02_verify_data.py       <- Verifica que los datos esten bien preprocesados
│   └── 03_convert_dicom.py     <- Convierte archivos DICOM (.dcm) a NIfTI (.nii.gz)
│
├── utils/                      <- Funciones de procesamiento de imagenes
│   └── image_processing.py     <- CLAHE, masking de ROIs, sharpening, augmentacion
│
├── reading/                    <- Carga de datos para el modelo
│   └── data_loader.py          <- Carga imagenes, genera batches, prepara datasets OvR
│
├── model/                      <- Arquitectura de la red neuronal
│   └── resnet3d_cbam.py        <- ResNet3D con modulo de atencion CBAM
│
├── training/                   <- Entrenamiento del modelo
│   └── train.py                <- Script principal de entrenamiento (editar y correr)
│
├── explainability/             <- Visualizacion de lo que aprende el modelo
│   └── gradcam.py              <- Grad-CAM 3D: muestra que regiones activa el modelo
│
├── report/                     <- Evaluacion y resultados
│   └── metrics.py              <- Matriz de confusion, curvas de accuracy, reportes
│
├── inference/                  <- Diagnostico de imagenes nuevas
│   └── predict.py              <- Script para diagnosticar un paciente nuevo
│
├── data/                       <- Datos (no subir a GitHub si son grandes)
│   ├── raw/                    <- Imagenes originales .dcm organizadas por clase
│   │   ├── AD/
│   │   ├── MCI/
│   │   └── CN/
│   ├── nifti/                  <- Imagenes convertidas a .nii.gz (genera 03_convert_dicom.py)
│   │   ├── AD/
│   │   ├── MCI/
│   │   └── CN/
│   └── processed/              <- Salida de FastSurfer (genera 01_run_fastsurfer.bat)
│       ├── AD/
│       ├── MCI/
│       └── CN/
│
├── requirements.txt            <- Dependencias Python
├── SETUP_GUIDE.md              <- Guia de instalacion resumida
└── PIPELINE_GUIDE.md           <- Este archivo
```

---

## 3. Requisitos de hardware y software

### Hardware recomendado

| Componente | Minimo | Recomendado |
|-----------|--------|-------------|
| GPU | NVIDIA con 4 GB VRAM | NVIDIA RTX 3050 o superior |
| RAM | 16 GB | 32 GB |
| Almacenamiento | 100 GB libres | 500 GB SSD |
| Sistema operativo | Windows 10/11 | Windows 11 |

> El proyecto fue desarrollado con una **NVIDIA GeForce RTX 3050 (6 GB VRAM)** en Windows 11. Mac con Apple Silicon puede usarse para inferencia pero no para entrenamiento con este setup.

### Software necesario

| Software | Para que sirve | Gratuito |
|---------|----------------|----------|
| Docker Desktop | Correr FastSurfer en contenedor | Si |
| Licencia FreeSurfer | Requerida por FastSurfer | Si (registro) |
| Anaconda | Entorno Python aislado | Si |
| dcm2niix | Convertir DICOM a NIfTI | Si |
| Python 3.8 | Lenguaje base | Si |
| TensorFlow 2.10 | Framework de deep learning | Si |

---

## 4. Instalacion paso a paso

### 4.1 Instalar Docker Desktop

1. Ve a: https://www.docker.com/products/docker-desktop/
2. Descarga la version para **Windows**
3. Ejecuta el instalador y sigue el asistente (opciones por defecto)
4. **Reinicia Windows** cuando termine
5. Abre Docker Desktop desde el escritorio y espera hasta que diga **"Engine running"** en la esquina inferior izquierda

**Verificar que funciona:** Abre PowerShell como Administrador y escribe:
```
docker --version
```
Debe aparecer algo como: `Docker version 24.x.x`

### 4.2 Activar soporte GPU en Docker

1. Actualiza los drivers NVIDIA: https://www.nvidia.com/drivers
2. En PowerShell, verifica que tu GPU es visible para Docker:
```
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```
Debe mostrar el nombre de tu GPU, temperatura y memoria disponible.

### 4.3 Obtener licencia gratuita de FreeSurfer

FastSurfer necesita una licencia de FreeSurfer para funcionar. Es completamente gratuita.

1. Ve a: https://surfer.nmr.mgh.harvard.edu/registration.html
2. Llena el formulario con tu nombre, email y universidad
3. En **Operating System** selecciona: **Linux / Intel** (aunque estes en Windows, FastSurfer corre dentro de un contenedor Linux)
4. Recibes un email con el archivo `license.txt` adjunto
5. Crea la carpeta `C:\fastsurfer_license\` en el Explorador de Windows
6. Guarda el archivo dentro: `C:\fastsurfer_license\license.txt`

### 4.4 Descargar FastSurfer

En PowerShell (puede tardar 10-20 minutos, descarga ~6 GB):
```
docker pull deepmi/fastsurfer:latest
```

Cuando termine, verifica:
```
docker images
```
Debe aparecer `deepmi/fastsurfer` en la lista.

### 4.5 Descargar dcm2niix

1. Ve a: https://github.com/rordenlab/dcm2niix/releases
2. Descarga el archivo `dcm2niix_win.zip` (version Windows)
3. Descomprime y guarda el `dcm2niix.exe` en: `C:\dcm2niix\dcm2niix.exe`

### 4.6 Instalar Anaconda

1. Ve a: https://www.anaconda.com/download
2. Descarga la version para Windows
3. Ejecuta el instalador con opciones por defecto
4. **NO marques** "Add Anaconda to PATH" (evita conflictos con otros Python que tengas)
5. Al terminar, busca **"Anaconda Prompt"** en el menu de inicio (no uses PowerShell normal para Python)

### 4.7 Crear el entorno Python del proyecto

Abre **Anaconda Prompt** (no PowerShell) y ejecuta estos comandos uno por uno:

```
conda create -n alzheimer python=3.8
```
Cuando pregunte `Proceed ([y]/n)?` escribe `y` y Enter.

```
conda activate alzheimer
```
El prompt cambia de `(base)` a `(alzheimer)`.

```
pip install -r C:\ruta\a\tu_repo\requirements.txt
```

> Cada vez que abras Anaconda Prompt para trabajar con el proyecto, recuerda ejecutar `conda activate alzheimer` primero.

---

## 5. Organizar las imagenes ADNI

### Estructura esperada de los archivos DICOM

Las imagenes ADNI vienen organizadas con una subcarpeta por paciente, y dentro los archivos `.dcm`:

```
C:\alzheimer_data\raw\
    AD\
        002_S_0295\          <- ID del paciente
            archivo001.dcm
            archivo002.dcm
            archivo003.dcm
            ... (muchos archivos por paciente)
        002_S_0413\
            archivo001.dcm
            ...
    MCI\
        002_S_0816\
            archivo001.dcm
            ...
    CN\
        002_S_0938\
            archivo001.dcm
            ...
```

### Reglas importantes

- Cada subcarpeta es un paciente
- El nombre de la subcarpeta sera el ID del sujeto en todo el pipeline
- No uses espacios ni caracteres especiales en los nombres
- Minimo recomendado: **80 pacientes por clase** para que el modelo aprenda algo util

### Crear las carpetas en Windows

Abre PowerShell y copia:
```
mkdir C:\alzheimer_data\raw\AD
mkdir C:\alzheimer_data\raw\MCI
mkdir C:\alzheimer_data\raw\CN
mkdir C:\alzheimer_data\nifti\AD
mkdir C:\alzheimer_data\nifti\MCI
mkdir C:\alzheimer_data\nifti\CN
mkdir C:\alzheimer_data\processed
```

---

## 6. Convertir DICOM a NIfTI

FastSurfer no acepta archivos DICOM directamente. Primero hay que convertirlos al formato NIfTI (`.nii.gz`).

### Editar el script de conversion

Abre `preprocessing/03_convert_dicom.py` con cualquier editor de texto (Bloc de Notas, VSCode, etc.) y cambia estas 3 lineas al inicio del archivo:

```python
# Carpeta donde estan tus .dcm organizados en AD/, MCI/, CN/
INPUT_DIR = r"C:\alzheimer_data\raw"

# Carpeta donde se guardaran los .nii.gz convertidos
OUTPUT_DIR = r"C:\alzheimer_data\nifti"

# Ruta al ejecutable dcm2niix
DCM2NIIX_PATH = r"C:\dcm2niix\dcm2niix.exe"
```

Guarda el archivo.

### Ejecutar la conversion

En **Anaconda Prompt** con el entorno activado:
```
conda activate alzheimer
cd C:\ruta\a\tu_repo
python preprocessing/03_convert_dicom.py
```

Veras mensajes como:
```
[INICIO] 002_S_0295
[OK] 002_S_0295
[INICIO] 002_S_0413
[OK] 002_S_0413
...
CONVERTIDOS: 133  ERRORES: 0
```

El resultado son archivos `.nii.gz` en `C:\alzheimer_data\nifti\AD\`, `MCI\` y `CN\`.

> Si un paciente da error, normalmente es porque sus archivos DICOM estan corruptos o incompletos. Puedes ignorarlos si tienes suficientes pacientes validos.

---

## 7. Preprocesar con FastSurfer

FastSurfer hace dos cosas importantes:
1. **Skull stripping**: elimina el craneo y deja solo el cerebro
2. **Segmentacion**: identifica y etiqueta cada region del cerebro (hipocampo, corteza, etc.)

La salida son dos archivos por paciente:
- `orig.mgz`: imagen del cerebro sin craneo
- `aparc.DKTatlas+aseg.deep.mgz`: mapa de segmentacion con las etiquetas de cada region

### Editar el script de FastSurfer

Abre `preprocessing/01_run_fastsurfer.bat` con el Bloc de Notas y cambia estas 3 lineas:

```bat
set RAW_DIR=C:\alzheimer_data\nifti
set OUTPUT_DIR=C:\alzheimer_data\processed
set FS_LICENSE=C:\fastsurfer_license\license.txt
```

Guarda el archivo.

### Ejecutar FastSurfer

Abre **PowerShell como Administrador** (click derecho en PowerShell → "Ejecutar como administrador"):
```
cd C:\ruta\a\tu_repo
preprocessing\01_run_fastsurfer.bat
```

Veras mensajes como:
```
[INICIO] Procesando: AD\002_S_0295
[OK] Completado: AD\002_S_0295
[INICIO] Procesando: AD\002_S_0413
...
TOTAL: 133  ERRORES: 0
```

### Tiempos aproximados

| Configuracion | Tiempo por paciente | 200 pacientes |
|--------------|--------------------:|:-------------:|
| Con GPU NVIDIA RTX 3050 | ~1-2 minutos | ~3-7 horas |
| Sin GPU (CPU) | ~20-30 minutos | ~70-100 horas |

> El script detecta automaticamente si un paciente ya fue procesado y lo salta. Si interrumpes el proceso, puedes volver a correr el script y continua desde donde quedo.

### Estructura de salida

```
C:\alzheimer_data\processed\
    AD\
        002_S_0295\
            mri\
                orig.mgz                        <- cerebro sin craneo
                aparc.DKTatlas+aseg.deep.mgz    <- segmentacion
        002_S_0413\
            mri\
                orig.mgz
                aparc.DKTatlas+aseg.deep.mgz
    MCI\
        ...
    CN\
        ...
```

---

## 8. Verificar los datos

Antes de entrenar, verifica que todos los pacientes tienen los archivos necesarios:

```
conda activate alzheimer
cd C:\ruta\a\tu_repo
python preprocessing/02_verify_data.py
```

Salida esperada:
```
============================================================
  VERIFICACION DE DATOS PROCESADOS
============================================================

  AD:
    Total sujetos:     135
    Correctos:         133
    Con errores:       2
    Estado:            CON ERRORES

  MCI:
    Total sujetos:     316
    Correctos:         316
    Con errores:       0
    Estado:            OK

  CN:
    Total sujetos:     196
    Correctos:         196
    Con errores:       0
    Estado:            OK

============================================================
  RESUMEN FINAL
============================================================
  Total sujetos encontrados:  647
  Sujetos listos para usar:   645
  Sujetos con errores:        2
============================================================
```

Si hay errores, el script te dira que archivo falta en cada paciente con error. Puedes volver a correr FastSurfer solo para esos pacientes o simplemente ignorarlos si tienes suficientes correctos.

---

## 9. Entrenar los modelos

Se entrenan 3 modelos usando la estrategia One-vs-Rest. Cada entrenamiento puede tardar varias horas.

### Editar la configuracion de entrenamiento

Abre `training/train.py` y localiza la seccion de configuracion al inicio del archivo. Las unicas variables que debes cambiar son:

```python
# ============================================================
# CONFIGURACION - EDITA ESTAS VARIABLES ANTES DE CORRER
# ============================================================

BASE_DIR = r"C:\alzheimer_data\processed"   # <- tu ruta de datos procesados
TASK = "AD_vs_rest"                          # <- cambiar por cada entrenamiento
BATCH_SIZE = 10                              # <- reducir a 4 si hay error de memoria GPU
NUM_EPOCHS = 100                             # <- numero de epocas
```

### Primer entrenamiento: Modelo AD

Con `TASK = "AD_vs_rest"` en el archivo, ejecuta:

```
conda activate alzheimer
cd C:\ruta\a\tu_repo
python training/train.py
```

Veras el progreso en tiempo real:
```
Epoch 1/100
46/46 [==============================] - 120s - loss: 0.6823 - accuracy: 0.5432
Epoch 2/100
46/46 [==============================] - 118s - loss: 0.6201 - accuracy: 0.6123
...
Epoch 47/100 - val_accuracy mejoró de 0.7823 a 0.7941. Guardando modelo...
...
```

Cuando termine, se crea el archivo: `model_ADNI_AD_vs_rest.h5`

### Segundo entrenamiento: Modelo MCI

Abre `training/train.py`, cambia **solo** esta linea:
```python
TASK = "MCI_vs_rest"
```
Guarda el archivo y vuelve a correr:
```
python training/train.py
```
Genera: `model_ADNI_MCI_vs_rest.h5`

### Tercer entrenamiento: Modelo CN

Cambia a:
```python
TASK = "CN_vs_rest"
```
Corre:
```
python training/train.py
```
Genera: `model_ADNI_CN_vs_rest.h5`

### Ver los resultados de cada modelo

Despues de cada entrenamiento puedes ver la matriz de confusion y las metricas:

```
python report/metrics.py ^
    --model model_ADNI_AD_vs_rest.h5 ^
    --data  C:\alzheimer_data\processed ^
    --task  AD_vs_rest
```

### Resumen de archivos generados

Al terminar los 3 entrenamientos tendras:

```
tu_repo\
    model_ADNI_AD_vs_rest.h5     <- modelo para detectar AD
    model_ADNI_MCI_vs_rest.h5    <- modelo para detectar MCI
    model_ADNI_CN_vs_rest.h5     <- modelo para detectar CN
```

Guarda estos 3 archivos en un lugar seguro. Son el resultado de todo el entrenamiento.

---

## 10. Diagnosticar un paciente nuevo

Para diagnosticar una imagen nueva necesitas:
1. La imagen MRI del paciente en formato `.nii.gz`
2. Pasar la imagen por FastSurfer (igual que en el paso 7, pero para un solo paciente)
3. Correr el script de inferencia con los 3 modelos

### Paso A: Preparar la imagen del paciente nuevo

Asegurate de tener el archivo NIfTI del paciente nuevo:
```
C:\paciente_nuevo\
    imagen_paciente.nii.gz
```

### Paso B: Pasar por FastSurfer

En PowerShell como Administrador:

```
docker run --rm --gpus all ^
  -v "C:\paciente_nuevo":/data_in ^
  -v "C:\paciente_nuevo_out":/data_out ^
  -v "C:\fastsurfer_license\license.txt":/fs_license.txt ^
  deepmi/fastsurfer:latest ^
  --t1 /data_in/imagen_paciente.nii.gz ^
  --sid paciente ^
  --sd /data_out ^
  --fs_license /fs_license.txt ^
  --seg_only
```

Esto genera en `C:\paciente_nuevo_out\paciente\mri\`:
- `orig.mgz`
- `aparc.DKTatlas+aseg.deep.mgz`

### Paso C: Obtener el diagnostico

En **Anaconda Prompt**:

```
conda activate alzheimer
cd C:\ruta\a\tu_repo

python inference/predict.py ^
  --subject_dir C:\paciente_nuevo_out\paciente ^
  --mode ovr ^
  --model_ad  model_ADNI_AD_vs_rest.h5 ^
  --model_mci model_ADNI_MCI_vs_rest.h5 ^
  --model_cn  model_ADNI_CN_vs_rest.h5
```

### Paso D (opcional): Ver el mapa Grad-CAM

Para ver que regiones del cerebro activo el modelo:

```
python inference/predict.py ^
  --subject_dir C:\paciente_nuevo_out\paciente ^
  --mode ovr ^
  --model_ad  model_ADNI_AD_vs_rest.h5 ^
  --model_mci model_ADNI_MCI_vs_rest.h5 ^
  --model_cn  model_ADNI_CN_vs_rest.h5 ^
  --gradcam ^
  --save_gradcam gradcam_resultado.png
```

---

## 11. Interpretar el resultado

### Salida del script de diagnostico

```
========================================================
  RESULTADO DEL DIAGNOSTICO (One-vs-Rest)
========================================================
  AD (Alzheimer)               :  87.3%  ########################  <-- DIAGNOSTICO
  MCI (Deterioro cognitivo)    :  45.1%  ###############
  CN (Normal)                  :  12.4%  ####

  Diagnostico mas probable: AD (Alzheimer)
========================================================
  ADVERTENCIA: Resultado orientativo.
  No reemplaza el criterio medico especializado.
========================================================
```

### Como leer los porcentajes

Los porcentajes muestran la probabilidad relativa de cada clase despues de normalizar los 3 modelos OvR. No son probabilidades absolutas sino relativas entre si.

| Porcentaje | Interpretacion |
|-----------|----------------|
| > 70% para una clase | El modelo tiene alta confianza en ese diagnostico |
| 40-70% | Confianza moderada, puede haber ambiguedad |
| < 40% | El modelo tiene dudas, interpretar con precaucion |

Si los 3 porcentajes son similares (ej: 40%, 35%, 25%), el modelo esta inseguro y el resultado debe interpretarse con mucha cautela.

### Mapa Grad-CAM

Si generaste el Grad-CAM, el archivo `.png` muestra cortes axiales del cerebro con un mapa de calor superpuesto:
- **Zonas rojas/amarillas**: regiones que el modelo considero importantes para su decision
- **Zonas azules/frias**: regiones ignoradas por el modelo

En un diagnostico correcto de AD, las zonas rojas deberian aparecer principalmente en el **hipocampo** y la **sustancia blanca**.

---

## 12. Advertencias importantes

**Uso medico**: Este sistema es una herramienta de investigacion academica. Los resultados son orientativos y **NO deben usarse como diagnostico clinico**. El diagnostico de Alzheimer requiere evaluacion por un medico especialista (neurologo o neuropsiquiatra) con pruebas cognitivas, historia clinica y criterios establecidos.

**Calidad de los datos**: El modelo solo es tan bueno como los datos con los que fue entrenado. Si las imagenes de entrenamiento son de baja calidad o el dataset es pequeno, los resultados seran poco fiables.

**Generalizacion**: El modelo fue entrenado con imagenes del dataset ADNI con un protocolo especifico de adquisicion. Imagenes de otros equipos o protocolos diferentes pueden dar resultados menos precisos.

**Balance de clases**: Si una clase tiene muchos mas sujetos que las otras, el modelo puede estar sesgado hacia esa clase. El script balancea automaticamente, pero es importante tener al menos 80 sujetos por clase.

---

## 13. Preguntas frecuentes

**P: Que pasa si no tengo GPU NVIDIA?**
Puedes correr todo en CPU pero sera extremadamente lento. FastSurfer tardaria ~20-30 minutos por imagen (en vez de 1-2 minutos) y el entrenamiento podria tardar semanas en lugar de dias. Se recomienda usar Google Colab con GPU gratuita como alternativa.

**P: Cuantas imagenes necesito como minimo?**
El minimo absoluto es 50 por clase. Con menos de 50 el modelo no aprende patrones confiables. Lo ideal es mas de 100 por clase. El dataset ADNI completo tiene cientos de sujetos por clase.

**P: Puedo usar el Mac M4 para algo?**
Si. El Mac M4 puede usarse para el paso de inferencia (diagnostico) instalando `tensorflow-macos` en lugar de tensorflow. Para FastSurfer y entrenamiento se recomienda Windows con GPU.

**P: Que hago si FastSurfer da error en algunos pacientes?**
Revisa el log de errores que genera el script. Los errores mas comunes son imagenes corruptas o de baja resolucion. Si tienes suficientes pacientes validos (mas de 80 por clase), puedes simplemente ignorar los que dan error.

**P: Puedo agregar mas imagenes despues de entrenar?**
Si, pero tendrias que volver a entrenar los modelos desde cero con el dataset completo. El modelo no se puede "actualizar" facilmente con nuevas imagenes sin reentrenar.

**P: Por que hay 3 modelos en lugar de uno?**
La estrategia One-vs-Rest (un modelo por clase) generalmente da mejores resultados que un solo modelo multiclase, especialmente cuando las clases son dificiles de distinguir entre si (como AD vs MCI). Ademas, permite tener mas control sobre la confianza de cada prediccion individualmente.

**P: Cuanto espacio en disco necesito?**
Aproximadamente:
- Imagenes DICOM originales: ~50-200 GB dependiendo de cuantos pacientes
- Archivos NIfTI convertidos: ~10-30 GB
- Salida de FastSurfer (.mgz): ~20-50 GB
- Modelos entrenados (.h5): ~500 MB cada uno

**P: Puedo usar imagenes de OASIS en lugar de ADNI?**
Si, pero OASIS no tiene pacientes MCI, solo AD y CN. Tendrias que adaptar el pipeline para solo 2 clases. Los notebooks originales del repositorio tienen la implementacion para OASIS AD vs CN.

**P: El modelo puede diagnosticar cualquier imagen MRI?**
Solo imagenes MRI T1 3D del cerebro que hayan sido procesadas con FastSurfer. No funciona con MRI de otras partes del cuerpo, MRI 2D, CT, PET u otras modalidades.
