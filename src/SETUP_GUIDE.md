# Guia de Instalacion y Uso: Pipeline Alzheimer Completo

Esta guia te explica paso a paso como preparar todo para que puedas:
1. Preprocesar imagenes MRI con FastSurfer
2. Entrenar el modelo
3. Subir una imagen nueva y obtener un diagnostico

---

## Maquinas que necesitas

| Maquina | Para que sirve |
|---------|----------------|
| **Windows con GPU NVIDIA** | Correr FastSurfer (rapido) + entrenar el modelo |
| **Mac M4** | Opcional: correr el script de inferencia (diagnostico) |

---

## PASO 1: Preparar Windows con GPU NVIDIA

### 1.1 Instalar Docker Desktop

1. Ve a: https://www.docker.com/products/docker-desktop/
2. Descarga la version para Windows
3. Instala siguiendo el asistente (opciones por defecto)
4. Abre Docker Desktop y espera a que diga "Engine running"

**Verificar que funciona:**
Abre PowerShell (busca "PowerShell" en el menu inicio) y escribe:
```
docker --version
```
Deberia mostrar algo como: `Docker version 24.x.x`

### 1.2 Activar soporte GPU en Docker

1. Instala los drivers NVIDIA mas recientes: https://www.nvidia.com/drivers
2. Instala NVIDIA Container Toolkit:
   - Ve a: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   - Sigue la seccion "Installation on Windows"

**Verificar que funciona:**
```
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
Deberia mostrar informacion de tu GPU.

### 1.3 Obtener licencia gratuita de FreeSurfer

FastSurfer necesita una licencia de FreeSurfer (es gratuita).

1. Ve a: https://surfer.nmr.mgh.harvard.edu/registration.html
2. Completa el formulario con tu email
3. Recibes un email con el archivo `license.txt`
4. Guarda ese archivo en una ubicacion conocida, por ejemplo:
   ```
   C:\fastsurfer_license\license.txt
   ```

### 1.4 Descargar la imagen Docker de FastSurfer

En PowerShell:
```
docker pull deepmi/fastsurfer:latest
```
Esto descarga ~5-8 GB. Hazlo con buena conexion a internet.

**Verificar que funciona:**
```
docker run --rm deepmi/fastsurfer:latest --help
```

### 1.5 Instalar Python en Windows

1. Descarga Anaconda: https://www.anaconda.com/download
2. Instala con opciones por defecto
3. Abre "Anaconda Prompt" (no PowerShell normal) y crea el entorno:

```bash
conda create -n alzheimer python=3.8
conda activate alzheimer
pip install tensorflow==2.10.0
pip install nibabel scikit-image opencv-python dipy scikit-learn matplotlib seaborn
```

> Nota: se usa TF 2.10.0 porque es la ultima version con soporte GPU nativo en Windows.

---

## PASO 2: Organizar tus imagenes MRI

Cuando tu companero te envie las imagenes, organízalas asi:

```
C:\alzheimer_data\raw\
    AD\
        paciente001.nii.gz
        paciente002.nii.gz
        ...
    MCI\
        paciente101.nii.gz
        ...
    CN\
        paciente201.nii.gz
        ...
```

**Reglas:**
- Cada archivo debe ser un MRI T1 3D en formato `.nii` o `.nii.gz`
- El nombre del archivo sera el ID del sujeto (sin espacios, sin caracteres especiales)
- Minimo ~80 sujetos por carpeta para que el modelo aprenda algo razonable

---

## PASO 3: Correr FastSurfer (preprocesar las imagenes)

1. Abre PowerShell como Administrador
2. Ve a la carpeta del proyecto:
   ```
   cd C:\ruta\a\3-Alz-Predict-ROIvsNonROI
   ```
3. Edita el archivo `scripts\01_run_fastsurfer.bat`:
   - Cambia `RAW_DIR` a donde estan tus `.nii.gz`
   - Cambia `OUTPUT_DIR` a donde quieres guardar la salida
   - Cambia `FS_LICENSE` a donde guardaste el `license.txt`
4. Ejecuta el script:
   ```
   scripts\01_run_fastsurfer.bat
   ```

Esto tardara varios minutos por sujeto. Con GPU ~1-2 min cada uno.

---

## PASO 4: Verificar que el preprocesado salio bien

```bash
conda activate alzheimer
python scripts\02_verify_data.py
```

Deberia mostrar algo como:
```
AD:  133 sujetos - OK
MCI: 316 sujetos - OK
CN:  196 sujetos - OK
Total: 645 sujetos listos para entrenar
```

Si algun sujeto tiene error, el script te dira cual y por que.

---

## PASO 5: Entrenar el modelo

1. Abre Anaconda Prompt:
   ```
   conda activate alzheimer
   jupyter notebook
   ```
2. Navega a:
   `Notebooks/ADNI/6 ROIs/Upgrade Filter 6ROI AD vs MCI/Rerun Notebook/`
3. Abre `Upgrade_Filter_6ROI_ADNI_ADvsMCI.ipynb`
4. En la primera celda configurable, cambia la ruta a tus datos procesados
5. Ve a `Kernel → Restart & Run All`
6. Espera a que termine (puede tardar varias horas)
7. Al terminar, se creara el archivo `UpgradedFilter_ROI_ADNI_ADvsMCI.h5` en la misma carpeta

---

## PASO 6: Obtener diagnostico de una imagen nueva

Una vez tienes el `.h5` entrenado, para diagnosticar un paciente nuevo:

1. Primero pasa la imagen del paciente por FastSurfer (igual que en Paso 3, pero solo esa imagen)
2. Luego corre el script de inferencia:

```bash
conda activate alzheimer
python inference\predict.py \
  --aseg  C:\ruta\paciente_nuevo\mri\aparc.DKTatlas+aseg.deep.mgz \
  --orig  C:\ruta\paciente_nuevo\mri\orig.mgz \
  --model UpgradedFilter_ROI_ADNI_ADvsMCI.h5
```

Veras algo como:
```
========================================
RESULTADO DEL DIAGNOSTICO
========================================
AD  (Alzheimer)         : 87.3%
MCI (Deterioro leve)    : 12.7%

Diagnostico mas probable: AD (Alzheimer)
========================================
ADVERTENCIA: Este resultado es solo orientativo.
No reemplaza el criterio de un medico especialista.
========================================
```

---

## Preguntas frecuentes

**¿Por que necesito la licencia de FreeSurfer?**
FastSurfer usa internamente algunas herramientas de FreeSurfer que requieren esa licencia. Es gratuita pero necesitas registrarte.

**¿Cuanto tiempo tarda todo?**
- FastSurfer: ~1-2 min por sujeto con GPU. 200 sujetos = ~4-7 horas.
- Entrenamiento: 3-10 horas dependiendo del numero de sujetos y tu GPU.

**¿Puedo usar el Mac M4 para algo?**
Si, puedes correr el script de inferencia (`predict.py`) en Mac M4, pero necesitas instalar `tensorflow-macos` en lugar de TF normal. Ver instrucciones al final del archivo `inference/predict.py`.

**¿Que pasa si no tengo suficientes imagenes?**
El modelo necesita minimo ~80 por clase para aprender algo. Con menos de 50 por clase los resultados seran muy poco fiables.

**¿El `.nii.gz` y el `.nii` funcionan igual?**
Si, FastSurfer acepta ambos formatos.
