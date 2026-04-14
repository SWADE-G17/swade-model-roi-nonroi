"""
preprocessing/03_convert_dicom.py

Convierte imagenes DICOM (.dcm) de ADNI a formato NIfTI (.nii.gz)
usando dcm2niix, y las organiza en la estructura esperada por FastSurfer.

REQUISITOS:
    - dcm2niix instalado y en el PATH (o especificar la ruta abajo)
      Descarga: https://github.com/rordenlab/dcm2niix/releases
    - Python 3.8+ con os, subprocess, shutil

USO:
    python preprocessing/03_convert_dicom.py

ESTRUCTURA DE ENTRADA ESPERADA:
    INPUT_DIR/
        AD/
            subcarpeta_paciente1/
                subcarpeta_sesion/
                    archivo001.dcm
                    archivo002.dcm
                    ...
            subcarpeta_paciente2/
                ...
        MCI/
            ...
        CN/
            ...

ESTRUCTURA DE SALIDA:
    OUTPUT_DIR/
        AD/
            paciente1.nii.gz
            paciente2.nii.gz
        MCI/
            ...
        CN/
            ...
"""

import os
import subprocess
import shutil
import glob

# ============================================================
# CONFIGURACION - EDITA ESTAS VARIABLES
# ============================================================

# Carpeta raiz con las imagenes DICOM organizadas en AD/, MCI/, CN/
INPUT_DIR = r"C:\Users\Harry\Documents\GitHub\Images\ADNI\Raw"

# Carpeta donde se guardaran los .nii.gz convertidos
OUTPUT_DIR = r"C:\Users\Harry\Documents\GitHub\Images\ADNI\Nifti"

# Ruta al ejecutable dcm2niix
# Si lo agregaste al PATH, solo pon "dcm2niix"
# Si no, pon la ruta completa, ej: r"C:\dcm2niix\dcm2niix.exe"
DCM2NIIX_PATH = r"C:\dcm2niix\dcm2niix.exe"

# Clases a procesar
CLASSES = ["AD", "MCI", "CN"]

# ============================================================
# NO MODIFICAR LO DE ABAJO
# ============================================================


def find_dicom_series(patient_dir):
    """
    Busca recursivamente la carpeta que contiene los archivos .dcm
    dentro de la carpeta de un paciente ADNI.

    ADNI tipicamente tiene esta estructura:
    paciente/ -> sesion/ -> serie/ -> *.dcm

    Returns:
        lista de carpetas que contienen .dcm directamente
    """
    dicom_folders = set()
    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if dcm_files:
            dicom_folders.add(root)
    return list(dicom_folders)


def convert_patient(patient_dir, output_dir, patient_id):
    """
    Convierte todos los .dcm de un paciente a .nii.gz.

    Args:
        patient_dir: carpeta raiz del paciente con los .dcm
        output_dir: carpeta donde guardar el .nii.gz
        patient_id: nombre/ID del paciente (se usa como nombre del archivo)

    Returns:
        True si la conversion fue exitosa, False si hubo error
    """
    dicom_series = find_dicom_series(patient_dir)

    if not dicom_series:
        print(f"  [SKIP] No se encontraron .dcm en: {patient_dir}")
        return False

    # Usar la primera serie encontrada (para ADNI tipicamente hay una sola)
    dicom_folder = dicom_series[0]

    # Carpeta temporal para la conversion
    temp_dir = os.path.join(output_dir, "_temp_" + patient_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        result = subprocess.run(
            [
                DCM2NIIX_PATH,
                "-z", "y",           # comprimir a .gz
                "-f", patient_id,    # nombre del archivo de salida
                "-o", temp_dir,      # carpeta de salida
                dicom_folder,        # carpeta con los .dcm
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"  [ERROR] dcm2niix fallo para {patient_id}:")
            print(f"    {result.stderr[:200]}")
            return False

        # Buscar el .nii.gz generado y moverlo a output_dir
        nii_files = glob.glob(os.path.join(temp_dir, "*.nii.gz"))
        if not nii_files:
            nii_files = glob.glob(os.path.join(temp_dir, "*.nii"))

        if not nii_files:
            print(f"  [ERROR] No se genero ningun .nii.gz para {patient_id}")
            return False

        # Si hay varios (varias series), usar el mas grande (tipicamente el T1)
        nii_file = max(nii_files, key=os.path.getsize)
        ext = ".nii.gz" if nii_file.endswith(".nii.gz") else ".nii"
        dest = os.path.join(output_dir, patient_id + ext)
        shutil.move(nii_file, dest)
        return True

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout para {patient_id}")
        return False
    except FileNotFoundError:
        print(f"\nERROR: No se encontro dcm2niix en: {DCM2NIIX_PATH}")
        print("Descargalo de: https://github.com/rordenlab/dcm2niix/releases")
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print()
    print("=" * 60)
    print("  CONVERSION DICOM -> NIfTI")
    print("=" * 60)
    print(f"  Entrada:   {INPUT_DIR}")
    print(f"  Salida:    {OUTPUT_DIR}")
    print(f"  dcm2niix:  {DCM2NIIX_PATH}")
    print("=" * 60)

    # Verificar que dcm2niix existe
    try:
        subprocess.run([DCM2NIIX_PATH, "--version"],
                       capture_output=True, timeout=10)
    except FileNotFoundError:
        print(f"\nERROR: No se encontro dcm2niix en: {DCM2NIIX_PATH}")
        print("Descargalo de: https://github.com/rordenlab/dcm2niix/releases")
        return

    total_ok = 0
    total_errors = 0

    for class_name in CLASSES:
        class_input = os.path.join(INPUT_DIR, class_name)
        class_output = os.path.join(OUTPUT_DIR, class_name)

        if not os.path.exists(class_input):
            print(f"\nADVERTENCIA: No existe {class_input} - saltando")
            continue

        os.makedirs(class_output, exist_ok=True)
        print(f"\n--- Procesando clase: {class_name} ---")

        patients = [
            d for d in sorted(os.listdir(class_input))
            if os.path.isdir(os.path.join(class_input, d))
        ]

        for patient_id in patients:
            # Saltar si ya fue convertido
            dest_gz = os.path.join(class_output, patient_id + ".nii.gz")
            dest_nii = os.path.join(class_output, patient_id + ".nii")
            if os.path.exists(dest_gz) or os.path.exists(dest_nii):
                print(f"  [SKIP] {patient_id} ya convertido")
                total_ok += 1
                continue

            print(f"  [INICIO] {patient_id}")
            patient_dir = os.path.join(class_input, patient_id)
            ok = convert_patient(patient_dir, class_output, patient_id)

            if ok:
                print(f"  [OK] {patient_id}")
                total_ok += 1
            else:
                total_errors += 1

    print()
    print("=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Convertidos exitosamente: {total_ok}")
    print(f"  Errores:                  {total_errors}")
    print("=" * 60)
    print()
    print("Siguiente paso: corre 01_run_fastsurfer.bat")
    print()


if __name__ == "__main__":
    main()
