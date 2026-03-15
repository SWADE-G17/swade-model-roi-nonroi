"""
02_verify_data.py

DESCRIPCION:
    Verifica que el preprocesado con FastSurfer fue exitoso para todos los sujetos.
    Comprueba que cada sujeto en AD/, MCI/ y CN/ tiene los dos archivos necesarios
    para entrenar el modelo:
      - mri/orig.mgz
      - mri/aparc.DKTatlas+aseg.deep.mgz

USO:
    python scripts/02_verify_data.py

    Opcional: especificar la carpeta de datos procesados:
    python scripts/02_verify_data.py --processed_dir C:/alzheimer_data/processed
"""

import os
import argparse

# ============================================================
# CONFIGURACION - cambia esta ruta si es necesario
# ============================================================
DEFAULT_PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed"
)

CLASES = ["AD", "MCI", "CN"]
ARCHIVOS_REQUERIDOS = [
    os.path.join("mri", "orig.mgz"),
    os.path.join("mri", "aparc.DKTatlas+aseg.deep.mgz"),
]
MINIMO_SUJETOS = 50


def verificar_sujeto(ruta_sujeto, nombre_sujeto):
    """Verifica que un sujeto tiene todos los archivos requeridos."""
    faltantes = []
    for archivo in ARCHIVOS_REQUERIDOS:
        ruta_archivo = os.path.join(ruta_sujeto, archivo)
        if not os.path.exists(ruta_archivo):
            faltantes.append(archivo)
    return faltantes


def verificar_clase(ruta_clase, nombre_clase):
    """Verifica todos los sujetos de una clase."""
    if not os.path.exists(ruta_clase):
        return None, None, None

    sujetos = [
        d for d in os.listdir(ruta_clase)
        if os.path.isdir(os.path.join(ruta_clase, d))
        and not d.startswith(".")
    ]

    ok = []
    con_errores = []

    for sujeto in sorted(sujetos):
        ruta_sujeto = os.path.join(ruta_clase, sujeto)
        faltantes = verificar_sujeto(ruta_sujeto, sujeto)
        if faltantes:
            con_errores.append((sujeto, faltantes))
        else:
            ok.append(sujeto)

    return ok, con_errores, sujetos


def main():
    parser = argparse.ArgumentParser(
        description="Verifica que los datos procesados por FastSurfer esten correctos"
    )
    parser.add_argument(
        "--processed_dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Carpeta con los datos procesados (default: {DEFAULT_PROCESSED_DIR})"
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir

    print()
    print("=" * 60)
    print("  VERIFICACION DE DATOS PROCESADOS")
    print("=" * 60)
    print(f"  Carpeta: {processed_dir}")
    print("=" * 60)

    if not os.path.exists(processed_dir):
        print()
        print(f"ERROR: No existe la carpeta: {processed_dir}")
        print()
        print("Asegurate de haber corrido primero: scripts/01_run_fastsurfer.bat")
        print("O edita la variable DEFAULT_PROCESSED_DIR en este script.")
        return

    resumen = {}
    hay_errores = False

    for clase in CLASES:
        ruta_clase = os.path.join(processed_dir, clase)
        ok, con_errores, todos = verificar_clase(ruta_clase, clase)

        if todos is None:
            print(f"\n  {clase:5s}: carpeta no encontrada - {ruta_clase}")
            print(f"         Crea la carpeta y pon ahi los sujetos procesados.")
            resumen[clase] = {"ok": 0, "errores": 0, "total": 0}
            continue

        resumen[clase] = {
            "ok": len(ok),
            "errores": len(con_errores),
            "total": len(todos),
        }

        estado = "OK" if len(con_errores) == 0 else "CON ERRORES"
        alerta_minimo = " ⚠ POCOS SUJETOS" if len(ok) < MINIMO_SUJETOS else ""

        print()
        print(f"  {clase}:")
        print(f"    Total sujetos:     {len(todos)}")
        print(f"    Correctos:         {len(ok)}")
        print(f"    Con errores:       {len(con_errores)}")
        print(f"    Estado:            {estado}{alerta_minimo}")

        if con_errores:
            hay_errores = True
            print(f"    Sujetos con errores:")
            for sujeto, faltantes in con_errores[:10]:
                print(f"      - {sujeto}")
                for f in faltantes:
                    print(f"          Falta: {f}")
            if len(con_errores) > 10:
                print(f"      ... y {len(con_errores) - 10} mas")

    # Resumen final
    total_ok = sum(v["ok"] for v in resumen.values())
    total_errores = sum(v["errores"] for v in resumen.values())
    total_todos = sum(v["total"] for v in resumen.values())

    print()
    print("=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Total sujetos encontrados:  {total_todos}")
    print(f"  Sujetos listos para usar:   {total_ok}")
    print(f"  Sujetos con errores:        {total_errores}")
    print("=" * 60)

    if not hay_errores and total_ok > 0:
        print()
        print("  TODO CORRECTO. Puedes abrir el notebook y entrenar el modelo.")
        print()
        print("  Siguiente paso:")
        print("  1. Abre Jupyter Notebook")
        print("  2. Abre: Notebooks/ADNI/6 ROIs/Upgrade Filter 6ROI AD vs MCI/")
        print("           Rerun Notebook/Upgrade_Filter_6ROI_ADNI_ADvsMCI.ipynb")
        print("  3. Cambia la variable BASE_DIR en la primera celda configurable")
        print(f"     a: {processed_dir}")
        print("  4. Corre todas las celdas (Kernel → Restart & Run All)")
    elif total_ok == 0:
        print()
        print("  No hay datos procesados todavia.")
        print("  Corre primero: scripts/01_run_fastsurfer.bat")
    else:
        print()
        print("  Hay errores en algunos sujetos.")
        print("  Opciones:")
        print("  1. Volver a correr FastSurfer para los sujetos con errores")
        print("  2. Ignorarlos si hay suficientes sujetos correctos por clase")
        print("     (minimo recomendado: 80 sujetos por clase)")

    print()

    # Verificar balance de clases
    ad_ok = resumen.get("AD", {}).get("ok", 0)
    mci_ok = resumen.get("MCI", {}).get("ok", 0)
    cn_ok = resumen.get("CN", {}).get("ok", 0)

    if ad_ok > 0 and cn_ok > 0:
        print("  BALANCE DE CLASES (AD vs CN):")
        ratio = max(ad_ok, cn_ok) / max(min(ad_ok, cn_ok), 1)
        print(f"    AD:  {ad_ok} sujetos")
        print(f"    CN:  {cn_ok} sujetos")
        if ratio > 3:
            print(f"    ADVERTENCIA: clases muy desbalanceadas (ratio {ratio:.1f}:1)")
            print(f"    El codigo las balancea automaticamente por oversampling.")
        else:
            print(f"    Balance aceptable (ratio {ratio:.1f}:1)")
    print()


if __name__ == "__main__":
    main()
