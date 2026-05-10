"""
report/pdf_report.py

Construye un reporte PDF con el resumen del procesamiento de un estudio MRI:

    - Identificacion del estudio y fecha de procesamiento.
    - Clase predicha y probabilidades por clase.
    - 6 cortes sagitales (3 por hemisferio) con el heatmap Grad-CAM
      superpuesto sobre el volumen T1 (orig.mgz). Se omite el corte
      interhemisferico (linea media) y los cortes muy laterales (borde).

USO:
    from report.pdf_report import generate_pdf_report

    generate_pdf_report(
        estudio_id="123",
        prediction={
            "predicted_name": "AD",
            "probabilities": [0.87, 0.13],
            "class_names": ["AD", "MCI"],
            "mode": "binary",
        },
        orig_path="/path/to/orig.mgz",
        heatmap_volume_path="/path/to/heatmap.nii.gz",   # opcional
        output_pdf_path="/path/to/report.pdf",
        num_slices_per_side=3,
    )
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Backend no-interactivo (worker headless)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers para volumenes
# ---------------------------------------------------------------------------

def _load_volume_3d(path: str) -> np.ndarray:
    """Carga un volumen NIfTI/MGZ y lo devuelve como array 3D float32."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Volumen esperado 3D, recibido shape={data.shape} ({path})"
        )
    return data.astype(np.float32, copy=False)


def _resize_to(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    if volume.shape == target_shape:
        return volume
    from skimage.transform import resize as _resize
    return np.asarray(
        _resize(volume, target_shape, anti_aliasing=True),
        dtype=np.float32,
    )


def _normalize_unit(arr: np.ndarray) -> np.ndarray:
    amin = float(arr.min())
    amax = float(arr.max())
    if amax > amin:
        return (arr - amin) / (amax - amin)
    return np.zeros_like(arr, dtype=np.float32)


def _select_sagittal_indices(
    width: int,
    num_per_side: int = 3,
    *,
    inner_margin_frac: float = 0.06,
    outer_margin_frac: float = 0.30,
) -> list[int]:
    """Devuelve ``2 * num_per_side`` indices de cortes sagitales.

    Mitad de los cortes quedan a un lado de la linea media y la otra mitad
    al lado opuesto. Se evita la zona interhemisferica (linea media) y los
    cortes muy cercanos al borde lateral del volumen para no caer fuera del
    cerebro.

    Args:
        width: tamano del eje sagital (eje 0 del volumen).
        num_per_side: cantidad de cortes por hemisferio (default 3).
        inner_margin_frac: fraccion del ancho a saltar a cada lado de la
            linea media para evitar el corte interhemisferico.
        outer_margin_frac: fraccion del ancho a saltar desde cada borde
            lateral del volumen para no caer fuera del cerebro.
    """
    if width < 4 or num_per_side < 1:
        return list(range(min(width, max(num_per_side * 2, 1))))

    center = width // 2
    inner_margin = max(int(round(width * inner_margin_frac)), 1)
    outer_margin = max(int(round(width * outer_margin_frac)), 1)

    left_lo = outer_margin
    left_hi = max(center - inner_margin, left_lo + 1)
    right_hi = max(width - 1 - outer_margin, left_hi + 1)
    right_lo = min(center + inner_margin, right_hi - 1)

    left_indices = np.linspace(left_lo, left_hi, num_per_side, dtype=int)
    right_indices = np.linspace(right_lo, right_hi, num_per_side, dtype=int)
    return [int(i) for i in left_indices] + [int(i) for i in right_indices]


# ---------------------------------------------------------------------------
# Construccion de figuras matplotlib -> PNG bytes
# ---------------------------------------------------------------------------

def _build_sagittal_grid_figure(
    volume: np.ndarray,
    overlay: np.ndarray,
    *,
    num_per_side: int = 3,
    title: str = "",
    cmap: str = "gray",
    overlay_cmap: str = "jet",
    overlay_alpha: float = 0.45,
) -> bytes:
    """PNG (bytes) con una rejilla 2x``num_per_side`` de cortes sagitales.

    La fila superior corresponde a un hemisferio y la inferior al opuesto.
    El ``overlay`` (heatmap Grad-CAM) se superpone con transparencia sobre
    todos los cortes. Tanto ``volume`` como ``overlay`` deben tener la
    misma forma 3D ``(D, H, W)`` donde ``D`` es el eje sagital.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume debe ser 3D, shape={volume.shape}")
    if overlay.shape != volume.shape:
        raise ValueError(
            f"overlay debe tener la misma forma que volume "
            f"(volume={volume.shape}, overlay={overlay.shape})"
        )

    sagittal_axis = volume.shape[0]
    indices = _select_sagittal_indices(sagittal_axis, num_per_side=num_per_side)
    left_indices = indices[:num_per_side]
    right_indices = indices[num_per_side:]

    fig, axes = plt.subplots(
        2,
        num_per_side,
        figsize=(num_per_side * 2.6, 2 * 2.9),
    )
    if num_per_side == 1:
        axes = np.array(axes).reshape(2, 1)

    side_labels = ("Hemisferio der.", "Hemisferio izq.")
    for row, (label, row_indices) in enumerate(
        zip(side_labels, (left_indices, right_indices))
    ):
        for col, idx in enumerate(row_indices):
            ax = axes[row, col]
            # orig.mgz (LIA): eje 1 va Superior->Inferior y eje 2 va
            # Posterior->Anterior, por lo que sin rotacion imshow ya
            # muestra craneo arriba y cara a la derecha.
            slc = volume[int(idx), :, :]
            slc_norm = _normalize_unit(slc)
            ax.imshow(slc_norm, cmap=cmap)

            heat = overlay[int(idx), :, :]
            ax.imshow(
                heat,
                cmap=overlay_cmap,
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_title(f"Corte sagital {int(idx)}", fontsize=9)
            ax.axis("off")

            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#374151",
                )

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _image_from_png_bytes(png_bytes: bytes, max_width: float) -> RLImage:
    """Convierte PNG en bytes a una imagen ReportLab ajustada al ancho dado."""
    img = RLImage(io.BytesIO(png_bytes))
    iw, ih = float(img.imageWidth), float(img.imageHeight)
    if iw <= 0:
        return img
    scale = max_width / iw
    img.drawWidth = max_width
    img.drawHeight = ih * scale
    return img


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def generate_pdf_report(
    estudio_id: Any,
    prediction: dict,
    orig_path: str,
    heatmap_volume_path: str | None,
    output_pdf_path: str,
    num_slices_per_side: int = 3,
    processing_date: datetime | None = None,
) -> str:
    """Genera el PDF de resultados y devuelve su ruta absoluta.

    Args:
        estudio_id: identificador del estudio (cualquier tipo serializable a str).
        prediction: dict con al menos ``predicted_name``, ``probabilities``
            y ``class_names``. Opcionalmente ``mode``, ``predicted_class``
            y ``raw_ovr``.
        orig_path: ruta al volumen ``orig.mgz`` (T1 reconstruido).
        heatmap_volume_path: ruta al volumen Grad-CAM (NIfTI/MGZ) ya
            registrado en la rejilla de orig.mgz. Si es None o no existe,
            el PDF incluye solo los cortes anatomicos sin heatmap.
        output_pdf_path: ruta donde guardar el PDF.
        num_slices_per_side: numero de cortes sagitales por hemisferio
            (default 3, total 6).
        processing_date: marca temporal; si es None se usa ``datetime.now()``.
    """
    processing_date = processing_date or datetime.now()
    num_slices_per_side = max(int(num_slices_per_side), 1)

    # ---- Cargar volumenes ---------------------------------------------------
    orig_volume = _load_volume_3d(orig_path)

    heatmap_volume: np.ndarray | None = None
    if heatmap_volume_path and os.path.isfile(heatmap_volume_path):
        try:
            heatmap_volume = _load_volume_3d(heatmap_volume_path)
            heatmap_volume = _resize_to(heatmap_volume, orig_volume.shape)
            heatmap_volume = _normalize_unit(heatmap_volume)
        except Exception:
            logger.warning(
                "No se pudo cargar el volumen de heatmap '%s'; el reporte se "
                "generara sin overlay.",
                heatmap_volume_path,
                exc_info=True,
            )
            heatmap_volume = None

    # ---- Render de la figura de cortes -------------------------------------
    gradcam_png: bytes | None = None
    if heatmap_volume is not None:
        gradcam_png = _build_sagittal_grid_figure(
            orig_volume,
            heatmap_volume,
            num_per_side=num_slices_per_side,
            title="Cortes sagitales con mapa de calor (Grad-CAM)",
        )

    # ---- Construccion del PDF ---------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=f"Reporte estudio {estudio_id}",
        author="MRI Processing Worker",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=1,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#444444"),
        alignment=1,
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#777777"),
        alignment=1,
    )

    story: list = []
    story.append(Paragraph("Reporte de diagnostico MRI", title_style))
    story.append(
        Paragraph(
            "Analisis automatizado con 3D ResNet + Grad-CAM",
            subtitle_style,
        )
    )

    # ---- Metadatos del estudio --------------------------------------------
    predicted_name = str(prediction.get("predicted_name", "—"))
    class_names = list(prediction.get("class_names", []))
    probabilities = [float(p) for p in prediction.get("probabilities", [])]
    mode = str(prediction.get("mode", "binary"))

    pred_idx: int = 0
    if "predicted_class" in prediction:
        try:
            pred_idx = int(prediction["predicted_class"])
        except (TypeError, ValueError):
            pred_idx = 0
    elif class_names and predicted_name in class_names:
        pred_idx = class_names.index(predicted_name)

    pred_prob_str = "—"
    if 0 <= pred_idx < len(probabilities):
        pred_prob_str = f"{probabilities[pred_idx] * 100.0:.1f}%"

    metadata_rows = [
        ["Identificacion del estudio", str(estudio_id)],
        ["Fecha de procesamiento", processing_date.strftime("%Y-%m-%d %H:%M:%S")],
        ["Modo del modelo", mode],
        ["Clase predicha", predicted_name],
        ["Probabilidad asociada", pred_prob_str],
    ]

    meta_table = Table(metadata_rows, colWidths=[5.5 * cm, 11 * cm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1F2937")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # ---- Tabla de probabilidades ------------------------------------------
    if class_names and probabilities:
        prob_rows: list[list[str]] = [["Clase", "Probabilidad"]]
        for i, name in enumerate(class_names):
            p = probabilities[i] * 100.0 if i < len(probabilities) else 0.0
            marker = "  <-- Predicha" if i == pred_idx else ""
            prob_rows.append([f"{name}{marker}", f"{p:.1f}%"])

        prob_table = Table(prob_rows, colWidths=[8.0 * cm, 8.5 * cm])
        prob_style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
        ]
        if 0 <= pred_idx < len(class_names):
            prob_style_cmds.append(
                (
                    "BACKGROUND",
                    (0, pred_idx + 1),
                    (-1, pred_idx + 1),
                    colors.HexColor("#FEF3C7"),
                )
            )
        prob_table.setStyle(TableStyle(prob_style_cmds))

        story.append(Paragraph("Probabilidades por clase", h2))
        story.append(prob_table)

    # ---- Visualizacion: cortes sagitales con Grad-CAM ---------------------
    total_slices = num_slices_per_side * 2
    story.append(Paragraph("Mapa de calor (Grad-CAM)", h2))
    if gradcam_png is not None:
        story.append(
            Paragraph(
                f"Se muestran {total_slices} cortes sagitales del volumen T1 "
                f"reconstruido (orig.mgz): {num_slices_per_side} por hemisferio, "
                "evitando la linea media y los bordes laterales. Sobre cada "
                "corte se superpone la activacion del modelo; las regiones "
                "mas calidas (rojo) indican mayor contribucion a la decision "
                "de la clase predicha.",
                body,
            )
        )
        story.append(Spacer(1, 4))
        story.append(_image_from_png_bytes(gradcam_png, max_width=17 * cm))
    else:
        story.append(
            Paragraph(
                "No se pudo generar el mapa Grad-CAM para este estudio.",
                body,
            )
        )

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Resultado orientativo. No reemplaza el criterio medico especializado.",
            note_style,
        )
    )

    doc.build(story)
    logger.info("Reporte PDF generado en %s", output_pdf_path)
    return output_pdf_path
