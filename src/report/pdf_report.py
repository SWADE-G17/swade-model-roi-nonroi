"""
report/pdf_report.py

Construye un reporte PDF con el resumen del procesamiento de un estudio MRI:

    - Identificacion del estudio y fecha de procesamiento.
    - Clase predicha y probabilidades por clase.
    - 5 cortes axiales del volumen MRI original (orig.mgz).
    - 5 cortes axiales con el heatmap Grad-CAM superpuesto sobre el volumen.

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
        num_slices=5,
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


def _select_slice_indices(depth: int, num_slices: int) -> np.ndarray:
    # Cortes equiespaciados en el cuarto central, donde se ve mejor el cerebro.
    lo = max(depth // 4, 0)
    hi = min(3 * depth // 4, depth - 1)
    if hi <= lo:
        return np.linspace(0, max(depth - 1, 0), num_slices, dtype=int)
    return np.linspace(lo, hi, num_slices, dtype=int)


# ---------------------------------------------------------------------------
# Construccion de figuras matplotlib -> PNG bytes
# ---------------------------------------------------------------------------

def _build_slice_figure(
    volume: np.ndarray,
    *,
    overlay: np.ndarray | None = None,
    num_slices: int = 5,
    title: str = "",
    cmap: str = "gray",
    overlay_cmap: str = "jet",
    overlay_alpha: float = 0.45,
) -> bytes:
    """Devuelve un PNG (bytes) con `num_slices` cortes axiales del volumen.

    Si se pasa `overlay`, este se superpone con transparencia. Tanto volume
    como overlay deben tener la misma forma 3D (D, H, W).
    """
    if volume.ndim != 3:
        raise ValueError(f"volume debe ser 3D, shape={volume.shape}")

    depth = volume.shape[2]
    indices = _select_slice_indices(depth, num_slices)

    fig, axes = plt.subplots(1, num_slices, figsize=(num_slices * 2.6, 2.9))
    if num_slices == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        slc = np.rot90(volume[:, :, int(idx)])
        slc_norm = _normalize_unit(slc)
        ax.imshow(slc_norm, cmap=cmap)

        if overlay is not None:
            heat = np.rot90(overlay[:, :, int(idx)])
            ax.imshow(
                heat,
                cmap=overlay_cmap,
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=1.0,
            )

        ax.set_title(f"Corte {int(idx)}", fontsize=9)
        ax.axis("off")

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
    num_slices: int = 5,
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
            el PDF incluye solo los cortes anatomicos.
        output_pdf_path: ruta donde guardar el PDF.
        num_slices: numero de cortes axiales a embeber (default 5).
        processing_date: marca temporal; si es None se usa ``datetime.now()``.
    """
    processing_date = processing_date or datetime.now()

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

    # ---- Render de las dos figuras de cortes -------------------------------
    volume_png = _build_slice_figure(
        orig_volume,
        num_slices=num_slices,
        title="Volumen MRI - Cortes axiales",
    )

    gradcam_png: bytes | None = None
    if heatmap_volume is not None:
        gradcam_png = _build_slice_figure(
            orig_volume,
            overlay=heatmap_volume,
            num_slices=num_slices,
            title="Mapa de calor (Grad-CAM) sobre el volumen",
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

    # ---- Visualizaciones --------------------------------------------------
    story.append(Paragraph("Visualizacion del volumen", h2))
    story.append(
        Paragraph(
            f"Se muestran {num_slices} cortes axiales del volumen T1 "
            "reconstruido (orig.mgz).",
            body,
        )
    )
    story.append(Spacer(1, 4))
    story.append(_image_from_png_bytes(volume_png, max_width=17 * cm))

    story.append(Paragraph("Mapa de calor (Grad-CAM)", h2))
    if gradcam_png is not None:
        story.append(
            Paragraph(
                f"Los mismos {num_slices} cortes con la activacion del modelo "
                "superpuesta. Las regiones mas calidas (rojo) indican mayor "
                "contribucion a la decision de la clase predicha.",
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
