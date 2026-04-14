"""
worker.py

Long-running consumer that listens to a RabbitMQ queue, processes MRI files
through the existing pipeline (FastSurfer preprocessing + 3D ResNet
prediction), and stores results in Supabase / MinIO.

Usage:
    cd src
    python worker.py
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

# Force non-interactive matplotlib backend before any library imports it.
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from the project root (one level above src/)
load_dotenv(os.path.join(SRC_DIR, "..", ".env"))

# Ensure src/ is on sys.path so ``from inference.predict import …`` works
# (mirrors how predict.py itself manipulates sys.path).
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Configuration (all from environment variables)
# ---------------------------------------------------------------------------

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_QUEUE = os.environ.get("RABBITMQ_QUEUE", "mri.processing.jobs")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minio123")
MINIO_INPUT_BUCKET = os.environ.get("MINIO_INPUT_BUCKET", "mri-files")
MINIO_HEATMAP_BUCKET = os.environ.get("MINIO_HEATMAP_BUCKET", "heatmaps")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

FASTSURFER_LICENSE = os.environ.get(
    "FASTSURFER_LICENSE", r"C:\fastsurfer_license\license.txt"
)
FASTSURFER_DOCKER_IMAGE = os.environ.get(
    "FASTSURFER_DOCKER_IMAGE", "deepmi/fastsurfer:latest"
)
FASTSURFER_USE_GPU = os.environ.get("FASTSURFER_USE_GPU", "false").lower() in (
    "1", "true", "yes",
)

MODEL_PATH = os.path.join(
    SRC_DIR,
    os.environ.get("MODEL_PATH", os.path.join("inference", "model_ADNI_CN_vs_rest.h5")),
)
MODEL_CLASSES = os.environ.get("MODEL_CLASSES", "rest,CN").split(",")

# ---------------------------------------------------------------------------
# Service clients (initialised once at module level)
# ---------------------------------------------------------------------------

from services.minio_client import MinIOClient  # noqa: E402
from services.supabase_client import SupabaseClient  # noqa: E402
from services.rabbitmq_consumer import RabbitMQConsumer  # noqa: E402

minio_client = MinIOClient(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
)

supabase_client = SupabaseClient(
    url=SUPABASE_URL,
    anon_key=SUPABASE_ANON_KEY,
)


# ===================================================================
# FastSurfer preprocessing (replicates 01_run_fastsurfer.bat logic
# for a single subject via Docker)
# ===================================================================

def run_fastsurfer(input_nii_path: str, output_dir: str, subject_id: str) -> str:
    """Run FastSurfer segmentation via Docker for a single NIfTI file.

    This mirrors the Docker invocation inside
    ``preprocessing/01_run_fastsurfer.bat``'s ``:process_subject``
    subroutine, adapted for single-file processing.

    Returns the FastSurfer subject directory
    (``<output_dir>/<subject_id>``).
    """
    input_dir = os.path.dirname(os.path.abspath(input_nii_path))
    input_filename = os.path.basename(input_nii_path)

    cmd = ["docker", "run", "--rm", "--user", "root"]
    if FASTSURFER_USE_GPU:
        cmd += ["--gpus", "all"]
    cmd += [
        "-v", f"{input_dir}:/data_in",
        "-v", f"{output_dir}:/data_out",
        "-v", f"{FASTSURFER_LICENSE}:/fs_license.txt",
        FASTSURFER_DOCKER_IMAGE,
        "--t1", f"/data_in/{input_filename}",
        "--sid", subject_id,
        "--sd", "/data_out",
        "--fs_license", "/fs_license.txt",
        "--seg_only",
        "--no_cereb",
        "--allow_root",
    ]

    logger.info("Running FastSurfer: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 h ceiling
    )

    if result.returncode != 0:
        logger.error("FastSurfer stdout:\n%s", result.stdout[-2000:] if result.stdout else "")
        logger.error("FastSurfer stderr:\n%s", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(
            f"FastSurfer exited with code {result.returncode}"
        )

    subject_dir = os.path.join(output_dir, subject_id)
    expected = os.path.join(subject_dir, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    if not os.path.isfile(expected):
        raise FileNotFoundError(
            f"Expected FastSurfer output not found: {expected}"
        )

    logger.info("FastSurfer finished — subject dir: %s", subject_dir)
    return subject_dir


# ===================================================================
# Message handler
# ===================================================================

_JAVA_SERIAL_MAGIC = b"\xac\xed"


def _deserialize_body(body: bytes) -> dict:
    """Decode the message body into a Python dict.

    Supports two formats:
      1. Plain UTF-8 JSON (standard)
      2. Java Object Serialization (Spring AMQP default when using
         ``SimpleMessageConverter`` with a POJO/Map)
    """
    if not body.startswith(_JAVA_SERIAL_MAGIC):
        return json.loads(body)

    import javaobj

    logger.debug("Detected Java-serialized payload (%d bytes)", len(body))
    java_obj = javaobj.loads(body)

    if isinstance(java_obj, str):
        return json.loads(java_obj)

    if isinstance(java_obj, dict):
        return {str(k): v for k, v in java_obj.items()}

    if hasattr(java_obj, "__dict__"):
        raw = {
            k: v for k, v in vars(java_obj).items()
            if not k.startswith("_")
        }
        if raw:
            return raw

    raise ValueError(
        f"Cannot convert Java object of type "
        f"{type(java_obj).__name__} to a message dict"
    )


def _coerce_job_fields(raw: dict) -> tuple[Any, str]:
    """Extract estudio id and MinIO object key from producer-specific shapes.

    Accepts flat JSON/Java maps or one level of nesting (``data`` / ``payload``
    / ``body`` / ``message``). Id aliases: ``id``, ``estudioId``, ``estudio_id``,
    ``studyId``. Path aliases: ``path``, ``filePath``, ``objectKey``, etc.
    """

    def _pick_id(d: dict) -> Any | None:
        for k in ("id", "estudioId", "estudio_id", "studyId", "study_id"):
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _pick_path(d: dict) -> str | None:
        for k in (
            "path",
            "filePath",
            "file_path",
            "objectKey",
            "object_key",
            "minioPath",
            "minio_path",
            "key",
            "s3Key",
            "s3_key",
        ):
            if k in d and d[k] is not None:
                return str(d[k])
        return None

    candidates: list[dict] = [raw]
    for wrap in ("data", "payload", "body", "message"):
        inner = raw.get(wrap)
        if isinstance(inner, dict):
            candidates.append(inner)

    for d in candidates:
        estudio_id = _pick_id(d)
        object_key = _pick_path(d)
        if estudio_id is not None and object_key is not None:
            return estudio_id, object_key

    merged_id: Any | None = None
    merged_path: str | None = None
    for d in candidates:
        if merged_id is None:
            merged_id = _pick_id(d)
        if merged_path is None:
            merged_path = _pick_path(d)
    if merged_id is not None and merged_path is not None:
        return merged_id, merged_path

    keys = sorted({str(k) for c in candidates for k in c.keys()})
    raise KeyError(
        "Job message must include an id field (id / estudioId / estudio_id / …) "
        "and a path field (path / objectKey / object_key / …). "
        f"Top-level keys: {sorted(raw.keys())!r}; union of candidate dicts: {keys!r}"
    )


def process_message(body: bytes) -> None:
    """Process a single MRI job received from the queue.

    Steps:
        1. Download ``.nii`` / ``.nii.gz`` from MinIO
        2. Preprocess with FastSurfer (Docker)
        3. Run 3D ResNet prediction
        4. Generate & upload Grad-CAM heatmap
        5. Upsert results into Supabase
    """
    message = _deserialize_body(body)
    estudio_id, object_key = _coerce_job_fields(message)

    logger.info(
        "▶ START estudio_id=%s  object_key=%s",
        estudio_id,
        object_key,
    )

    temp_dir = tempfile.mkdtemp(prefix="mri_worker_")
    try:
        # ---- 1. Download from MinIO ----
        filename = os.path.basename(object_key)
        local_nii = os.path.join(temp_dir, filename)
        minio_client.download_file(MINIO_INPUT_BUCKET, object_key, local_nii)

        # ---- 2. FastSurfer preprocessing ----
        fastsurfer_out = os.path.join(temp_dir, "fastsurfer_output")
        os.makedirs(fastsurfer_out)
        subject_dir = run_fastsurfer(local_nii, fastsurfer_out, str(estudio_id))

        mri_dir = os.path.join(subject_dir, "mri")
        aseg_path = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")
        orig_path = os.path.join(mri_dir, "orig.mgz")

        # ---- 3. Prediction ----
        from inference.predict import predict_binary

        try:
            result = predict_binary(
                aseg_path,
                orig_path,
                MODEL_PATH,
                class_names=list(MODEL_CLASSES),
            )
        except SystemExit as exc:
            raise RuntimeError(
                f"predict_binary terminated (exit code {exc.code})"
            ) from exc

        prediction_payload = {
            "predicted_class": int(result["predicted_class"]),
            "predicted_name": str(result["predicted_name"]),
            "probabilities": [float(p) for p in result["probabilities"]],
            "class_names": list(result["class_names"]),
            "mode": str(result["mode"]),
        }
        logger.info(
            "Prediction: %s (probs=%s)",
            prediction_payload["predicted_name"],
            prediction_payload["probabilities"],
        )

        # ---- 4. Heatmap (best-effort) ----
        heatmap_db_path: str | None = None
        try:
            from explainability.gradcam import run_gradcam_on_subject

            heatmap_local = os.path.join(temp_dir, f"{estudio_id}_heatmap.png")
            try:
                run_gradcam_on_subject(
                    MODEL_PATH,
                    aseg_path,
                    orig_path,
                    class_names=result["class_names"],
                    save_path=heatmap_local,
                )
            except SystemExit:
                raise RuntimeError("GradCAM generation terminated unexpectedly")

            if os.path.isfile(heatmap_local):
                heatmap_object_key = f"{estudio_id}_heatmap.png"
                minio_client.upload_file(
                    MINIO_HEATMAP_BUCKET,
                    heatmap_object_key,
                    heatmap_local,
                    content_type="image/png",
                )
                heatmap_db_path = f"heatmaps/{heatmap_object_key}"
                logger.info("Heatmap uploaded: %s", heatmap_db_path)
        except Exception:
            logger.warning("Heatmap generation/upload failed (non-fatal)", exc_info=True)

        # ---- 5. Persist to Supabase ----
        supabase_client.upsert_resultado(
            estudio_id, prediction_payload, heatmap_db_path
        )

        logger.info("✔ DONE estudio_id=%s", estudio_id)

    except Exception:
        logger.error("✘ FAILED estudio_id=%s", estudio_id, exc_info=True)
        try:
            supabase_client.update_resultado_error(
                estudio_id,
                "Processing failed — check worker logs for details.",
            )
        except Exception:
            pass
        raise  # propagate so the consumer NACKs the message

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug("Cleaned up %s", temp_dir)


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("  MRI Processing Worker")
    logger.info("=" * 60)
    logger.info("RabbitMQ : %s:%s  queue=%s", RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_QUEUE)
    logger.info("MinIO    : %s  input=%s  heatmaps=%s", MINIO_ENDPOINT, MINIO_INPUT_BUCKET, MINIO_HEATMAP_BUCKET)
    logger.info("Supabase : %s", SUPABASE_URL)
    logger.info("Model    : %s  classes=%s", MODEL_PATH, MODEL_CLASSES)
    logger.info("GPU      : %s", "enabled" if FASTSURFER_USE_GPU else "disabled (CPU only)")
    logger.info("=" * 60)

    minio_client.ensure_bucket(MINIO_HEATMAP_BUCKET)

    consumer = RabbitMQConsumer(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        user=RABBITMQ_USER,
        password=RABBITMQ_PASS,
        queue=RABBITMQ_QUEUE,
        on_message=process_message,
    )
    consumer.start()


if __name__ == "__main__":
    main()
