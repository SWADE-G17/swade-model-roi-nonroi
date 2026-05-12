"""
services/supabase_client.py

HTTP client for the Supabase PostgREST API.
Used to upsert prediction results into the ``resultado`` table.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0


class SupabaseClient:
    """Lightweight client that talks directly to Supabase PostgREST."""

    def __init__(self, url: str, anon_key: str):
        self._base_url = url.rstrip("/")
        self._headers = {
            "apikey": anon_key,
            "Authorization": f"Bearer {anon_key}",
            "Content-Type": "application/json",
        }

    def upsert_resultado(
        self,
        estudio_id: Any,
        prediction: dict,
        heatmap_path: str | None = None,
        orig_path: str | None = None,
        report_path: str | None = None,
    ) -> None:
        """Insert or upsert a row in the ``resultado`` table.

        Uses PostgREST's ``resolution=merge-duplicates`` header so that
        a second call for the same *estudio_id* updates the existing row
        instead of raising a conflict (requires a UNIQUE constraint on
        ``estudio_id`` in the database).
        """
        url = f"{self._base_url}/rest/v1/resultado"

        payload: dict[str, Any] = {
            "estudio_id": estudio_id,
            "prediction": prediction,
        }
        if heatmap_path is not None:
            payload["heatmap_path"] = heatmap_path
        if orig_path is not None:
            payload["orig_path"] = orig_path
        if report_path is not None:
            payload["report_path"] = report_path

        headers = {
            **self._headers,
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }

        response = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
        response.raise_for_status()
        logger.info("Upserted resultado for estudio_id=%s", estudio_id)

    def update_estudio_status(self, estudio_id: Any, status: str) -> None:
        """PATCH a row in ``public.estudio`` by primary key ``id``.

        ``status`` must satisfy ``chk_estudio_status`` (e.g. ``procesando``,
        ``listo``, ``error``).
        """
        url = f"{self._base_url}/rest/v1/estudio?id=eq.{estudio_id}"
        headers = {**self._headers, "Prefer": "return=minimal"}
        response = httpx.patch(
            url,
            json={"status": status},
            headers=headers,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        logger.info("Updated estudio id=%s status=%s", estudio_id, status)

    def update_estudio_error(
        self,
        estudio_id: Any,
        error_message: str,
    ) -> None:
        """Mark an ``estudio`` row as failed.

        Sets ``status = 'error'`` and stores the human-readable cause in
        ``error_message``. Uses ``Prefer: count=exact`` so we can detect
        (and loudly warn) when the row id does not exist instead of
        silently returning 204 with zero rows affected.
        """
        url = f"{self._base_url}/rest/v1/estudio?id=eq.{estudio_id}"
        payload = {
            "status": "error",
            "error_message": error_message,
        }
        headers = {
            **self._headers,
            "Prefer": "return=minimal,count=exact",
        }

        try:
            response = httpx.patch(url, json=payload, headers=headers, timeout=_TIMEOUT)
            response.raise_for_status()
        except Exception:
            logger.warning(
                "Could not record error state for estudio id=%s",
                estudio_id,
                exc_info=True,
            )
            return

        affected = _affected_rows(response.headers.get("content-range"))
        if affected == 0:
            logger.warning(
                "PATCH estudio id=%s affected 0 rows (row not found?)",
                estudio_id,
            )
            return

        logger.info(
            "Recorded error state for estudio id=%s (rows=%s)",
            estudio_id,
            "?" if affected is None else affected,
        )


def _affected_rows(content_range: str | None) -> int | None:
    """Parse PostgREST's ``Content-Range`` header (e.g. ``0-0/1`` or ``*/0``).

    Returns the total rows affected, or ``None`` if the header is missing
    or unparseable.
    """
    if not content_range or "/" not in content_range:
        return None
    try:
        return int(content_range.rsplit("/", 1)[1])
    except ValueError:
        return None
