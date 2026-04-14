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

        headers = {
            **self._headers,
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }

        response = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
        response.raise_for_status()
        logger.info("Upserted resultado for estudio_id=%s", estudio_id)

    def update_resultado_error(
        self,
        estudio_id: Any,
        error_message: str,
    ) -> None:
        """Optionally mark a ``resultado`` row with an error status."""
        url = (
            f"{self._base_url}/rest/v1/resultado"
            f"?estudio_id=eq.{estudio_id}"
        )
        payload = {
            "prediction": {"error": error_message},
        }
        headers = {**self._headers, "Prefer": "return=minimal"}

        try:
            response = httpx.patch(url, json=payload, headers=headers, timeout=_TIMEOUT)
            response.raise_for_status()
            logger.info("Recorded error state for estudio_id=%s", estudio_id)
        except Exception:
            logger.warning(
                "Could not record error state for estudio_id=%s",
                estudio_id,
                exc_info=True,
            )
