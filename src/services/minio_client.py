"""
services/minio_client.py

Wrapper around the MinIO Python SDK for downloading MRI files
and uploading generated heatmap images.
"""

import logging

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinIOClient:
    """Thin helper around :class:`minio.Minio`."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        secure = endpoint.startswith("https://")
        endpoint = endpoint.replace("https://", "").replace("http://", "")

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def ensure_bucket(self, bucket_name: str) -> None:
        """Create *bucket_name* if it does not already exist."""
        try:
            if not self._client.bucket_exists(bucket_name):
                self._client.make_bucket(bucket_name)
                logger.info("Created bucket '%s'", bucket_name)
            else:
                logger.debug("Bucket '%s' already exists", bucket_name)
        except S3Error:
            logger.error("Failed to ensure bucket '%s'", bucket_name, exc_info=True)
            raise

    def download_file(
        self, bucket: str, object_key: str, local_path: str
    ) -> None:
        """Download *object_key* from *bucket* to *local_path*."""
        self._client.fget_object(bucket, object_key, local_path)
        logger.info("Downloaded %s/%s -> %s", bucket, object_key, local_path)

    def upload_file(
        self,
        bucket: str,
        object_key: str,
        local_path: str,
        content_type: str | None = None,
    ) -> None:
        """Upload *local_path* to *bucket*/*object_key*."""
        kwargs = {}
        if content_type:
            kwargs["content_type"] = content_type
        self._client.fput_object(bucket, object_key, local_path, **kwargs)
        logger.info("Uploaded %s -> %s/%s", local_path, bucket, object_key)
