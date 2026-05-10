"""
services/rabbitmq_consumer.py

Long-running RabbitMQ consumer with automatic reconnection.
Consumes one message at a time (prefetch_count=1) to preserve FIFO order.
"""

import logging
import time
from typing import Callable

import pika
import pika.exceptions

logger = logging.getLogger(__name__)

_INITIAL_RETRY_DELAY = 5
_MAX_RETRY_DELAY = 60


class RabbitMQConsumer:
    """Blocking consumer that reconnects automatically on failure."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        queue: str,
        on_message: Callable[[bytes], None],
    ):
        self._host = host
        self._port = port
        self._credentials = pika.PlainCredentials(user, password)
        self._queue = queue
        self._on_message = on_message
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.adapters.blocking_connection.BlockingChannel | None = None

    def _connect(self) -> None:
        params = pika.ConnectionParameters(
            host=self._host,
            port=self._port,
            credentials=self._credentials,
            heartbeat=1800,
            blocked_connection_timeout=600,
        )
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()

        # passive=True: verify the queue exists (declared by the Java backend)
        self._channel.queue_declare(queue=self._queue, passive=True)
        self._channel.basic_qos(prefetch_count=1)

        logger.info(
            "Connected to RabbitMQ at %s:%s, queue '%s'",
            self._host,
            self._port,
            self._queue,
        )

    def _handle_delivery(self, channel, method, _properties, body):
        """Called by pika for each incoming message."""
        tag = method.delivery_tag
        logger.info("Received message (delivery_tag=%s, size=%d bytes)", tag, len(body))

        try:
            self._on_message(body)
            channel.basic_ack(delivery_tag=tag)
            logger.info("ACK delivery_tag=%s", tag)
        except Exception:
            logger.error("Processing failed, sending NACK (delivery_tag=%s)", tag, exc_info=True)
            channel.basic_nack(delivery_tag=tag, requeue=False)

    def start(self) -> None:
        """Enter the consume-reconnect loop.  Blocks indefinitely."""
        retry_delay = _INITIAL_RETRY_DELAY

        while True:
            try:
                self._connect()
                self._channel.basic_consume(
                    queue=self._queue,
                    on_message_callback=self._handle_delivery,
                )
                retry_delay = _INITIAL_RETRY_DELAY
                logger.info("Waiting for messages on '%s' …", self._queue)
                self._channel.start_consuming()

            except KeyboardInterrupt:
                logger.info("Shutdown requested (KeyboardInterrupt)")
                self._stop_gracefully()
                break

            except pika.exceptions.ConnectionClosedByBroker as exc:
                logger.warning("Connection closed by broker: %s", exc)

            except pika.exceptions.AMQPConnectionError as exc:
                logger.warning("AMQP connection error: %s", exc)

            except Exception:
                logger.error("Unexpected error in consumer loop", exc_info=True)

            logger.info("Reconnecting in %ds …", retry_delay)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, _MAX_RETRY_DELAY)

        logger.info("Consumer stopped.")

    def _stop_gracefully(self) -> None:
        try:
            if self._channel and self._channel.is_open:
                self._channel.stop_consuming()
        except Exception:
            pass
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass
