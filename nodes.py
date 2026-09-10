"""Upload ComfyUI IMAGE batches to Google Cloud Storage through backend-issued signed PUT URLs.

Flow
----
1. The caller submits the workflow with two inputs on this node: ``exchange_url`` (a backend
   endpoint) and a per-task, single-purpose ``token``.
2. When the node runs it knows the real batch size, so it POSTs ``{token, count, content_type}``
   to ``exchange_url`` and receives ``{object_keys[], signed_urls[]}``. Signed URLs are therefore
   minted at upload time, not at submit time, and never sit in a queue waiting to expire.
3. Each image is encoded and ``PUT`` to its signed URL.
4. The object keys are returned through ``ui`` so they appear under ``outputs[<node_id>]`` in the
   ComfyUI history / RunComfy webhook payload as ``{"object_key": ["key-1", ...]}``
   (ComfyUI flattens every ``ui`` value into a list).

No Google SDK and no long-lived credentials ever reach the ComfyUI machine. The exchange endpoint is
expected to be idempotent for a token until the task reaches a terminal state, so retrying the
exchange after a network hiccup is safe.

Python >= 3.9; only depends on packages ComfyUI already ships with (``numpy``, ``Pillow``,
``requests``).
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import requests
from PIL import Image

_FORMATS: dict[str, tuple[str, str]] = {
    # name -> (PIL format, content type)
    "png": ("PNG", "image/png"),
    "jpeg": ("JPEG", "image/jpeg"),
    "webp": ("WEBP", "image/webp"),
}

_DEFAULT_TIMEOUT_S = 120
_EXCHANGE_TIMEOUT_S = 30
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.0


class GcsUploadError(RuntimeError):
    """Raised when the exchange or an upload fails; fails the whole prompt on purpose."""


def tensor_to_pil(image: Any) -> Image.Image:
    """Convert one ``[H, W, C]`` float tensor/array in ``[0, 1]`` to a PIL image."""
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    array = np.asarray(image)
    array = np.clip(255.0 * array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def encode_image(image: Any, fmt: str, quality: int) -> bytes:
    pil_format, _ = _FORMATS[fmt]
    pil_image = tensor_to_pil(image)
    if fmt == "jpeg" and pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if fmt in ("jpeg", "webp"):
        save_kwargs["quality"] = quality
    pil_image.save(buffer, format=pil_format, **save_kwargs)
    return buffer.getvalue()


def _is_retryable(response: requests.Response | None, exc: Exception | None) -> bool:
    if exc is not None:
        return True  # connection errors / timeouts
    assert response is not None
    return response.status_code >= 500 or response.status_code == 429


def _request_with_retry(
    what: str,
    send: Callable[[], requests.Response],
    max_attempts: int = _MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> requests.Response:
    """Run ``send`` until it returns 2xx. Retries on 429/5xx/network errors, fails fast on other 4xx."""
    last_error = ""
    attempt = 0
    for attempt in range(1, max_attempts + 1):
        response: requests.Response | None = None
        exc: Exception | None = None
        try:
            response = send()
            if response.ok:
                return response
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except requests.RequestException as caught:
            exc = caught
            last_error = repr(caught)

        if not _is_retryable(response, exc) or attempt == max_attempts:
            break
        sleep(_BACKOFF_BASE_S * (2 ** (attempt - 1)))

    raise GcsUploadError(f"{what} failed after {attempt} attempt(s): {last_error}")


def _pick(payload: dict[str, Any], *names: str) -> Any:
    """Read the first present key; tolerates snake_case (proto JSON) and camelCase (Connect default)."""
    for name in names:
        if name in payload:
            return payload[name]
    return None


def exchange_token(
    exchange_url: str,
    token: str,
    count: int,
    content_type: str,
    timeout_s: float = _EXCHANGE_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[str], list[str]]:
    """POST the one-time token to the backend and get ``(object_keys, signed_urls)`` for ``count`` images."""
    if not exchange_url.strip():
        raise GcsUploadError("'exchange_url' must not be empty")
    if not token.strip():
        raise GcsUploadError("'token' must not be empty")

    body = {"token": token, "count": count, "content_type": content_type}
    response = _request_with_retry(
        "token exchange",
        lambda: requests.post(
            exchange_url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        ),
        sleep=sleep,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise GcsUploadError(
            f"token exchange returned non-JSON body: {response.text[:300]}"
        ) from exc
    if not isinstance(payload, dict):
        raise GcsUploadError(f"token exchange returned unexpected payload: {payload!r}")

    keys = _pick(payload, "object_keys", "objectKeys")
    urls = _pick(payload, "signed_urls", "signedUrls")
    for name, value in (("object_keys", keys), ("signed_urls", urls)):
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise GcsUploadError(
                f"token exchange response '{name}' must be a list of non-empty strings"
            )
    if not (len(keys) == len(urls) == count):
        raise GcsUploadError(
            f"token exchange size mismatch: requested {count}, got keys={len(keys)}, urls={len(urls)}"
        )
    return keys, urls


def put_with_retry(
    url: str,
    data: bytes,
    content_type: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_attempts: int = _MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """``PUT`` bytes to a signed URL with the retry policy of ``_request_with_retry``."""
    _request_with_retry(
        "upload",
        lambda: requests.put(
            url, data=data, headers={"Content-Type": content_type}, timeout=timeout_s
        ),
        max_attempts=max_attempts,
        sleep=sleep,
    )


class UploadImagesToGcs:
    """Exchange a one-time token for signed URLs, then upload each image of the batch to GCS."""

    CATEGORY = "image/upload"
    FUNCTION = "upload"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):  # ComfyUI convention
        return {
            "required": {
                "images": ("IMAGE",),
                "exchange_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Backend endpoint that swaps the token for signed PUT URLs "
                        "(POST JSON {token, count, content_type}).",
                    },
                ),
                "token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Per-task upload token issued by the backend when the job was submitted.",
                    },
                ),
                "format": (list(_FORMATS.keys()), {"default": "png"}),
            },
            "optional": {
                "quality": (
                    "INT",
                    {"default": 95, "min": 1, "max": 100, "tooltip": "jpeg / webp only"},
                ),
                "timeout_seconds": (
                    "INT",
                    {"default": _DEFAULT_TIMEOUT_S, "min": 1, "max": 3600},
                ),
            },
        }

    def upload(
        self,
        images: Sequence[Any],
        exchange_url: str,
        token: str,
        format: str = "png",
        quality: int = 95,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
    ):
        batch_size = len(images)
        if batch_size == 0:
            raise GcsUploadError("received an empty image batch")

        _, content_type = _FORMATS[format]
        keys, urls = exchange_token(exchange_url, token, batch_size, content_type)

        for index, (image, url) in enumerate(zip(images, urls)):
            data = encode_image(image, format, quality)
            try:
                put_with_retry(url, data, content_type, timeout_s=timeout_seconds)
            except GcsUploadError as exc:
                raise GcsUploadError(f"image #{index} ({keys[index]}): {exc}") from exc

        return {"ui": {"object_key": keys}}


NODE_CLASS_MAPPINGS = {"UploadImagesToGcs": UploadImagesToGcs}
NODE_DISPLAY_NAME_MAPPINGS = {"UploadImagesToGcs": "Upload Images to GCS (token exchange)"}
