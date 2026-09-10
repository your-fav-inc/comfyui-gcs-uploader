"""Upload ComfyUI IMAGE batches to Google Cloud Storage through backend-issued signed PUT URLs.

Flow
----
1. The caller submits the workflow with two inputs on this node: ``webhook_url`` (a backend
   endpoint) and a per-task ``token`` with a short TTL and a small use budget.
2. When the node runs it knows the real batch size, so it POSTs
   ``{token, event: "request_urls", count, content_type}`` to ``webhook_url`` and receives
   ``{object_keys[], signed_urls[]}``. Signed URLs are minted at upload time, not at submit time,
   so queue time can never expire them. Each ``request_urls`` call consumes one use of the token.
3. Each image is encoded and ``PUT`` to its signed URL.
4. The node POSTs ``{token, event: "completed", object_keys}`` to the same ``webhook_url`` so the
   backend can mark the task done without trusting a third-party callback. This call validates the
   token but does not consume a use.
5. The object keys are also returned through ``ui`` so they appear under ``outputs[<node_id>]`` in
   the ComfyUI history as ``{"object_key": ["key-1", ...]}`` (ComfyUI flattens ``ui`` values).

No Google SDK and no long-lived credentials ever reach the ComfyUI machine. Upload failures are not
reported by the node; the backend learns about them from the runner's own failure callback.

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
_WEBHOOK_TIMEOUT_S = 30
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.0


class GcsUploadError(RuntimeError):
    """Raised when a webhook call or an upload fails; fails the whole prompt on purpose."""


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


def _post_webhook(
    webhook_url: str,
    body: dict[str, Any],
    what: str,
    timeout_s: float,
    sleep: Callable[[float], None],
) -> requests.Response:
    if not webhook_url.strip():
        raise GcsUploadError("'webhook_url' must not be empty")
    if not str(body.get("token", "")).strip():
        raise GcsUploadError("'token' must not be empty")
    return _request_with_retry(
        what,
        lambda: requests.post(
            webhook_url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        ),
        sleep=sleep,
    )


def request_upload_urls(
    webhook_url: str,
    token: str,
    count: int,
    content_type: str,
    timeout_s: float = _WEBHOOK_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[str], list[str]]:
    """Send ``event=request_urls`` and get ``(object_keys, signed_urls)`` for ``count`` images."""
    body = {
        "token": token,
        "event": "request_urls",
        "count": count,
        "content_type": content_type,
    }
    response = _post_webhook(webhook_url, body, "request_urls", timeout_s, sleep)
    try:
        payload = response.json()
    except ValueError as exc:
        raise GcsUploadError(f"request_urls returned non-JSON body: {response.text[:300]}") from exc
    if not isinstance(payload, dict):
        raise GcsUploadError(f"request_urls returned unexpected payload: {payload!r}")

    keys = _pick(payload, "object_keys", "objectKeys")
    urls = _pick(payload, "signed_urls", "signedUrls")
    for name, value in (("object_keys", keys), ("signed_urls", urls)):
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise GcsUploadError(
                f"request_urls response '{name}' must be a list of non-empty strings"
            )
    if not (len(keys) == len(urls) == count):
        raise GcsUploadError(
            f"request_urls size mismatch: requested {count}, got keys={len(keys)}, urls={len(urls)}"
        )
    return keys, urls


def report_completed(
    webhook_url: str,
    token: str,
    object_keys: list[str],
    timeout_s: float = _WEBHOOK_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Send ``event=completed`` with the uploaded keys. Any non-2xx after retries fails the prompt."""
    body = {"token": token, "event": "completed", "object_keys": object_keys}
    _post_webhook(webhook_url, body, "completed", timeout_s, sleep)


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
    """Request signed URLs via the webhook, upload the batch to GCS, then report completion."""

    CATEGORY = "image/upload"
    FUNCTION = "upload"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):  # ComfyUI convention
        return {
            "required": {
                "images": ("IMAGE",),
                "webhook_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Backend endpoint receiving JSON POSTs: "
                        "event=request_urls (returns signed PUT URLs) and event=completed.",
                    },
                ),
                "token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Per-task upload token issued by the backend when the job was "
                        "submitted (short TTL, limited uses).",
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
        webhook_url: str,
        token: str,
        format: str = "png",
        quality: int = 95,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
    ):
        batch_size = len(images)
        if batch_size == 0:
            raise GcsUploadError("received an empty image batch")

        _, content_type = _FORMATS[format]
        keys, urls = request_upload_urls(webhook_url, token, batch_size, content_type)

        for index, (image, url) in enumerate(zip(images, urls)):
            data = encode_image(image, format, quality)
            try:
                put_with_retry(url, data, content_type, timeout_s=timeout_seconds)
            except GcsUploadError as exc:
                raise GcsUploadError(f"image #{index} ({keys[index]}): {exc}") from exc

        report_completed(webhook_url, token, keys)
        return {"ui": {"object_key": keys}}


NODE_CLASS_MAPPINGS = {"UploadImagesToGcs": UploadImagesToGcs}
NODE_DISPLAY_NAME_MAPPINGS = {"UploadImagesToGcs": "Upload Images to GCS (webhook)"}
