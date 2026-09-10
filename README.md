# comfyui-gcs-uploader

A single-purpose [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node that uploads
generated images straight to **Google Cloud Storage**. It swaps a per-task **one-time token** for
**pre-signed PUT URLs** at upload time, uploads each image, and reports the object keys as the
node's output.

Why this design?

- **No credentials on the ComfyUI machine.** Only a short-lived, single-task token travels with the
  workflow. Your backend keeps the service account and signs URLs on demand.
- **Signed URLs are minted when the upload happens**, not when the job is submitted, so queue time
  can never make them expire.
- **Batch size is discovered at runtime.** The node asks for exactly as many URLs as it has images.
- **Revocable.** The backend can refuse the exchange once the task is cancelled or finished.
- **Zero extra dependencies.** Only `numpy`, `Pillow`, `requests` — all bundled with ComfyUI.

This directory is fully self-contained and can be published as its own git repository as-is.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> comfyui-gcs-uploader
```

Restart ComfyUI. The node appears under **image/upload → Upload Images to GCS (token exchange)**.

Python ≥ 3.9. No `pip install` step is needed on a stock ComfyUI environment.

## Node: `UploadImagesToGcs`

| Input             | Type     | Notes                                                                                  |
| ----------------- | -------- | -------------------------------------------------------------------------------------- |
| `images`          | `IMAGE`  | Batch of `[B, H, W, C]`. One upload per image.                                         |
| `exchange_url`    | `STRING` | Backend endpoint that swaps the token for signed URLs. Injected by the caller per job. |
| `token`           | `STRING` | Per-task one-time token issued by the backend at submit time. Injected per job.        |
| `format`          | enum     | `png` (default) / `jpeg` / `webp`. Determines the `Content-Type` sent to the backend and to GCS. |
| `quality`         | `INT`    | Optional, `jpeg`/`webp` only. Default 95.                                              |
| `timeout_seconds` | `INT`    | Optional per-upload timeout. Default 120. (The exchange call uses a fixed 30s.)        |

**Output** — this is an `OUTPUT_NODE`. It returns the keys via `ui`, so they show up in the ComfyUI
history (and any wrapper that forwards it, e.g. RunComfy's webhook) as:

```json
"outputs": {
  "<node_id>": {
    "object_key": ["requests/abc/0.png", "requests/abc/1.png"]
  }
}
```

> ComfyUI flattens every `ui` value into a list, so `object_key` is **always a list**, even for a
> single image. Consumers should read `object_key[0]` for `B=1`.

## Exchange protocol

The node sends one `POST` to `exchange_url`:

```http
POST <exchange_url>
Content-Type: application/json

{"token": "<token>", "count": 2, "content_type": "image/png"}
```

and expects a `2xx` JSON response with two equal-length lists (snake_case or camelCase both work,
so a Connect / gRPC-JSON endpoint is fine):

```json
{"object_keys": ["requests/abc/0.png", "requests/abc/1.png"],
 "signed_urls": ["https://storage.googleapis.com/...", "https://storage.googleapis.com/..."]}
```

Backend contract the node relies on:

- Sign **V4 `PUT`** URLs with `content_type` equal to the one in the request; GCS rejects mismatches
  with `403`. A few minutes of TTL is enough — the upload starts immediately.
- Be **idempotent per token until the task is terminal**: repeated calls return the same
  `object_keys` (fresh `signed_urls` are fine). The node retries the exchange on `429`/`5xx`/network
  errors, so a lost response must not brick the job.
- Reject with a `4xx` (e.g. `403`) once the token is expired, the task was cancelled/failed, or the
  task already completed. The node fails fast on any `4xx` and surfaces the response body.

Example with `google-cloud-storage`:

```python
from datetime import timedelta
from google.cloud import storage

bucket = storage.Client().bucket("my-bucket")


def sign_put(object_key: str, content_type: str) -> str:
    return bucket.blob(object_key).generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=10),
        method="PUT",
        content_type=content_type,
    )
```

## Injecting inputs per job (RunComfy example)

`NODE_ID` is the id of this node in your exported API-format workflow.

```python
overrides = {
    NODE_ID: {
        "inputs": {
            "exchange_url": "https://api.example.com/image.v1.UploadService/ExchangeUploadToken",
            "token": one_time_token,
            "format": "png",
        }
    }
}
```

## Failure behaviour

- Exchange and each upload retry 3 times on network errors, `429` and `5xx` (backoff 1s, 2s).
  Any other `4xx` fails immediately.
- Any failure raises and fails the whole prompt; there is no partial success. No upload is
  attempted if the exchange fails.

## Development

```bash
uv sync            # or: pip install numpy Pillow requests pytest
uv run pytest      # or: pytest
```

Tests mock `requests.post` / `requests.put`; no network or GCP access is needed.
