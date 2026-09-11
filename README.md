# comfyui-gcs-uploader

A single-purpose [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node that uploads
generated images straight to **Google Cloud Storage**. It talks to **one backend webhook**: first
to turn a short-lived, per-task **token** into **pre-signed PUT URLs**, then to report which object
keys were uploaded.

Why this design?

- **No credentials on the ComfyUI machine.** Only a short-lived, limited-use token travels with the
  workflow. Your backend keeps the service account and signs URLs on demand.
- **Signed URLs are minted when the upload happens**, not when the job is submitted, so queue time
  can never make them expire.
- **Batch size is discovered at runtime.** The node asks for exactly as many URLs as it has images.
- **Completion is reported by the node itself**, authenticated by the token. You never have to trust
  a third-party runner's callback to learn that the task succeeded.
- **Zero extra dependencies.** Only `numpy`, `Pillow`, `requests` — all bundled with ComfyUI.

This directory is fully self-contained and can be published as its own git repository as-is.
The reference backend implementation lives in the same monorepo
(`apps/backend/image_service`, route `POST /webhooks/comfy-upload`); the design write-up is in
`docs/backend/comfyui-gcs-upload.md`.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> comfyui-gcs-uploader
```

Restart ComfyUI. The node appears under **image/upload → Upload Images to GCS (webhook)**.

Python ≥ 3.9. No `pip install` step is needed on a stock ComfyUI environment.

## Node: `UploadImagesToGcs`

| Input             | Type     | Notes                                                                                   |
| ----------------- | -------- | --------------------------------------------------------------------------------------- |
| `images`          | `IMAGE`  | Batch of `[B, H, W, C]`. One upload per image.                                          |
| `webhook_url`     | `STRING` | Backend endpoint that receives both events below. Injected by the caller per job.       |
| `token`           | `STRING` | Per-task token issued by the backend at submit time (short TTL, limited uses). Injected per job. |
| `format`          | enum     | `png` (default) / `jpeg` / `webp`. Determines the `Content-Type` sent to the backend and to GCS. |
| `quality`         | `INT`    | Optional, `jpeg`/`webp` only. Default 95.                                               |
| `timeout_seconds` | `INT`    | Optional per-upload timeout. Default 120. (Webhook calls use a fixed 30s.)              |
| `task_id`         | `STRING` | Optional. Your backend's task id, echoed into every `log` event and stdout line for correlation. |

**Output** — this is an `OUTPUT_NODE`. It also returns the keys via `ui`, so they show up in the
ComfyUI history (and any wrapper that forwards it) as:

```json
"outputs": {
  "<node_id>": {
    "object_key": ["requests/abc/0.png", "requests/abc/1.png"]
  }
}
```

> ComfyUI flattens every `ui` value into a list, so `object_key` is **always a list**, even for a
> single image.

## Webhook protocol

Both calls are `POST <webhook_url>` with `Content-Type: application/json`. The `event` field tells
them apart.

### 1. `request_urls` — consumes one token use

```json
{"token": "<token>", "event": "request_urls", "count": 2, "content_type": "image/png"}
```

Expected `2xx` response (snake_case or camelCase both work):

```json
{"object_keys": ["requests/abc/0.png", "requests/abc/1.png"],
 "signed_urls": ["https://storage.googleapis.com/...", "https://storage.googleapis.com/..."]}
```

### 2. `completed` — validates the token, does not consume a use

Sent after **every** image has been uploaded successfully:

```json
{"token": "<token>", "event": "completed", "object_keys": ["requests/abc/0.png", "requests/abc/1.png"]}
```

Any `2xx` response is fine; the body is ignored.

### 3. `log` — best-effort progress, never retried, never fatal

```json
{"token": "<token>", "event": "log", "stage": "node.uploaded", "level": "info",
 "message": "", "fields": {"index": 0, "object_key": "requests/abc/0.png", "bytes": 812345,
                           "elapsed_ms": 430, "task_id": "abc"}}
```

Stages emitted: `node.started` (`batch_size`, `format`), one `node.uploaded` per image,
`node.failed` (`level=error`, `message` is the error text) before the prompt fails. The same line
is also printed to ComfyUI stdout. A `log` call makes exactly one HTTP attempt with a 10s timeout
and swallows every error; the backend may respond with anything.

### Backend contract

- Issue a token per task with a **TTL** and a **use budget of 3** `request_urls` calls. Normal
  runs use one; the spare uses cover the node's retries after a lost response.
- Sign **V4 `PUT`** URLs with `content_type` equal to the one in the request; GCS rejects mismatches
  with `403`. A few minutes of TTL is enough — the upload starts immediately.
- On `completed`, verify the token is valid and that `object_keys` match what you issued, then mark
  the task done.
- Reject with `4xx` (e.g. `403`) when the token is expired, exhausted, or the task is already in a
  terminal state. The node fails fast on any `4xx` and surfaces the response body.
- The node **does not** report failures. Learn about failed / cancelled runs from your runner's own
  callback and invalidate the token there.

Example signing with `google-cloud-storage`:

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
            "webhook_url": "https://api.example.com/webhooks/comfy-upload",
            "token": task_token,
            "format": "png",
        }
    }
}
```

## Failure behaviour

- Each webhook call and each upload retries 3 times on network errors, `429` and `5xx`
  (backoff 1s, 2s). Any other `4xx` fails immediately.
- Any failure raises and fails the whole prompt; there is no partial success. No upload is attempted
  if `request_urls` fails, and no `completed` event is sent if any upload fails.

## Development

```bash
uv sync            # or: pip install numpy Pillow requests pytest
uv run pytest      # or: pytest
```

Tests mock `requests.post` / `requests.put`; no network or GCP access is needed.
