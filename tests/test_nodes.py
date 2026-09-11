from __future__ import annotations

import io
from unittest import mock

import numpy as np
import pytest
import requests
from PIL import Image

import nodes
from nodes import (
    GcsUploadError,
    UploadImagesToGcs,
    encode_image,
    log_event,
    normalize_webhook_url,
    put_with_retry,
    report_completed,
    request_upload_urls,
)

WEBHOOK_URL = "https://backend.example/webhooks/comfy-upload"
NO_SLEEP = mock.patch.object(nodes.time, "sleep", lambda _: None)


def _response(status: int, text: str = "", json_body: object = None) -> mock.Mock:
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.ok = 200 <= status < 300
    response.text = text
    if json_body is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = json_body
    return response


def _urls_ok(keys: list[str], urls: list[str], camel: bool = False) -> mock.Mock:
    body = (
        {"objectKeys": keys, "signedUrls": urls}
        if camel
        else {"object_keys": keys, "signed_urls": urls}
    )
    return _response(200, json_body=body)


def _image(h: int = 4, w: int = 6) -> np.ndarray:
    return np.random.default_rng(0).random((h, w, 3), dtype=np.float32)


# --- encode_image ------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_encode_image_roundtrip(fmt):
    data = encode_image(_image(), fmt, quality=90)
    pil = Image.open(io.BytesIO(data))
    assert pil.size == (6, 4)
    assert pil.format == nodes._FORMATS[fmt][0]


def test_encode_accepts_torch_like_tensor():
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    data = encode_image(FakeTensor(_image()), "png", quality=95)
    assert Image.open(io.BytesIO(data)).size == (6, 4)


# --- normalize_webhook_url ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api.example/webhooks/comfy-upload", "https://api.example/webhooks/comfy-upload"),
        ("  api.example/x ", "https://api.example/x"),
        ("https://api.example/x", "https://api.example/x"),
        ("http://localhost:8080/x", "http://localhost:8080/x"),
    ],
)
def test_normalize_adds_https_when_scheme_missing(raw, expected):
    assert normalize_webhook_url(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", ".serverless/tmp123.html", "/tmp/x.html"])
def test_normalize_rejects_empty_or_runner_rewritten_paths(raw):
    with pytest.raises(GcsUploadError):
        normalize_webhook_url(raw)


def test_request_urls_posts_to_scheme_restored_url():
    with mock.patch.object(requests, "post", return_value=_urls_ok(["k0"], ["u0"])) as post:
        request_upload_urls("api.example/hook", "tok", 1, "image/png", sleep=lambda _: None)
    assert post.call_args.args[0] == "https://api.example/hook"


# --- request_upload_urls -----------------------------------------------------


def test_request_urls_posts_event_and_parses_snake_case():
    with mock.patch.object(
        requests, "post", return_value=_urls_ok(["k0", "k1"], ["u0", "u1"])
    ) as post:
        keys, urls = request_upload_urls(WEBHOOK_URL, "tok", 2, "image/png", sleep=lambda _: None)
    assert (keys, urls) == (["k0", "k1"], ["u0", "u1"])
    assert post.call_args.args[0] == WEBHOOK_URL
    assert post.call_args.kwargs["json"] == {
        "token": "tok",
        "event": "request_urls",
        "count": 2,
        "content_type": "image/png",
    }
    assert post.call_args.kwargs["headers"] == {"Content-Type": "application/json"}


def test_request_urls_accepts_camel_case():
    with mock.patch.object(requests, "post", return_value=_urls_ok(["k0"], ["u0"], camel=True)):
        assert request_upload_urls(WEBHOOK_URL, "tok", 1, "image/png", sleep=lambda _: None) == (
            ["k0"],
            ["u0"],
        )


@pytest.mark.parametrize(
    ("url", "token"),
    [("", "tok"), ("   ", "tok"), (WEBHOOK_URL, ""), (WEBHOOK_URL, "  ")],
)
def test_request_urls_rejects_missing_inputs(url, token):
    with pytest.raises(GcsUploadError):
        request_upload_urls(url, token, 1, "image/png")


def test_request_urls_rejects_size_mismatch():
    with (
        mock.patch.object(requests, "post", return_value=_urls_ok(["k0"], ["u0"])),
        pytest.raises(GcsUploadError, match="size mismatch"),
    ):
        request_upload_urls(WEBHOOK_URL, "tok", 2, "image/png", sleep=lambda _: None)


@pytest.mark.parametrize(
    "body",
    [
        {"object_keys": ["k"], "signed_urls": [1]},
        {"object_keys": "k", "signed_urls": ["u"]},
        {"signed_urls": ["u"]},
        ["not", "a", "dict"],
    ],
)
def test_request_urls_rejects_malformed_payload(body):
    with (
        mock.patch.object(requests, "post", return_value=_response(200, json_body=body)),
        pytest.raises(GcsUploadError),
    ):
        request_upload_urls(WEBHOOK_URL, "tok", 1, "image/png", sleep=lambda _: None)


def test_request_urls_rejects_non_json_body():
    with (
        mock.patch.object(requests, "post", return_value=_response(200, text="<html>")),
        pytest.raises(GcsUploadError, match="non-JSON"),
    ):
        request_upload_urls(WEBHOOK_URL, "tok", 1, "image/png", sleep=lambda _: None)


def test_request_urls_fails_fast_on_4xx_with_error_message():
    body = {"code": "permission_denied", "message": "token exhausted"}
    with (
        mock.patch.object(
            requests,
            "post",
            return_value=_response(403, text='{"message":"token exhausted"}', json_body=body),
        ) as post,
        pytest.raises(GcsUploadError, match="HTTP 403.*token exhausted"),
    ):
        request_upload_urls(WEBHOOK_URL, "tok", 1, "image/png", sleep=lambda _: None)
    post.assert_called_once()


def test_request_urls_retries_on_5xx_then_succeeds():
    responses = [_response(503), _urls_ok(["k0"], ["u0"])]
    sleeps: list[float] = []
    with mock.patch.object(requests, "post", side_effect=responses) as post:
        request_upload_urls(WEBHOOK_URL, "tok", 1, "image/png", sleep=sleeps.append)
    assert post.call_count == 2
    assert sleeps == [1.0]


# --- report_completed --------------------------------------------------------


def test_report_completed_posts_keys():
    with mock.patch.object(requests, "post", return_value=_response(200)) as post:
        report_completed(WEBHOOK_URL, "tok", ["k0", "k1"], sleep=lambda _: None)
    assert post.call_args.kwargs["json"] == {
        "token": "tok",
        "event": "completed",
        "object_keys": ["k0", "k1"],
    }


def test_report_completed_retries_then_fails():
    with (
        mock.patch.object(requests, "post", return_value=_response(500)) as post,
        pytest.raises(GcsUploadError, match="completed failed after 3"),
    ):
        report_completed(WEBHOOK_URL, "tok", ["k0"], sleep=lambda _: None)
    assert post.call_count == 3


# --- put_with_retry ----------------------------------------------------------


def test_put_success_first_try():
    with mock.patch.object(requests, "put", return_value=_response(200)) as put:
        put_with_retry("https://u", b"x", "image/png", sleep=lambda _: None)
    put.assert_called_once()
    assert put.call_args.kwargs["headers"] == {"Content-Type": "image/png"}


def test_put_retries_on_connection_error():
    responses = [requests.ConnectionError("boom"), _response(200)]
    with mock.patch.object(requests, "put", side_effect=responses) as put:
        put_with_retry("https://u", b"x", "image/png", sleep=lambda _: None)
    assert put.call_count == 2


def test_put_fails_fast_on_4xx():
    with (
        mock.patch.object(requests, "put", return_value=_response(403, "denied")) as put,
        pytest.raises(GcsUploadError, match="HTTP 403"),
    ):
        put_with_retry("https://u", b"x", "image/png", sleep=lambda _: None)
    put.assert_called_once()


def test_put_gives_up_after_max_attempts():
    with (
        mock.patch.object(requests, "put", return_value=_response(500)) as put,
        pytest.raises(GcsUploadError, match="after 3 attempt"),
    ):
        put_with_retry("https://u", b"x", "image/png", sleep=lambda _: None)
    assert put.call_count == 3


# --- log_event ---------------------------------------------------------------


def test_log_event_posts_once_and_attaches_task_id():
    with mock.patch.object(requests, "post", return_value=_response(200)) as post:
        log_event(WEBHOOK_URL, "tok", "node.started", fields={"batch_size": 2}, task_id="t-1")
    post.assert_called_once()
    assert post.call_args.kwargs["json"] == {
        "token": "tok",
        "event": "log",
        "stage": "node.started",
        "level": "info",
        "message": "",
        "fields": {"batch_size": 2, "task_id": "t-1"},
    }


def test_log_event_never_raises():
    with mock.patch.object(requests, "post", side_effect=requests.ConnectionError("down")):
        log_event(WEBHOOK_URL, "tok", "node.failed", "boom", level="error")


def test_log_event_skips_network_without_webhook_or_token():
    with mock.patch.object(requests, "post") as post:
        log_event("", "tok", "node.started")
        log_event(WEBHOOK_URL, "", "node.started")
    post.assert_not_called()


# --- node --------------------------------------------------------------------


def _events(post: mock.Mock) -> list[str]:
    return [c.kwargs["json"]["event"] for c in post.call_args_list]


def _stages(post: mock.Mock) -> list[str]:
    return [
        c.kwargs["json"]["stage"] for c in post.call_args_list if c.kwargs["json"]["event"] == "log"
    ]


def test_node_requests_urls_uploads_reports_and_returns_keys():
    keys = ["req/0.png", "req/1.png"]
    urls = ["https://u/0", "https://u/1"]
    log_ok = _response(200)
    with (
        NO_SLEEP,
        mock.patch.object(
            requests,
            "post",
            side_effect=[log_ok, _urls_ok(keys, urls), log_ok, log_ok, _response(200)],
        ) as post,
        mock.patch.object(requests, "put", return_value=_response(200)) as put,
    ):
        result = UploadImagesToGcs().upload(
            [_image(), _image()], WEBHOOK_URL, "tok", format="jpeg", task_id="t-1"
        )
    assert result == {"ui": {"object_key": keys}}
    assert _events(post) == ["log", "request_urls", "log", "log", "completed"]
    assert _stages(post) == ["node.started", "node.uploaded", "node.uploaded"]
    request = post.call_args_list[1].kwargs["json"]
    assert request["count"] == 2
    assert request["content_type"] == "image/jpeg"
    uploaded = post.call_args_list[2].kwargs["json"]["fields"]
    assert uploaded["object_key"] == "req/0.png"
    assert uploaded["task_id"] == "t-1"
    assert uploaded["bytes"] > 0
    assert post.call_args_list[4].kwargs["json"] == {
        "token": "tok",
        "event": "completed",
        "object_keys": keys,
    }
    assert [c.args[0] for c in put.call_args_list] == urls
    assert all(c.kwargs["headers"] == {"Content-Type": "image/jpeg"} for c in put.call_args_list)


def test_node_rejects_empty_batch():
    with pytest.raises(GcsUploadError, match="empty image batch"):
        UploadImagesToGcs().upload([], WEBHOOK_URL, "tok")


def test_node_does_not_upload_when_request_urls_fails():
    with (
        NO_SLEEP,
        mock.patch.object(requests, "post", return_value=_response(403, "nope")) as post,
        mock.patch.object(requests, "put") as put,
        pytest.raises(GcsUploadError, match="request_urls"),
    ):
        UploadImagesToGcs().upload([_image()], WEBHOOK_URL, "tok")
    put.assert_not_called()
    assert _events(post) == ["log", "request_urls", "log"]  # no "completed" callback
    assert _stages(post) == ["node.started", "node.failed"]
    assert post.call_args_list[-1].kwargs["json"]["level"] == "error"


def test_node_does_not_report_completed_when_upload_fails():
    with (
        NO_SLEEP,
        mock.patch.object(
            requests,
            "post",
            side_effect=[_response(200), _urls_ok(["req/0.png"], ["https://u"]), _response(200)],
        ) as post,
        mock.patch.object(requests, "put", return_value=_response(403)),
        pytest.raises(GcsUploadError, match=r"image #0 \(req/0.png\)"),
    ):
        UploadImagesToGcs().upload([_image()], WEBHOOK_URL, "tok")
    assert _events(post) == ["log", "request_urls", "log"]
    assert _stages(post) == ["node.started", "node.failed"]
    assert "req/0.png" in post.call_args_list[-1].kwargs["json"]["message"]


def test_node_log_failure_does_not_break_upload():
    keys, urls = ["req/0.png"], ["https://u"]
    with (
        NO_SLEEP,
        mock.patch.object(
            requests,
            "post",
            side_effect=[
                requests.ConnectionError("log down"),
                _urls_ok(keys, urls),
                requests.ConnectionError("log down"),
                _response(200),
            ],
        ),
        mock.patch.object(requests, "put", return_value=_response(200)),
    ):
        assert UploadImagesToGcs().upload([_image()], WEBHOOK_URL, "tok") == {
            "ui": {"object_key": keys}
        }


def test_node_metadata_matches_comfy_conventions():
    assert UploadImagesToGcs.OUTPUT_NODE is True
    assert UploadImagesToGcs.RETURN_TYPES == ()
    required = UploadImagesToGcs.INPUT_TYPES()["required"]
    assert set(required) == {"images", "webhook_url", "token", "format"}
    assert nodes.NODE_CLASS_MAPPINGS["UploadImagesToGcs"] is UploadImagesToGcs
