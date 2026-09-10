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
    exchange_token,
    put_with_retry,
)

EXCHANGE_URL = "https://backend.example/image.v1.UploadService/ExchangeUploadToken"
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


def _exchange_ok(keys: list[str], urls: list[str], camel: bool = False) -> mock.Mock:
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


# --- exchange_token ----------------------------------------------------------


def test_exchange_posts_json_and_parses_snake_case():
    with mock.patch.object(
        requests, "post", return_value=_exchange_ok(["k0", "k1"], ["u0", "u1"])
    ) as post:
        keys, urls = exchange_token(EXCHANGE_URL, "tok", 2, "image/png", sleep=lambda _: None)
    assert (keys, urls) == (["k0", "k1"], ["u0", "u1"])
    assert post.call_args.args[0] == EXCHANGE_URL
    assert post.call_args.kwargs["json"] == {
        "token": "tok",
        "count": 2,
        "content_type": "image/png",
    }
    assert post.call_args.kwargs["headers"] == {"Content-Type": "application/json"}


def test_exchange_accepts_camel_case():
    with mock.patch.object(requests, "post", return_value=_exchange_ok(["k0"], ["u0"], camel=True)):
        assert exchange_token(EXCHANGE_URL, "tok", 1, "image/png", sleep=lambda _: None) == (
            ["k0"],
            ["u0"],
        )


@pytest.mark.parametrize(
    ("url", "token"),
    [("", "tok"), ("   ", "tok"), (EXCHANGE_URL, ""), (EXCHANGE_URL, "  ")],
)
def test_exchange_rejects_missing_inputs(url, token):
    with pytest.raises(GcsUploadError):
        exchange_token(url, token, 1, "image/png")


def test_exchange_rejects_size_mismatch():
    with (
        mock.patch.object(requests, "post", return_value=_exchange_ok(["k0"], ["u0"])),
        pytest.raises(GcsUploadError, match="size mismatch"),
    ):
        exchange_token(EXCHANGE_URL, "tok", 2, "image/png", sleep=lambda _: None)


@pytest.mark.parametrize(
    "body",
    [
        {"object_keys": ["k"], "signed_urls": [1]},
        {"object_keys": "k", "signed_urls": ["u"]},
        {"signed_urls": ["u"]},
        ["not", "a", "dict"],
    ],
)
def test_exchange_rejects_malformed_payload(body):
    with (
        mock.patch.object(requests, "post", return_value=_response(200, json_body=body)),
        pytest.raises(GcsUploadError),
    ):
        exchange_token(EXCHANGE_URL, "tok", 1, "image/png", sleep=lambda _: None)


def test_exchange_rejects_non_json_body():
    with (
        mock.patch.object(requests, "post", return_value=_response(200, text="<html>")),
        pytest.raises(GcsUploadError, match="non-JSON"),
    ):
        exchange_token(EXCHANGE_URL, "tok", 1, "image/png", sleep=lambda _: None)


def test_exchange_fails_fast_on_4xx_with_connect_error_message():
    body = {"code": "permission_denied", "message": "token already used"}
    with (
        mock.patch.object(
            requests,
            "post",
            return_value=_response(403, text='{"message":"token already used"}', json_body=body),
        ) as post,
        pytest.raises(GcsUploadError, match="HTTP 403.*token already used"),
    ):
        exchange_token(EXCHANGE_URL, "tok", 1, "image/png", sleep=lambda _: None)
    post.assert_called_once()


def test_exchange_retries_on_5xx_then_succeeds():
    responses = [_response(503), _exchange_ok(["k0"], ["u0"])]
    sleeps: list[float] = []
    with mock.patch.object(requests, "post", side_effect=responses) as post:
        exchange_token(EXCHANGE_URL, "tok", 1, "image/png", sleep=sleeps.append)
    assert post.call_count == 2
    assert sleeps == [1.0]


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


# --- node --------------------------------------------------------------------


def test_node_exchanges_by_batch_size_uploads_and_returns_keys():
    keys = ["req/0.png", "req/1.png"]
    urls = ["https://u/0", "https://u/1"]
    with (
        NO_SLEEP,
        mock.patch.object(requests, "post", return_value=_exchange_ok(keys, urls)) as post,
        mock.patch.object(requests, "put", return_value=_response(200)) as put,
    ):
        result = UploadImagesToGcs().upload(
            [_image(), _image()], EXCHANGE_URL, "tok", format="jpeg"
        )
    assert result == {"ui": {"object_key": keys}}
    assert post.call_args.kwargs["json"]["count"] == 2
    assert post.call_args.kwargs["json"]["content_type"] == "image/jpeg"
    assert [c.args[0] for c in put.call_args_list] == urls
    assert all(c.kwargs["headers"] == {"Content-Type": "image/jpeg"} for c in put.call_args_list)


def test_node_rejects_empty_batch():
    with pytest.raises(GcsUploadError, match="empty image batch"):
        UploadImagesToGcs().upload([], EXCHANGE_URL, "tok")


def test_node_does_not_upload_when_exchange_fails():
    with (
        NO_SLEEP,
        mock.patch.object(requests, "post", return_value=_response(403, "nope")),
        mock.patch.object(requests, "put") as put,
        pytest.raises(GcsUploadError, match="token exchange"),
    ):
        UploadImagesToGcs().upload([_image()], EXCHANGE_URL, "tok")
    put.assert_not_called()


def test_node_error_mentions_failing_key():
    with (
        NO_SLEEP,
        mock.patch.object(
            requests, "post", return_value=_exchange_ok(["req/0.png"], ["https://u"])
        ),
        mock.patch.object(requests, "put", return_value=_response(403)),
        pytest.raises(GcsUploadError, match=r"image #0 \(req/0.png\)"),
    ):
        UploadImagesToGcs().upload([_image()], EXCHANGE_URL, "tok")


def test_node_metadata_matches_comfy_conventions():
    assert UploadImagesToGcs.OUTPUT_NODE is True
    assert UploadImagesToGcs.RETURN_TYPES == ()
    required = UploadImagesToGcs.INPUT_TYPES()["required"]
    assert set(required) == {"images", "exchange_url", "token", "format"}
    assert nodes.NODE_CLASS_MAPPINGS["UploadImagesToGcs"] is UploadImagesToGcs
