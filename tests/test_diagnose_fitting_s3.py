"""One-shot CLI tests: no real AWS, Runware or HTTP requests."""

import builtins
import importlib.util
import json
import logging
import os
import socket
import sys
import warnings
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_fitting_s3.py"
if not SCRIPT.is_file():
    pytest.skip("scripts/ is not copied into the Docker test stage", allow_module_level=True)

BUCKET = "lookddak-diagnostic-test"
HOST = f"{BUCKET}.s3.ap-northeast-2.amazonaws.com"
KEY = "body-images/original.webp"
ACCESS = "FAKE_ACCESS_FOR_DIAGNOSTIC_TEST"
SIGNATURE = "FAKE_SIGNATURE_FOR_DIAGNOSTIC_TEST"
TOKEN = "FAKE_TOKEN_FOR_DIAGNOSTIC_TEST"


def original_url(host=HOST, path=KEY, **query_overrides):
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{ACCESS}/20261112/ap-northeast-2/s3/aws4_request",
        "X-Amz-Date": "20261112T010000Z",
        "X-Amz-Expires": "1800",
        "X-Amz-SignedHeaders": "host",
        "X-Amz-Signature": SIGNATURE,
        "X-Amz-Security-Token": TOKEN,
    }
    query.update(query_overrides)
    return f"https://{host}/{path}?{urlencode(query)}"


@pytest.fixture
def diagnostic(monkeypatch):
    spec = importlib.util.spec_from_file_location("fitting_s3_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in list(os.environ):
        if name.startswith("AWS_") or name == "S3_BUCKET":
            monkeypatch.delenv(name)
    monkeypatch.setattr(module, "utcnow", lambda: datetime(2026, 11, 12, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(
        module, "load_settings", Mock(side_effect=AssertionError("Loaded settings"))
    )
    monkeypatch.setattr(module, "aws_checks", Mock(side_effect=AssertionError("Used AWS")))
    monkeypatch.setattr(socket, "create_connection", Mock(side_effect=AssertionError("Real HTTP")))
    disabled = logging.root.manager.disable
    yield module
    logging.disable(disabled)


def invoke(diagnostic, monkeypatch, capsys, url, *options):
    prompt = Mock(return_value=url)
    monkeypatch.setattr(diagnostic.getpass, "getpass", prompt)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--original-url-only", *options])
    real_import = builtins.__import__

    def no_aws_or_service_import(name, *args, **kwargs):
        if name.split(".")[0] in {"app", "boto3", "botocore"}:
            raise AssertionError("Imported service/AWS code")
        return real_import(name, *args, **kwargs)

    # Guard imports even when other tests have already imported the AWS SDK/settings.
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", no_aws_or_service_import)
        code = diagnostic.main()
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    prompt.assert_called_once_with("Original URL (hidden): ")
    for secret in (url, ACCESS, SIGNATURE, TOKEN, "CanonicalRequest", "PRIVATE_ERROR_BODY"):
        if secret:
            assert secret not in captured.out + captured.err
    diagnostic.load_settings.assert_not_called()
    diagnostic.aws_checks.assert_not_called()
    return code, report


def http_response(diagnostic, monkeypatch, url, body, code=200, mime="image/webp", **headers):
    response = HTTPError(
        url,
        code,
        "PRIVATE_ERROR_BODY",
        {"Content-Type": mime, "Content-Length": str(len(body)), **headers},
        BytesIO(body),
    )
    opener = Mock()
    if code == 200:
        opener.open.return_value = response
    else:
        opener.open.side_effect = response
    factory = Mock(return_value=opener)
    monkeypatch.setattr(diagnostic, "build_opener", factory)
    return opener, factory


@pytest.mark.parametrize(
    "fmt,extension,mime",
    [
        ("WEBP", "webp", "image/webp"),
        ("JPEG", "jpg", "image/jpeg"),
        ("PNG", "png", "image/png"),
    ],
)
def test_only_mode_without_aws_settings_decodes_get_response(
    diagnostic,
    monkeypatch,
    capsys,
    tmp_path,
    fmt,
    extension,
    mime,
):
    url = original_url(path=f"body-images/original.{extension}")
    output = BytesIO()
    Image.new("RGB", (4, 6), "blue").save(output, format=fmt)
    body = output.getvalue()
    opener, _ = http_response(diagnostic, monkeypatch, url, body, mime=mime)
    # Even an explicitly supplied nonexistent env file must not be read in this mode.
    code, report = invoke(
        diagnostic, monkeypatch, capsys, url, "--env-file", str(tmp_path / "missing")
    )
    assert code == 0
    assert report["mode"] == "original_url_only"
    assert report["runware_called"] is False
    assert "aws_credentials" not in report and "new_url" not in report
    assert report["original_url"]["https_s3_object_url"] is True
    get = report["original_http_get"]
    assert get["http_status"] == 200 and get["bytes_read"] == len(body)
    assert get["content_type"] == mime and get["content_length_header"] == len(body)
    assert get["image_decode"] == {
        "status": "ok",
        "format": fmt,
        "width": 4,
        "height": 6,
        "frames": 1,
        "extension_matches_bytes": True,
    }
    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    assert request.full_url == url and request.get_method() == "GET"
    assert not request.has_header("Authorization") and not request.has_header("Range")


@pytest.mark.parametrize(
    "host,path,style",
    [
        (HOST, KEY, "virtual"),
        (f"{BUCKET}.s3.amazonaws.com", KEY, "virtual"),
        (f"{BUCKET}.s3-ap-northeast-2.amazonaws.com", KEY, "virtual"),
        (f"{BUCKET}.s3.dualstack.ap-northeast-2.amazonaws.com", KEY, "virtual"),
        ("s3.ap-northeast-2.amazonaws.com", f"{BUCKET}/{KEY}", "path"),
        ("s3.amazonaws.com", f"{BUCKET}/{KEY}", "path"),
        (f"{BUCKET}.s3.cn-north-1.amazonaws.com.cn", KEY, "virtual"),
    ],
)
def test_standalone_s3_endpoint_formats(diagnostic, host, path, style):
    info, key = diagnostic.standalone_url_info(original_url(host=host, path=path))
    assert info["status"] == "ok"
    assert info["s3_host_valid"] and info["https_s3_object_url"]
    assert info["addressing_style"] == style and key == KEY
    assert "configured_regional_host" not in info


@pytest.mark.parametrize(
    "url",
    [
        original_url(host="example.com"),
        original_url(host=HOST + ".example.com"),
        original_url().replace("https:", "http:"),
        original_url(host="user:password@" + HOST),
        original_url(host=HOST + ":not-a-port"),
        original_url() + "#fragment",
        original_url().replace("https:", "https:\n"),
        original_url(path=""),
        original_url(host="s3.amazonaws.com", path=BUCKET),
        "https://[invalid/",
        "",
    ],
)
def test_invalid_urls_never_make_http_request(diagnostic, monkeypatch, capsys, url):
    factory = Mock(side_effect=AssertionError("GET must be skipped"))
    monkeypatch.setattr(diagnostic, "build_opener", factory)
    code, report = invoke(diagnostic, monkeypatch, capsys, url)
    assert code == 1
    assert report["original_url"]["status"] == "failed"
    assert report["original_http_get"]["reason"] == "invalid_s3_object_url"
    factory.assert_not_called()


def test_expired_403_does_not_infer_historical_runware_failure(diagnostic, monkeypatch, capsys):
    url = original_url()
    monkeypatch.setattr(diagnostic, "utcnow", lambda: datetime(2026, 11, 12, 2, tzinfo=UTC))
    body = (
        "<Error><Code>AccessDenied</Code><Message>Request has expired</Message>"
        f"<CanonicalRequest>{escape(url)}</CanonicalRequest>"
        "<Debug>PRIVATE_ERROR_BODY</Debug></Error>"
    ).encode()
    http_response(diagnostic, monkeypatch, url, body, code=403, mime="application/xml")
    code, report = invoke(diagnostic, monkeypatch, capsys, url)
    assert code == 1  # Diagnosis runs normally; the CURRENT GET failed.
    info = report["original_url"]
    assert info["issued_at_utc"] == "2026-11-12T01:00:00+00:00"
    assert info["configured_expires_seconds"] == 1800
    assert info["nominal_expires_at_utc"] == "2026-11-12T01:30:00+00:00"
    assert info["nominal_expired"] is True and info["temporary_token_present"] is True
    assert info["credential_expiry_from_url"] == "not_available"
    assert info["effective_expiry"] == "may_be_earlier_due_to_credentials_or_policy"
    assert report["historical_original_access"] == "not_proven_by_current_probe"
    assert report["current_403_does_not_prove_original_runware_failure"] is True
    get = report["original_http_get"]
    assert get["http_status"] == 403 and get["s3_message_request_expired"] is True
    assert get["content_type"] == "application/xml" and get["bytes_read"] == len(body)
    assert get["image_decode"]["status"] == "skipped"


@pytest.mark.parametrize(
    "status,error", [(403, "ExpiredToken"), (404, "NoSuchKey"), (500, "InternalError")]
)
def test_http_error_codes_are_reported_safely(diagnostic, monkeypatch, capsys, status, error):
    url = original_url()
    body = f"<Error><Code>{error}</Code><Message>PRIVATE_ERROR_BODY</Message></Error>".encode()
    http_response(diagnostic, monkeypatch, url, body, code=status, mime="application/xml")
    code, report = invoke(diagnostic, monkeypatch, capsys, url)
    assert code == 1
    assert report["original_http_get"]["http_status"] == status
    assert report["original_http_get"]["s3_error_code"] == error


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_is_observed_but_never_followed(diagnostic, monkeypatch, capsys, status):
    url = original_url()
    opener, factory = http_response(
        diagnostic,
        monkeypatch,
        url,
        b"PRIVATE_ERROR_BODY",
        code=status,
        mime="text/html",
        Location="https://api.runware.ai/v1?" + TOKEN,
    )
    code, report = invoke(diagnostic, monkeypatch, capsys, url)
    get = report["original_http_get"]
    assert code == 1 and get["http_status"] == status
    assert get["redirect"] and get["redirect_location_present"]
    assert get["redirects_followed"] is False
    opener.open.assert_called_once()
    handler = factory.call_args.args[0]
    assert handler.redirect_request(None, None, status, "", {}, "https://api.runware.ai") is None


def test_200_corrupt_image_does_not_claim_decode_success(diagnostic, monkeypatch, capsys):
    url = original_url()
    http_response(diagnostic, monkeypatch, url, b"corrupt WEBP bytes")
    code, report = invoke(diagnostic, monkeypatch, capsys, url)
    assert code == 1
    assert report["original_http_get"]["http_status"] == 200
    assert report["original_http_get"]["image_decode"]["status"] == "failed"


def test_download_limit_reports_incomplete_decode(diagnostic, monkeypatch, capsys):
    url = original_url()
    http_response(diagnostic, monkeypatch, url, b"more than two bytes")
    code, report = invoke(diagnostic, monkeypatch, capsys, url, "--max-bytes", "2")
    assert code == 2
    get = report["original_http_get"]
    assert get["bytes_read"] == 3 and get["image_decode"]["reason"] == "download_limit_exceeded"


def test_network_exception_cannot_leak_url(diagnostic, monkeypatch, capsys):
    url = original_url()
    opener = Mock()
    opener.open.side_effect = OSError(url + " PRIVATE_ERROR_BODY")
    monkeypatch.setattr(diagnostic, "build_opener", Mock(return_value=opener))
    code, report = invoke(diagnostic, monkeypatch, capsys, url)
    assert code == 1 and report["original_http_get"]["error_type"] == "OSError"


def test_hidden_input_fails_closed_without_echo(diagnostic, monkeypatch, capsys):
    def no_terminal(prompt):
        warnings.warn("Cannot hide input", diagnostic.getpass.GetPassWarning, stacklevel=2)
        return builtins.input(prompt)

    monkeypatch.setattr(diagnostic.getpass, "getpass", no_terminal)
    plain_input = Mock(side_effect=AssertionError("Must not read echoed input"))
    monkeypatch.setattr(builtins, "input", plain_input)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--original-url-only"])
    assert diagnostic.main() == 1
    assert json.loads(capsys.readouterr().out)["diagnostic"]["error_type"] == "GetPassWarning"
    plain_input.assert_not_called()


@pytest.mark.parametrize(
    "other", [["--original-url-file", "unused.txt"], ["--original-url-prompt"]]
)
def test_only_mode_requires_its_own_hidden_input(diagnostic, monkeypatch, other):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--original-url-only", *other])
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 2


def test_offline_only_mode_reads_metadata_without_http(diagnostic, monkeypatch, capsys):
    factory = Mock(side_effect=AssertionError("Offline GET"))
    monkeypatch.setattr(diagnostic, "build_opener", factory)
    code, report = invoke(diagnostic, monkeypatch, capsys, original_url(), "--offline")
    assert code == 2
    assert report["original_http_get"]["reason"] == "offline"
    factory.assert_not_called()


@pytest.mark.parametrize(
    "query",
    [
        {"X-Amz-Date": "invalid"},
        {"X-Amz-Expires": "invalid"},
        {"X-Amz-Expires": "0"},
        {"X-Amz-Expires": "604801"},
    ],
)
def test_invalid_expiry_is_unknown_instead_of_success(diagnostic, monkeypatch, capsys, query):
    code, report = invoke(
        diagnostic,
        monkeypatch,
        capsys,
        original_url(**query),
        "--offline",
    )
    assert code == 2
    info = report["original_url"]
    assert info["status"] == "unknown" and info["expiry_status"] == "unknown_or_invalid"
    assert info["credential_expiry_from_url"] == "not_available"


def test_full_mode_still_loads_settings_and_calls_aws(diagnostic, monkeypatch, capsys):
    from app.config import settings

    monkeypatch.setattr(settings, "S3_BUCKET", BUCKET)
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS", 1800)
    load = Mock()

    def no_credentials(report, args, bucket, region):
        assert bucket == BUCKET and region == "ap-northeast-2"
        report["aws_credentials"] = {"status": "failed", "error_type": "NoCredentialsError"}
        report["new_url"] = {"status": "skipped", "reason": "credentials_unavailable"}

    aws = Mock(side_effect=no_credentials)
    monkeypatch.setattr(diagnostic, "load_settings", load)
    monkeypatch.setattr(diagnostic, "aws_checks", aws)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--env-file", "selected.env"])
    assert diagnostic.main() == 1
    report = json.loads(capsys.readouterr().out)
    load.assert_called_once_with(Path("selected.env"))
    aws.assert_called_once()
    assert report["aws_credentials"]["error_type"] == "NoCredentialsError"
    assert report["original_url"]["reason"] == "not_supplied"


def test_full_mode_preserves_configured_target_validation(diagnostic):
    info = diagnostic.url_info(original_url(), BUCKET, KEY, "ap-northeast-2")
    assert info["status"] == "ok" and info["https_s3_exact_bucket_key"]
    assert info["configured_regional_host"] and info["signing_region_matches_config"]
    assert (
        diagnostic.url_info(original_url(), "other-bucket", KEY, "ap-northeast-2")["status"]
        == "failed"
    )
