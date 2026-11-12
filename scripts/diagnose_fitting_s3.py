from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import math
import os
import re
import shlex
import sys
import warnings
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree

DEFAULT_KEY = "body-images/dda336d7-9ab2-4a7e-a385-ae644b7a787a.webp"
SETTINGS = {"S3_BUCKET", "AWS_REGION", "VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS"}
MIMES = {
    "image/webp",
    "image/jpeg",
    "image/png",
    "application/octet-stream",
    "text/plain",
    "application/xml",
    "text/xml",
    "text/html",
    "application/json",
}
BUCKET_PATTERN = r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]"
REGION_PATTERN = r"[a-z]{2}-[a-z-]+-\d+"
S3_OBJECT_HOST = re.compile(
    rf"(?:(?P<bucket>{BUCKET_PATTERN})\.)?s3"
    rf"(?:(?:[.-](?P<region>{REGION_PATTERN}))|(?:\.dualstack\.(?P<dual_region>{REGION_PATTERN})))?"
    r"\.amazonaws\.com(?:\.cn)?"
)
S3_CODES = {
    "AccessDenied",
    "ExpiredToken",
    "InvalidToken",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "AuthorizationQueryParametersError",
    "AuthorizationHeaderMalformed",
    "RequestTimeTooSkewed",
    "RequestExpired",
    "NoSuchKey",
    "NoSuchBucket",
    "PermanentRedirect",
    "TemporaryRedirect",
    "InvalidArgument",
    "InvalidRequest",
    "AllAccessDisabled",
    "SlowDown",
    "InternalError",
}


def utcnow():
    return datetime.now(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat() if value else None


def failure(error):
    """Exception messages, traces, SDK responses and causes may contain secrets."""
    result = {"status": "failed", "error_type": type(error).__name__}
    response = getattr(error, "response", {})
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in S3_CODES or code in {"403", "404", "400", "301"}:
            result["s3_error_code"] = code
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(status, int):
            result["http_status"] = status
    return result


def load_settings(path):
    if path is None:
        return
    for line in path.read_text().splitlines():
        name, sep, value = line.removeprefix("export ").partition("=")
        name = name.strip()
        if sep and name in SETTINGS:
            values = shlex.split(value, comments=True)
            if len(values) != 1:
                raise ValueError("Invalid selected setting")
            os.environ.setdefault(name, values[0])


def safe_region(value):
    return (
        value if isinstance(value, str) and re.fullmatch(r"[a-z]{2}-[a-z-]+-\d+", value) else None
    )


def safe_mime(value):
    mime = (value or "").split(";", 1)[0].strip().lower()
    return mime if mime in MIMES else "other_or_missing"


def presigned_url_metadata(url):
    """Allowlisted metadata only. Do not return arbitrary host/query/header text."""
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    credential = query.get("X-Amz-Credential", [""])[0].split("/")
    signing_region = safe_region(credential[2]) if len(credential) == 5 else None
    info = {
        "status": "ok",
        "url_fingerprint_sha256_16": hashlib.sha256(url.encode()).hexdigest()[:16],
        "sigv4": query.get("X-Amz-Algorithm") == ["AWS4-HMAC-SHA256"],
        "has_signature": len(query.get("X-Amz-Signature", [])) == 1,
        "signing_region": signing_region,
        "host_only_signed": query.get("X-Amz-SignedHeaders") == ["host"],
        "temporary_token_present": "X-Amz-Security-Token" in query,
        "credential_expiry_from_url": "not_available",
        "method_from_url": "not_available",
    }
    try:
        if len(query.get("X-Amz-Date", [])) != 1 or len(query.get("X-Amz-Expires", [])) != 1:
            raise ValueError("Missing/duplicate expiry fields")
        issued = datetime.strptime(query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        seconds = int(query["X-Amz-Expires"][0])
        if not 1 <= seconds <= 604800:
            raise ValueError("Invalid expiry")
        expires = issued + timedelta(seconds=seconds)
        now = utcnow()
        info.update(
            issued_at_utc=iso(issued),
            configured_expires_seconds=seconds,
            nominal_expires_at_utc=iso(expires),
            checked_at_utc=iso(now),
            nominal_expired=now >= expires,
            issued_in_future=issued > now,
            nominal_remaining_seconds=int((expires - now).total_seconds()),
            effective_expiry="may_be_earlier_due_to_credentials_or_policy",
        )
    except (ValueError, OverflowError):
        info["expiry_status"] = "unknown_or_invalid"
        if info["status"] == "ok":
            info["status"] = "unknown"
    return info


def url_info(url, bucket, key, region):
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    suffix = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    virtual = f"{bucket}.s3.{region}.{suffix}"
    global_host = f"{bucket}.s3.amazonaws.com"
    # Old URLs may be global, regional (including legacy s3-REGION), or path style.
    s3_host = (
        re.fullmatch(r"(?:s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com(?:\.cn)?)", hostname) is not None
    )
    bucket_host = (
        hostname.startswith(bucket + ".")
        and re.fullmatch(
            r"s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com(?:\.cn)?", hostname[len(bucket) + 1 :]
        )
        is not None
    )
    path = unquote(parts.path)
    target_matches = (bucket_host and path == "/" + key) or (s3_host and path == f"/{bucket}/{key}")
    allowed = bool(
        parts.scheme == "https"
        and not parts.username
        and not parts.password
        and parts.port in (None, 443)
        and target_matches
        and not parts.fragment
    )
    info = presigned_url_metadata(url)
    info.update(
        https_s3_exact_bucket_key=allowed,
        configured_regional_host=hostname == virtual,
        global_s3_host=hostname == global_host,
        signing_region_matches_config=info["signing_region"] == region,
    )
    if not allowed:
        info["status"] = "failed"
    return info


def standalone_url_info(url):
    """Validate the S3 object endpoint without configured bucket/region or AWS SDKs."""
    invalid = {"status": "failed", "url_format_valid": False, "https_s3_object_url": False}
    try:
        # urlsplit/urllib can silently strip raw control characters. Reject them first.
        if not url or any(ord(char) <= 32 or ord(char) == 127 for char in url):
            return invalid, ""
        parts = urlsplit(url)
        host = S3_OBJECT_HOST.fullmatch(parts.hostname or "")
        format_valid = bool(
            parts.scheme == "https"
            and parts.hostname
            and not parts.username
            and not parts.password
            and parts.port in (None, 443)
            and not parts.fragment
        )
        path = unquote(parts.path).lstrip("/")
        bucket = host.group("bucket") if host else None
        if host and not bucket:
            bucket, _, path = path.partition("/")
        has_object = bool(bucket and re.fullmatch(BUCKET_PATTERN, bucket) and path)
        allowed = bool(format_valid and host and has_object)
        info = presigned_url_metadata(url)
        info.update(
            url_format_valid=format_valid,
            s3_host_valid=host is not None,
            https_s3_object_url=allowed,
            addressing_style=("virtual" if host.group("bucket") else "path") if host else None,
            endpoint_region=(host.group("region") or host.group("dual_region")) if host else None,
        )
        if not allowed:
            info["status"] = "failed"
        return info, path
    except ValueError:
        return invalid, ""


def read_hidden_original_url():
    # Fail closed if the terminal cannot disable echo; never fall back to readline.
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass("Original URL (hidden): ").strip()


def run_original_url_only(args):
    url = read_hidden_original_url()
    info, key = standalone_url_info(url)
    report = {
        "mode": "original_url_only",
        "checked_at_utc": iso(utcnow()),
        "runware_called": False,
        "original_url": info,
        "historical_original_access": "not_proven_by_current_probe",
        "current_403_does_not_prove_original_runware_failure": True,
    }
    if args.offline:
        report["original_http_get"] = {"status": "skipped", "reason": "offline"}
    elif not info["https_s3_object_url"]:
        report["original_http_get"] = {"status": "skipped", "reason": "invalid_s3_object_url"}
    else:
        report["original_http_get"] = probe_get(
            url,
            timeout=args.timeout,
            max_bytes=args.max_bytes,
            key=key,
        )
    return report


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def decode_image(body, key=DEFAULT_KEY):
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(body)) as image:
                fmt, size, frames = image.format, image.size, getattr(image, "n_frames", 1)
                image.verify()
            # Reopen after verify, then load every frame to detect truncated image data.
            with Image.open(BytesIO(body)) as image:
                if frames > 100:
                    return {"status": "unknown", "reason": "over_100_frames_decode_limit"}
                for frame in range(frames):
                    image.seek(frame)
                    image.load()
        return {
            "status": "ok",
            "format": fmt if fmt in {"WEBP", "JPEG", "PNG"} else "other",
            "width": size[0],
            "height": size[1],
            "frames": frames,
            "extension_matches_bytes": Path(key).suffix.lower()
            in {
                "WEBP": {".webp"},
                "JPEG": {".jpg", ".jpeg"},
                "PNG": {".png"},
            }.get(fmt, set()),
        }
    except Exception as error:
        return failure(error)


def probe_get(url, *, timeout, max_bytes, key=DEFAULT_KEY):
    """GET exactly as signed; no Authorization, HEAD, range or Runware call."""
    report = {"method": "GET", "checked_at_utc": iso(utcnow()), "redirects_followed": False}
    response = None
    try:
        request = Request(url, method="GET", headers={"User-Agent": "LookDDak-S3-Diagnostic/1.0"})
        try:
            response = build_opener(NoRedirect()).open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            status, headers = response.code, response.headers
            report.update(
                status="ok" if status == 200 else "failed",
                http_status=status,
                content_type=safe_mime(headers.get("Content-Type")),
                redirect=300 <= status < 400,
                redirect_location_present=bool(headers.get("Location")),
                bucket_region_header=safe_region(headers.get("x-amz-bucket-region")),
            )
            length = headers.get("Content-Length", "")
            report["content_length_header"] = int(length) if length.isdigit() else None
            # Error documents may contain AccessKeyId, Signature, CanonicalRequest, etc.
            # Parse only known codes and a boolean expiry message; never output the XML.
            body = response.read(max_bytes + 1 if status == 200 else 65536)
            report["bytes_read"] = len(body)
        if status != 200:
            try:
                root = ElementTree.fromstring(body)
                code = root.findtext("Code")
                report["s3_error_code"] = code if code in S3_CODES else "other_or_missing"
                report["s3_message_request_expired"] = (
                    root.findtext("Message", "").strip().lower() == "request has expired"
                )
            except ElementTree.ParseError:
                report["s3_error_code"] = "not_xml_or_incomplete"
            report["image_decode"] = {"status": "skipped", "reason": "http_not_200"}
            return report
        if len(body) > max_bytes:
            report.update(
                status="unknown",
                image_decode={
                    "status": "skipped",
                    "reason": "download_limit_exceeded",
                    "max_bytes": max_bytes,
                },
            )
            return report
        report["content_length_matches"] = report["content_length_header"] in (None, len(body))
        report["image_decode"] = decode_image(body, key)
        decoded = report["image_decode"]
        expected_mime = {"WEBP": "image/webp", "JPEG": "image/jpeg", "PNG": "image/png"}.get(
            decoded.get("format")
        )
        report["mime_matches_bytes"] = (
            report["content_type"] == expected_mime if expected_mime else None
        )
        if decoded["status"] != "ok":
            report["status"] = decoded["status"]
        elif not all(
            (
                report["content_length_matches"],
                report["mime_matches_bytes"],
                decoded["extension_matches_bytes"],
            )
        ):
            report["status"] = "failed"
        return report
    except Exception as error:
        report.update(failure(error))
        return report
    finally:
        if response is not None:
            response.close()


def region_from_response(response):
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    return safe_region(headers.get("x-amz-bucket-region"))


def aws_checks(report, args, bucket, region):
    import boto3
    from botocore.config import Config

    from app.clients.s3 import S3ImageStorage, get_s3_presign_client
    from app.virtual_fitting.async_storage import user_image_get_url

    try:
        # Exactly the production signing client, default session and credential chain.
        presigner = get_s3_presign_client()
        session = boto3.DEFAULT_SESSION
        credentials = session.get_credentials()
        if credentials is None:
            from botocore.exceptions import NoCredentialsError

            raise NoCredentialsError()
        credentials.get_frozen_credentials()  # Resolves/refreshes; never print these values.
        report["aws_credentials"] = {"status": "ok", "resolved": True}
        # This is SDK-internal metadata, not available for every credential provider.
        expiry = getattr(credentials, "_expiry_time", None)
        report["aws_credentials"]["expiry_at_utc_if_available"] = iso(expiry)
    except Exception as error:
        report["aws_credentials"] = failure(error)
        for name in (
            "aws_authentication",
            "bucket_region",
            "object_head",
            "new_url",
            "new_http_get",
        ):
            report[name] = {"status": "skipped", "reason": "credentials_unavailable"}
        return

    config = Config(
        connect_timeout=args.timeout, read_timeout=args.timeout, retries={"max_attempts": 0}
    )
    try:
        # Auth validation only: do not log Account, Arn, UserId or raw response.
        session.client("sts", region_name=region, config=config).get_caller_identity()
        report["aws_authentication"] = {"status": "ok", "sts_identity_verified": True}
    except Exception as error:
        report["aws_authentication"] = failure(error)

    # Bounded timeout/retries for diagnosis; same bucket/region and default identity.
    s3 = session.client("s3", region_name=region, config=config)
    try:
        response = s3.head_bucket(Bucket=bucket)
        actual_region = region_from_response(response)
        report["bucket_region"] = {"status": "unknown"}
    except Exception as error:
        report["bucket_region"] = failure(error)
        actual_region = region_from_response(getattr(error, "response", {}) or {})
    if actual_region:
        report["bucket_region"].update(
            status="ok" if actual_region == region else "failed",
            observed_region=actual_region,
            matches_config=actual_region == region,
        )
    try:
        response = s3.head_object(Bucket=bucket, Key=args.key)
        report["object_head"] = {
            "status": "ok",
            "exists": True,
            "http_status": response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
            "content_type": safe_mime(response.get("ContentType")),
            "content_length": response.get("ContentLength"),
            "last_modified_utc": iso(response.get("LastModified")),
            "sse_kms": response.get("ServerSideEncryption") == "aws:kms",
        }
    except Exception as error:
        report["object_head"] = failure(error)
        status = report["object_head"].get("http_status")
        report["object_head"]["exists"] = False if status == 404 else None
        if status == 403:
            report["object_head"]["reason"] = "forbidden_or_missing_without_ListBucket"

    try:
        storage = S3ImageStorage(s3, bucket=bucket, presign_client=presigner)
        url = user_image_get_url(storage, args.key)
        info = url_info(url, bucket, args.key, region)
        info["production_operation"] = "get_object"
        info["generation_succeeded"] = True
        if not all(
            (
                info["sigv4"],
                info["has_signature"],
                info["configured_regional_host"],
                info["signing_region_matches_config"],
            )
        ):
            info["status"] = "failed"
        report["new_url"] = info
        if info["https_s3_exact_bucket_key"]:
            report["new_http_get"] = probe_get(
                url,
                timeout=args.timeout,
                max_bytes=args.max_bytes,
                key=args.key,
            )
        else:
            report["new_http_get"] = {"status": "skipped", "reason": "unexpected_s3_target"}
    except Exception as error:
        report["new_url"] = failure(error)
        report["new_http_get"] = {"status": "skipped", "reason": "presign_failed"}


def run(args):
    # Keep this branch BEFORE env-file loading and all service/AWS imports.
    if args.original_url_only:
        return run_original_url_only(args)
    load_settings(args.env_file)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.clients.s3 import get_s3_bucket, validate_s3_key
    from app.config import settings

    args.key = validate_s3_key(args.key)
    bucket, region = get_s3_bucket(), settings.AWS_REGION
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or not safe_region(region):
        raise ValueError("Invalid bucket/region")
    if not 1 <= settings.VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS <= 604800:
        raise ValueError("Invalid presigned URL lifetime")
    report = {
        "checked_at_utc": iso(utcnow()),
        "bucket": bucket,
        "configured_region": region,
        "key": args.key,
        "configured_expires_seconds": settings.VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS,
        "offline": args.offline,
        "runware_called": False,
    }
    if args.original_url_file or args.original_url_prompt:
        if args.original_url_file:
            url = args.original_url_file.read_text().strip()
        else:
            url = read_hidden_original_url()
        report["original_url"] = url_info(url, bucket, args.key, region)
        if args.offline:
            report["original_http_get"] = {"status": "skipped", "reason": "offline"}
        elif report["original_url"]["https_s3_exact_bucket_key"]:
            report["original_http_get"] = probe_get(
                url,
                timeout=args.timeout,
                max_bytes=args.max_bytes,
                key=args.key,
            )
        else:
            report["original_http_get"] = {"status": "skipped", "reason": "unexpected_s3_target"}
    else:
        report["original_url"] = {"status": "skipped", "reason": "not_supplied"}
    if args.offline:
        for name in (
            "aws_credentials",
            "aws_authentication",
            "bucket_region",
            "object_head",
            "new_url",
            "new_http_get",
        ):
            report[name] = {"status": "skipped", "reason": "offline"}
    else:
        try:
            aws_checks(report, args, bucket, region)
        except Exception as error:
            # Preserve completed checks, particularly the independent old URL probe.
            report["aws_diagnostic"] = failure(error)
            for name in (
                "aws_credentials",
                "aws_authentication",
                "bucket_region",
                "object_head",
                "new_url",
                "new_http_get",
            ):
                report.setdefault(name, {"status": "skipped", "reason": "aws_diagnostic_failed"})
    original, new = report["original_url"], report["new_url"]
    if "url_fingerprint_sha256_16" in original and "url_fingerprint_sha256_16" in new:
        report["comparison"] = {
            "identical_url_fingerprint": original["url_fingerprint_sha256_16"]
            == new["url_fingerprint_sha256_16"],
            "original_nominal_expired": original.get("nominal_expired"),
            "new_nominal_expired": new.get("nominal_expired"),
            "historical_original_access": "not_proven_by_current_probe",
        }
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    old = parser.add_mutually_exclusive_group()
    old.add_argument("--original-url-file", type=Path)
    old.add_argument("--original-url-prompt", action="store_true")
    old.add_argument(
        "--original-url-only",
        action="store_true",
        help="Probe only a hidden-input existing S3 URL; no AWS credentials or settings required.",
    )
    args = parser.parse_args()
    # Includes SDK debug logging enabled by a user's environment/import hooks.
    logging.disable(logging.CRITICAL)
    try:
        if (
            not math.isfinite(args.timeout)
            or not 0 < args.timeout <= 60
            or not 0 < args.max_bytes <= 100 * 1024 * 1024
        ):
            raise ValueError("Invalid timeout/download limit")
        report = run(args)
    except KeyboardInterrupt:
        report = {"diagnostic": {"status": "unknown", "reason": "interrupted"}}
    except Exception as error:
        report = {"diagnostic": failure(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    states = [value.get("status") for value in report.values() if isinstance(value, dict)]
    if "failed" in states:
        return 1
    incomplete = any(
        value.get("status") in {"unknown", "skipped"}
        for name, value in report.items()
        if isinstance(value, dict)
        and not (name == "original_url" and value.get("reason") == "not_supplied")
    )
    return 2 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
