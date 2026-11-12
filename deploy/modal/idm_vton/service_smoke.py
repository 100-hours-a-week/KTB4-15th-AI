"""배포된 lookddak-idm-vton-fallback / generate 를 이름으로 찾아 1회 부르는 독립 테스트(Mac 에서 실행).

AI Server 와 같은 방식으로 요청을 만든다: 결과 Key 에 Content-Type(image/webp)·x-amz-meta-fitting-job-id·attempt-id 를 서명한
Presigned PUT, spawn 1회(재시도 없음), FunctionCall.get 으로 대기, S3 HEAD·GET 으로 WebP·메타데이터·sha256 확인.
입력은 기본으로 inference_check 와 같은 공식 예제(고정 커밋)의 공개 URL 이다. URL·서명은 출력하지 않는다.

    cd deploy/modal/idm_vton
    MODAL_PROFILE=lookddak AWS_PROFILE=<S3 PutObject 권한 프로필> ../../../.venv/bin/python service_smoke.py \
        --bucket <버킷> --key-prefix virtual-fitting/idm-smoke [--person-url ... --garment-url ... --description ...]

결과 객체는 <key-prefix>/<실행 시각>/idm-vton.webp 하나만 만든다(지우지 않는다). --save 로 로컬에도 저장한다.
"""

import argparse
import hashlib
import io
import time
import uuid

from detectron2_build_check import SOURCE_COMMIT

RAW = f"https://raw.githubusercontent.com/yisol/IDM-VTON/{SOURCE_COMMIT}/gradio_demo/example"
DEFAULT_PERSON_URL = f"{RAW}/human/00034_00.jpg"
DEFAULT_GARMENT_URL = f"{RAW}/cloth/04469_00.jpg"
DEFAULT_DESCRIPTION = "teal blue velvet short sleeve v-neck wrap top"
UPLOAD_EXPIRES = 1800


def main() -> None:
    import boto3
    import modal
    from botocore.config import Config
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key-prefix", default="virtual-fitting/idm-smoke")
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--person-url", default=DEFAULT_PERSON_URL)
    parser.add_argument("--garment-url", default=DEFAULT_GARMENT_URL)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--job-id", type=int, default=990000001)
    parser.add_argument("--save", default="")
    # 이미 spawn 한 호출에 다시 붙어 기다리기만 한다(다시 부르지 않음). --key·--attempt-id 를 함께 준다.
    parser.add_argument("--call-id", default="")
    parser.add_argument("--key", default="")
    parser.add_argument("--attempt-id", default="")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    key = args.key or f"{args.key_prefix.rstrip('/')}/{stamp}/idm-vton.webp"
    attempt = args.attempt_id or str(uuid.uuid4())
    metadata = {"fitting-job-id": str(args.job_id), "attempt-id": attempt}
    s3 = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": args.bucket,
            "Key": key,
            "ContentType": "image/webp",
            "Metadata": metadata,
        },
        ExpiresIn=UPLOAD_EXPIRES,
    )
    request = {
        "fitting_job_id": args.job_id,
        "attempt_id": attempt,
        "person_image_url": args.person_url,
        "garment_image_url": args.garment_url,
        "garment_description": args.description,
        "upload_url": upload_url,
        "upload_headers": {
            "Content-Type": "image/webp",
            **{f"x-amz-meta-{k}": v for k, v in metadata.items()},
        },
        "not_after_epoch": time.time() + UPLOAD_EXPIRES - 60,
    }
    started = time.perf_counter()
    if args.call_id:
        call = modal.FunctionCall.from_id(args.call_id)
        print(f"attached call_id={args.call_id} key={key}")
    else:
        function = modal.Function.from_name("lookddak-idm-vton-fallback", "generate")
        call = function.spawn(request)  # 1회. 재시도하지 않는다.
        print(f"spawned call_id={call.object_id} key={key} attempt_id={attempt}")
    while True:
        try:
            output = call.get(timeout=15)
            break
        except modal.exception.FunctionTimeoutError:
            raise
        except (modal.exception.TimeoutError, TimeoutError):
            # get(timeout) 의 대기 초과는 내장 TimeoutError 다(Modal 1.6.1 실측). 함수는 계속 돈다.
            print(f"  running... {time.perf_counter() - started:.0f}s", flush=True)
    wall = time.perf_counter() - started
    print(f"wall={wall:.1f}s output={ {k: v for k, v in output.items() if k != 'detail'} }")
    if output.get("status") != "succeeded":
        raise SystemExit(
            f"FAILED error_kind={output.get('error_kind')} detail={output.get('detail')}"
        )
    head = s3.head_object(Bucket=args.bucket, Key=key)
    body = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
    with Image.open(io.BytesIO(body)) as image:
        image.load()
        info = (image.format, image.mode, image.size)
    checks = {
        "content_type": head.get("ContentType") == "image/webp",
        # 붙기 모드에서는 시도 ID 를 함수 응답에서 받는다(응답·메타데이터·요청이 같은 시도인지).
        "metadata": head.get("Metadata")
        == {"fitting-job-id": str(args.job_id), "attempt-id": output["attempt_id"]},
        "sha256": hashlib.sha256(body).hexdigest() == output["sha256"],
        "bytes": len(body) == output["bytes"],
        "webp": info[0] == "WEBP",
    }
    print(f"s3 object: {info} {len(body):,} bytes checks={checks}")
    if args.save:
        with open(args.save, "wb") as file:
            file.write(body)
        print(f"saved {args.save}")
    if not all(checks.values()):
        raise SystemExit("S3 결과 확인 실패")


if __name__ == "__main__":
    main()
