import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def setup_logger(name: str = "app") -> logging.Logger:
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def _validate_jpeg(source):
    """Fully decode a JPEG to reject login pages, corrupt files, and truncated transfers."""
    with Image.open(source) as image:
        if image.format != "JPEG":
            raise ValueError("Expected a JPEG image")
        image.load()


def _fetch_cxr(aws, source, region, profile=None):
    """Fetch one S3 image using local AWS credentials without exposing image paths or raw errors."""
    connect_timeout, read_timeout = config.CXR_DOWNLOAD_TIMEOUT_SECONDS
    command = [aws, "s3", "cp", source, "-", "--region", region,
               "--only-show-errors", "--no-cli-pager", "--no-progress",
               "--cli-connect-timeout", str(connect_timeout),
               "--cli-read-timeout", str(read_timeout)]
    if profile:
        command.extend(["--profile", profile])
    started = perf_counter()
    result = subprocess.run(command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=dict(os.environ, AWS_CLI_AUTO_PROMPT="off"),
                            timeout=config.CXR_DOWNLOAD_TOTAL_TIMEOUT_SECONDS, check=False)
    seconds = perf_counter() - started
    # AWS errors may contain patient object keys; retain only a bounded error code.
    match = re.search(rb"An error occurred \(([A-Za-z0-9_-]{1,64})\)", result.stderr)
    error_code = match.group(1).decode("ascii") if match else None
    return result.returncode, error_code, result.stdout, seconds


def download_cohort_cxrs(
    parquet_path: Path,
    output_dir: Path,
    sample_size: int | None = config.CXR_DOWNLOAD_SAMPLE_SIZE,
    profile: str | None = None,
    *,
    source_prefix: str,
    region: str,
):
    """Download a bounded set of cohort JPEGs and return aggregate outcomes and transfer speed.

    sample_size limits unique images; None selects all images referenced by the input artifact.
    """
    logger = setup_logger("CXR-Download")
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive")
    source_prefix = source_prefix.strip().rstrip("/")
    region = region.strip()
    if not source_prefix.startswith("s3://") or not region:
        raise ValueError("An S3 source prefix and AWS region are required")
    df = pd.read_parquet(parquet_path, columns=["cxr_path"])
    image_paths = df["cxr_path"].dropna().unique().tolist()
    if sample_size is not None:
        image_paths = image_paths[:sample_size]
    summary = dict(selected=len(image_paths), downloaded=0, skipped=0, failed=0,
                   not_attempted=0, bytes_downloaded=0, transfer_seconds=0.0)
    root = output_dir.resolve()
    logger.info("Selected %d unique images", len(image_paths))

    aws = None
    for index, rel_path in enumerate(tqdm(image_paths, desc="CXR attempts"), start=1):
        try:
            relative = Path(rel_path)
            local_file = (root / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts
                    or not local_file.is_relative_to(root) or relative.suffix.lower() != ".jpg"):
                raise ValueError("Invalid relative image path")
        except (TypeError, ValueError, OSError, RuntimeError):
            summary["failed"] += 1
            logger.error("Image %d: invalid relative image path", index)
            continue

        if local_file.is_file():
            try:
                _validate_jpeg(local_file)
            except (OSError, ValueError, Image.DecompressionBombError):
                logger.warning("Image %d: existing file is invalid; downloading replacement", index)
            else:
                summary["skipped"] += 1
                continue

        if aws is None:
            aws = shutil.which("aws")
            if aws is None:
                summary["failed"] += 1
                logger.error("AWS CLI is required; install it and configure credentials. Stopping.")
                break
        source = f"{source_prefix}/{relative.as_posix()}"
        try:
            code, error_code, content, transfer_seconds = _fetch_cxr(aws, source, region, profile)
        except (subprocess.TimeoutExpired, OSError) as exc:
            summary["failed"] += 1
            logger.error("Image %d: %s; check AWS CLI availability and connection", index, type(exc).__name__)
            continue
        if code != 0:
            summary["failed"] += 1
            logger.error("Image %d: AWS exit %d (%s); check credentials, dataset access, and connection",
                         index, code, error_code or "transfer failed")
            continue

        try:
            _validate_jpeg(BytesIO(content))
        except (OSError, ValueError, Image.DecompressionBombError):
            summary["failed"] += 1
            logger.error("Image %d: downloaded content is an invalid JPEG; nothing saved", index)
            continue

        tmp_file = local_file.with_suffix(".jpg.part")
        try:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp_file.write_bytes(content)
                tmp_file.replace(local_file)
            finally:
                tmp_file.unlink(missing_ok=True)
        except OSError as exc:
            summary["failed"] += 1
            logger.error("Image %d: %s while saving; check disk space and permissions", index, type(exc).__name__)
            continue
        summary["downloaded"] += 1
        summary["bytes_downloaded"] += len(content)
        summary["transfer_seconds"] += transfer_seconds

    summary["not_attempted"] = summary["selected"] - sum(summary[k] for k in ("downloaded", "skipped", "failed"))
    seconds = summary["transfer_seconds"]
    mib = summary["bytes_downloaded"] / (1024 * 1024)
    summary["mib_per_second"] = mib / seconds if seconds else 0.0
    logger.info("Downloaded: %d; already valid: %d; failed: %d; not attempted: %d",
                summary["downloaded"], summary["skipped"], summary["failed"], summary["not_attempted"])
    if summary["downloaded"]:
        logger.info("Successful transfers: %.2f MiB in %.2f s (%.2f MiB/s)", mib, seconds, summary["mib_per_second"])
    return summary


def main():
    """Expose a one-image access check before an explicit cohort download."""
    parser = argparse.ArgumentParser(description="Download and validate cohort CXR images from S3 using AWS CLI.")
    parser.add_argument("--input", type=Path, default=config.PATIENT_WINDOWS_PARQUET)
    parser.add_argument("--output_dir", type=Path, default=config.MIMIC_CXR_DIR)
    parser.add_argument("--profile", help="Optional AWS profile; otherwise use the configured AWS credentials")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample_size", type=int, default=config.CXR_DOWNLOAD_SAMPLE_SIZE,
                           help="Maximum number of unique images (default: %(default)s)")
    selection.add_argument("--all", action="store_true", help="Download all referenced images after a successful sample check")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample_size must be positive")
    try:
        source_prefix = input("S3 source prefix (s3://... ending in /files/): ").strip()
        region = input("AWS region: ").strip()
        summary = download_cohort_cxrs(args.input, args.output_dir,
                                       None if args.all else args.sample_size, args.profile,
                                       source_prefix=source_prefix, region=region)
    except KeyboardInterrupt:
        setup_logger("CXR-Download").error("Download interrupted")
        return 130
    except Exception as exc:
        setup_logger("CXR-Download").error("Downloader failed (%s); check the input artifact and local access", type(exc).__name__)
        return 1
    return int(bool(summary["failed"] or summary["not_attempted"]))


if __name__ == "__main__":
    sys.exit(main())
