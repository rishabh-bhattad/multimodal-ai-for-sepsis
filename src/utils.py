import argparse
import getpass
import logging
import os
import re
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

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


def _fetch_cxr(wget, url, auth):
    """Use genuine Wget for PhysioNet's permitted HTTP-auth client and keep credentials in a pipe.

    Return the process code, final HTTP status, body bytes, and transfer duration; raw logs stay private.
    """
    if any(character in auth[1] for character in ("\r", "\n")):
        raise ValueError("Password must be a single line")
    connect_timeout, read_timeout = config.CXR_DOWNLOAD_TIMEOUT_SECONDS
    command = [wget, "--no-config", "--no-netrc", "--no-hsts", "--server-response",
               "--ask-password", f"--user={auth[0]}",
               "--tries=1", f"--dns-timeout={connect_timeout}",
               f"--connect-timeout={connect_timeout}", f"--read-timeout={read_timeout}",
               "--max-redirect=0", "--output-document=-", "--", url]
    env = dict(os.environ, LC_ALL="C")
    env.pop("WGET_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    started = perf_counter()
    # A detached child reads --ask-password from stdin instead of the user's terminal.
    result = subprocess.run(command, input=(auth[1] + "\n").encode("utf-8"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                            start_new_session=True,
                            timeout=config.CXR_DOWNLOAD_TOTAL_TIMEOUT_SECONDS, check=False)
    seconds = perf_counter() - started
    # Wget's logs can contain patient URLs and cookies; retain only numeric HTTP statuses.
    statuses = re.findall(rb"^\s*HTTP/\S+\s+(\d{3})\b", result.stderr, re.MULTILINE)
    status = int(statuses[-1]) if statuses else None
    return result.returncode, status, result.stdout, seconds


def download_cohort_cxrs(
    parquet_path: Path,
    output_dir: Path,
    sample_size: int | None = config.CXR_DOWNLOAD_SAMPLE_SIZE,
):
    """Download a bounded set of cohort JPEGs and return aggregate outcomes and transfer speed.

    sample_size limits unique images; None selects all images referenced by the input artifact.
    """
    logger = setup_logger("CXR-Download")
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive")
    df = pd.read_parquet(parquet_path, columns=["cxr_path"])
    image_paths = df["cxr_path"].dropna().unique().tolist()
    if sample_size is not None:
        image_paths = image_paths[:sample_size]
    summary = dict(selected=len(image_paths), downloaded=0, skipped=0, failed=0,
                   not_attempted=0, bytes_downloaded=0, transfer_seconds=0.0)
    root = output_dir.resolve()
    logger.info("Selected %d unique images", len(image_paths))

    auth = None
    wget = None
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

        if wget is None:
            wget = shutil.which(config.CXR_DOWNLOAD_COMMAND)
            if wget is None:
                summary["failed"] += 1
                logger.error("GNU Wget is required for PhysioNet authentication; install it and retry. Stopping.")
                break
        if auth is None:
            auth = (input("PhysioNet User: ").strip(), getpass.getpass("PhysioNet Password: "))
        url = f"{config.CXR_DOWNLOAD_BASE_URL.rstrip('/')}/{quote(relative.as_posix(), safe='/')}"
        try:
            code, status, content, transfer_seconds = _fetch_cxr(wget, url, auth)
        except (subprocess.TimeoutExpired, OSError) as exc:
            summary["failed"] += 1
            logger.error("Image %d: %s; check Wget availability and connection", index, type(exc).__name__)
            continue
        if status != 200 or code != 0:
            summary["failed"] += 1
            if status in (401, 403) or (status is not None and 300 <= status < 400):
                logger.error("Image %d: HTTP %d; access refused or redirected. Check account/project access. Stopping.", index, status)
                break
            if status is not None and status != 200:
                hint = "check dataset version and remote files/ path" if status == 404 else "server request failed; retry later"
                logger.error("Image %d: HTTP %d; %s", index, status, hint)
            else:
                hint = {4: "network failure or timeout", 5: "TLS certificate verification failed",
                        6: "authentication failed", 3: "local I/O failure"}.get(code, "transfer failed")
                logger.error("Image %d: Wget exit %d (%s); nothing saved", index, code, hint)
            continue

        try:
            _validate_jpeg(BytesIO(content))
        except (OSError, ValueError, Image.DecompressionBombError):
            summary["failed"] += 1
            logger.error("Image %d: HTTP 200 contained an invalid JPEG; nothing saved", index)
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
    parser = argparse.ArgumentParser(description="Download and validate cohort CXR images from PhysioNet.")
    parser.add_argument("--input", type=Path, default=config.PATIENT_WINDOWS_PARQUET)
    parser.add_argument("--output_dir", type=Path, default=config.MIMIC_CXR_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample_size", type=int, default=config.CXR_DOWNLOAD_SAMPLE_SIZE,
                           help="Maximum number of unique images (default: %(default)s)")
    selection.add_argument("--all", action="store_true", help="Download all referenced images after a successful sample check")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample_size must be positive")
    try:
        summary = download_cohort_cxrs(args.input, args.output_dir, None if args.all else args.sample_size)
    except KeyboardInterrupt:
        setup_logger("CXR-Download").error("Download interrupted")
        return 130
    except Exception as exc:
        setup_logger("CXR-Download").error("Downloader failed (%s); check the input artifact and local access", type(exc).__name__)
        return 1
    return int(bool(summary["failed"] or summary["not_attempted"]))


if __name__ == "__main__":
    sys.exit(main())
