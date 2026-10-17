import getpass
import logging
import sys
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm


def setup_logger(name: str = "app") -> logging.Logger:
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def download_cohort_cxrs(parquet_path: Path, output_dir: Path):
    """Downloads only the specific chest X-ray images referenced in patient_windows.parquet."""
    auth = (input("PhysioNet User: "), getpass.getpass("PhysioNet Password: "))

    df = pd.read_parquet(parquet_path)
    image_paths = df["cxr_path"].dropna().unique()

    base_url = "https://physionet.org/files/mimic-cxr-jpg/2.0.0"
    with requests.Session() as session:
        session.auth = auth
        for rel_path in tqdm(image_paths, desc="Downloading CXRs"):
            local_file = output_dir / rel_path
            if local_file.exists():
                continue

            local_file.parent.mkdir(parents=True, exist_ok=True)
            r = session.get(f"{base_url}/{rel_path}")
            if r.status_code == 200:
                tmp_file = local_file.with_suffix(".tmp")
                tmp_file.write_bytes(r.content)
                tmp_file.replace(local_file)
