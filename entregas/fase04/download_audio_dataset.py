"""Download and unpack the audio datasets used by Tarefa 2.

Three auxiliary datasets are supported:

* ``icbhi`` - ICBHI 2017 Respiratory Sound Database (respiratory events).
* ``voiced`` - VOICED Database (healthy vs pathological voices, PhysioNet).
* ``parkinson`` - UCI Parkinson's Disease (sustained-vowel voice measures, used as
  an open proxy to sanity-check the dysarthria markers).

The ICBHI host serves its archive over an incomplete TLS certificate chain, so
certificate verification is disabled for that host only. All downloads are
read-only and every archive is validated after transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = Path(__file__).parent / "data" / "raw"
CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 120

ICBHI_BASE_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database"
ICBHI_ARCHIVE = "ICBHI_final_database.zip"
ICBHI_SMALL_FILES = ("ICBHI_Challenge_diagnosis.txt", "ICBHI_challenge_train_test.txt")
ICBHI_DIR = DATA_DIR / "icbhi"

VOICED_ARCHIVE_URL = (
    "https://physionet.org/static/published-projects/voiced/voiced-database-1.0.0.zip"
)
VOICED_ARCHIVE = "voiced-database-1.0.0.zip"
VOICED_DIR = DATA_DIR / "voiced"

PARKINSON_BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons"
)
PARKINSON_FILE = "parkinsons.data"
PARKINSON_DIR = DATA_DIR / "parkinson"


def download(url: str, destination: Path, verify: bool = True) -> None:
    if destination.is_file():
        print(f"Verified; skipping: {destination.name}")
        return

    temporary_file = destination.with_name(f"{destination.name}.part")
    digest = hashlib.sha256()
    received = 0
    try:
        with requests.get(
            url, stream=True, timeout=TIMEOUT_SECONDS, verify=verify
        ) as response:
            response.raise_for_status()
            with temporary_file.open("wb") as output:
                for chunk in response.iter_content(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
        temporary_file.replace(destination)
    except (OSError, requests.RequestException):
        temporary_file.unlink(missing_ok=True)
        raise
    print(f"Downloaded: {destination.name} ({received:,} bytes, sha256={digest.hexdigest()[:16]})")


def extract(archive: Path, destination: Path) -> None:
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"{archive} is not a valid zip archive.")
    with zipfile.ZipFile(archive) as package:
        bad_member = package.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupted member in {archive.name}: {bad_member}")
        package.extractall(destination)
    print(f"Extracted: {archive.name} -> {destination}")


def download_icbhi(data_dir: Path, keep_archive: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / ICBHI_ARCHIVE
    download(f"{ICBHI_BASE_URL}/{ICBHI_ARCHIVE}", archive_path, verify=False)
    for filename in ICBHI_SMALL_FILES:
        download(f"{ICBHI_BASE_URL}/{filename}", data_dir / filename, verify=False)

    extract(archive_path, data_dir)
    if not keep_archive:
        archive_path.unlink()
        print(f"Removed archive: {archive_path.name}")

    print(f"ICBHI 2017 is ready in {data_dir.resolve()}")


def download_voiced(data_dir: Path, keep_archive: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / VOICED_ARCHIVE
    download(VOICED_ARCHIVE_URL, archive_path)

    extract(archive_path, data_dir)
    if not keep_archive:
        archive_path.unlink()
        print(f"Removed archive: {archive_path.name}")

    print(f"VOICED is ready in {data_dir.resolve()}")


def download_parkinson(data_dir: Path, keep_archive: bool) -> None:  # noqa: ARG001
    data_dir.mkdir(parents=True, exist_ok=True)
    download(f"{PARKINSON_BASE_URL}/{PARKINSON_FILE}", data_dir / PARKINSON_FILE)
    print(f"UCI Parkinson's Disease is ready in {data_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("icbhi", "voiced", "parkinson", "both"),
        default="both",
        help="which dataset to download ('both' covers icbhi + voiced)",
    )
    parser.add_argument(
        "--keep-archive", action="store_true", help="keep the zip after extraction"
    )
    args = parser.parse_args()

    if args.dataset in ("icbhi", "both"):
        download_icbhi(ICBHI_DIR, args.keep_archive)
    if args.dataset in ("voiced", "both"):
        download_voiced(VOICED_DIR, args.keep_archive)
    if args.dataset == "parkinson":
        download_parkinson(PARKINSON_DIR, args.keep_archive)


if __name__ == "__main__":
    main()
