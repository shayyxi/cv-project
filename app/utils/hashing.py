from pathlib import Path
import hashlib


def sha256_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hash_obj = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()

def sha256_bytes(data: bytes) -> str:
    """
    Computes the SHA-256 hash of bytes.
    """
    return hashlib.sha256(data).hexdigest()