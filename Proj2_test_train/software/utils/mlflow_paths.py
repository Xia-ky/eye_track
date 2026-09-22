"""Path helpers for ESDA's local MLflow artifact store."""

from pathlib import Path
from urllib.parse import unquote, urlparse


def local_artifact_path(artifact_uri: str) -> Path:
    """Convert a local path or file URI to a filesystem path."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "":
        return Path(artifact_uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError(
            f"ESDA requires a local file MLflow artifact URI: {artifact_uri}"
        )
    return Path(unquote(parsed.path))

