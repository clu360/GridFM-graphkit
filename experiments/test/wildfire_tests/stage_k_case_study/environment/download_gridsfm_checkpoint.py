"""Download the frozen GridSFM checkpoint once and verify its identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import hf_hub_download


HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "GRIDSFM_ENVIRONMENT_CONTRACT.json").read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(target: Path, observed_sha256: str) -> None:
    payload = {
        "repository": CONTRACT["checkpoint"]["repository"],
        "filename": CONTRACT["checkpoint"]["filename"],
        "revision": CONTRACT["checkpoint"]["revision"],
        "sha256": observed_sha256,
    }
    sidecar = target.with_suffix(target.suffix + ".provenance.json")
    fd, temporary = tempfile.mkstemp(prefix=sidecar.name + ".", dir=sidecar.parent)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, sidecar)
    sidecar.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    spec = CONTRACT["checkpoint"]
    destination = Path(args.checkpoint_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / spec["filename"]

    if target.is_file() and sha256_file(target) == spec["sha256"]:
        write_provenance(target, spec["sha256"])
        target.chmod(0o444)
        print(target)
        return 0

    downloaded = Path(
        hf_hub_download(
            repo_id=spec["repository"],
            filename=spec["filename"],
            revision=spec["revision"],
            local_dir=destination,
        )
    ).resolve()
    observed = sha256_file(downloaded)
    if observed != spec["sha256"]:
        downloaded.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {observed} != {spec['sha256']}")
    write_provenance(downloaded, observed)
    downloaded.chmod(0o444)
    print(downloaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
