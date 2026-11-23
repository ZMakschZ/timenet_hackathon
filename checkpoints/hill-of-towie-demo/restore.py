"""Verify the Git LFS parts or restore the selected Hill of Towie demo checkpoint."""

import argparse
import hashlib
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path


def main():
    """Verify the checkpoint hash and optionally write the assembled weights."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Fetch the Git LFS objects before running this script. Omit output to verify only.",
    )
    parser.add_argument("output", nargs="?", type=Path, help="Path for the restored trained.safetensors")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    expected = (source / "trained.safetensors.sha256").read_text().split()[0]
    parts = [source / f"trained.safetensors.part-{index:03d}" for index in range(49)]
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    destination = (
        tempfile.NamedTemporaryFile(dir=args.output.parent) if args.output is not None else nullcontext()
    )
    digest = hashlib.sha256()
    with destination as output:
        for part in parts:
            with part.open("rb") as stream:
                for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(data)
                    if output is not None:
                        output.write(data)
        if digest.hexdigest() != expected:
            raise SystemExit("Checkpoint checksum mismatch. Check that all 49 Git LFS objects are available.")
        if output is not None:
            output.flush()
            os.link(output.name, args.output)
    print(f"Verified SHA-256 {expected}")


if __name__ == "__main__":
    main()
