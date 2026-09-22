#!/usr/bin/env python3
"""Build a deterministic two-record smoke subset from the real 3ET data."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping


def select_first_record(lines: Iterable[str]) -> str:
    records = [line.strip() for line in lines if line.strip()]
    if not records:
        raise ValueError("file list is empty")
    return records[0]


def event_stop_index(
    timestamps,
    start: int,
    duration_us: int,
) -> int:
    return bisect.bisect_left(timestamps, start + duration_us)


def merge_smoke_config(
    base: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(content)
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _crop_record(
    source_data: Path,
    output_data: Path,
    record_id: str,
    *,
    labels: int,
    duration_us: int,
) -> dict[str, object]:
    import h5py

    source_dir = source_data / "train" / record_id
    source_h5 = source_dir / f"{record_id}.h5"
    source_labels = source_dir / "label.txt"
    if not source_h5.is_file() or not source_labels.is_file():
        raise FileNotFoundError(f"record files are missing: {record_id}")

    label_lines = source_labels.read_text(encoding="utf-8").splitlines()
    if len(label_lines) < labels:
        raise ValueError(
            f"record {record_id} has {len(label_lines)} labels; "
            f"{labels} required"
        )

    destination_dir = output_data / "train" / record_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_h5 = destination_dir / f"{record_id}.h5"
    destination_labels = destination_dir / "label.txt"

    with h5py.File(source_h5, "r") as source:
        if "events" not in source:
            raise ValueError(f"record {record_id} has no events dataset")
        events = source["events"]
        if not events.shape or events.shape[0] == 0:
            raise ValueError(f"record {record_id} has no events")
        timestamps = events["t"][:]
        if bool((timestamps[1:] < timestamps[:-1]).any()):
            raise ValueError(
                f"record {record_id} event timestamps are not monotonic"
            )
        start = int(timestamps[0])
        required_end = start + duration_us
        if int(timestamps[-1]) < required_end:
            raise ValueError(
                f"record {record_id} does not cover {duration_us} us"
            )
        stop = event_stop_index(timestamps, start, duration_us)
        if stop < 1:
            raise ValueError(f"record {record_id} crop has no events")
        with h5py.File(destination_h5, "w") as destination:
            _copy_attrs(source, destination)
            cropped = destination.create_dataset(
                "events",
                data=events[:stop],
                dtype=events.dtype,
            )
            _copy_attrs(events, cropped)

    _atomic_text(
        destination_labels,
        "\n".join(label_lines[:labels]) + "\n",
    )
    return {
        "record_id": record_id,
        "events": stop,
        "labels": labels,
        "first_timestamp_us": start,
        "last_timestamp_us": int(timestamps[stop - 1]),
        "duration_us": duration_us,
        "source_h5": str(source_h5),
        "output_h5": str(destination_h5),
        "h5_sha256": _sha256(destination_h5),
        "labels_sha256": _sha256(destination_labels),
    }


def build_subset(
    source_data: Path,
    source_lists: Path,
    output_root: Path,
    *,
    labels: int = 200,
    duration_us: int = 2_000_000,
) -> dict[str, object]:
    if labels < 1 or duration_us < 1:
        raise ValueError("labels and duration-us must be positive")
    source_data = Path(source_data)
    source_lists = Path(source_lists)
    output_root = Path(output_root)

    records = {
        split: select_first_record(
            (source_lists / f"{split}_files.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for split in ("train", "val")
    }
    temporary = output_root.with_name(f".{output_root.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        report: dict[str, object] = {
            split: _crop_record(
                source_data,
                temporary / "event_data",
                record,
                labels=labels,
                duration_us=duration_us,
            )
            for split, record in records.items()
        }
        for split, record in records.items():
            _atomic_text(
                temporary / "dataset" / f"{split}_files.txt",
                f"{record}\n",
            )
        report["labels_per_record"] = labels
        report["duration_us"] = duration_us
        atomic_write_json(temporary / "subset_report.json", report)
        shutil.rmtree(output_root, ignore_errors=True)
        temporary.replace(output_root)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--source-lists", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--labels", type=int, default=200)
    parser.add_argument("--duration-us", type=int, default=2_000_000)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--smoke-overrides", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_subset(
            args.source_data,
            args.source_lists,
            args.output_root,
            labels=args.labels,
            duration_us=args.duration_us,
        )
        base = json.loads(args.base_config.read_text(encoding="utf-8"))
        overrides = json.loads(
            args.smoke_overrides.read_text(encoding="utf-8")
        )
        atomic_write_json(
            args.runtime_config,
            merge_smoke_config(base, overrides),
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
