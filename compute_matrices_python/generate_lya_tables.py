#!/usr/bin/env python3
"""Bulk-generate prototype LyA data tables from descriptor-backed channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lya_channel_manifest import write_manifest
from lya_descriptor_writer import DEFAULT_BIAS, write_debug_radial_probe, write_packed_fftlog_matrices


ROOT = Path(__file__).resolve().parent
DESCRIPTOR_DIR = ROOT / "lya_descriptors" / "13"
TABLE_DIR = ROOT / "lya_generated_tables" / "13"
RADIAL_TABLE_DIR = ROOT / "lya_generated_tables" / "13_radial"
MANIFEST_PATH = ROOT / "lya_generated_tables" / "lya_channel_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "old", "prepared"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--radial-output-dir", type=Path, default=RADIAL_TABLE_DIR)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--channel", action="append", default=[], help="Channel name to include, e.g. T13_B_PI3PAR")
    parser.add_argument("--mu", action="append", type=int, default=[], help="Restrict to one or more mu powers")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of descriptor-backed entries to process")
    parser.add_argument("--nmax", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = write_manifest(DESCRIPTOR_DIR, args.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    written = []
    radial_written = []
    skipped = []
    allowed_channels = set(args.channel)
    allowed_mu = set(args.mu)

    for entry in manifest["entries"]:
        print('Processing manifest entry: ',entry["channel_name"], entry["mu_power"])
        if entry["status"] != "sympy-reduced":
            skipped.append((entry["channel_name"], entry["mu_power"], entry["status"]))
            continue
        if allowed_channels and entry["channel_name"] not in allowed_channels:
            continue
        if allowed_mu and int(entry["mu_power"]) not in allowed_mu:
            continue
        if args.limit is not None and len(written) >= args.limit:
            break
        descriptor_path = Path(entry["path"])
        output_path = args.output_dir
        radial_output_path = args.radial_output_dir / f"LYA_M13_RADIAL__{entry['channel_name']}__MU{entry['mu_power']}.dat"
        written_paths = write_packed_fftlog_matrices(descriptor_path, output_path, nmax=args.nmax, bias=DEFAULT_BIAS, backend=args.backend)
        write_debug_radial_probe(descriptor_path, radial_output_path, backend=args.backend)
        written.extend(path for path, _ in written_paths)
        radial_written.append(radial_output_path)

    print(f"Backend: {args.backend}")
    print(f"Wrote {len(written)} descriptor-backed production-kernel LyA tables to {args.output_dir}")
    print(f"Wrote {len(radial_written)} debug LyA radial probes to {args.radial_output_dir}")
    if skipped:
        print("Skipped non-descriptor-backed entries:")
        for channel_name, mu_power, status in skipped:
            print(f"  {channel_name} mu^{mu_power}: {status}")


if __name__ == "__main__":
    main()
