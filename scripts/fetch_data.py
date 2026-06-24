#!/usr/bin/env python3
"""Fetch and normalize fusion datasets for deepiri-fuselk testing and development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepiri_fuselk.data.fetchers import FETCHERS, PUBLIC_DEFAULTS, run_fetch
from deepiri_fuselk.data.fetchers.manifest import load_manifest
from deepiri_fuselk.data.sources import load_catalog


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download public fusion data and normalize into .fuselk-data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_data.py --list-sources
  python scripts/fetch_data.py --all
  python scripts/fetch_data.py --source synthetic --shots 200
  python scripts/fetch_data.py --source odl --max-odl 100 --force

After fetch:
  fuselk data import .fuselk-data/shots/CMOD_1000606012.h5
  fuselk train elm --shots 200
  fuselk gui
        """,
    )
    parser.add_argument("--root", type=Path, default=Path(".fuselk-data"), help="Data root")
    parser.add_argument("--all", action="store_true", help="Fetch all public default sources")
    parser.add_argument("--source", action="append", dest="sources", help="Source id (repeatable)")
    parser.add_argument("--list-sources", action="store_true", help="Print catalog and exit")
    parser.add_argument("--manifest", action="store_true", help="Print fetch manifest and exit")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--shots", type=int, default=100, help="Synthetic corpus size")
    parser.add_argument("--grid", type=int, default=32, help="Synthetic grid size")
    parser.add_argument("--max-odl", type=int, default=50, help="Max ODL discharges to normalize")
    args = parser.parse_args()

    if args.list_sources:
        sources, loops = load_catalog()
        print("# fuselk data sources\n")
        for s in sources:
            print(f"## {s.id} [{s.tier}] — {s.name}")
            print(f"   device: {s.device}  format: {s.format}")
            if s.download:
                print(f"   download: {s.download}")
            print(f"   → modules: {', '.join(s.fuselk_modules)}")
            if s.notes:
                print(f"   notes: {s.notes[:120]}...")
            print()
        print("# feedback loops\n")
        for fb in loops:
            print(f"- {fb.name}: {fb.in_} → {fb.through} → {fb.out}")
        print(f"\n# fetchable now: {', '.join(sorted(FETCHERS))}")
        return 0

    if args.manifest:
        m = load_manifest(args.root)
        print(json.dumps(m.to_dict(), indent=2))
        return 0

    selected = list(PUBLIC_DEFAULTS) if args.all else (args.sources or list(PUBLIC_DEFAULTS))
    results = run_fetch(
        args.root,
        selected,
        force=args.force,
        n_shots=args.shots,
        grid_size=args.grid,
        max_odl_discharges=args.max_odl,
    )
    print(json.dumps({k: v.__dict__ for k, v in results.items()}, indent=2))
    print(f"\nDone. Data root: {args.root.resolve()}")
    print("Next: fuselk data import .fuselk-data/shots/<shot>.h5  |  fuselk train elm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
