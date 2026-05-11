#!/usr/bin/env python3

from __future__ import annotations

import csv
import statistics
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_eval_metrics.py <name> <metrics_csv>")

    name = sys.argv[1]
    path = sys.argv[2]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unsafe = sum(float(r["unsafe"]) for r in rows)
    collision = sum(float(r["collision"]) for r in rows)
    fallen = sum(float(r["fallen"]) for r in rows)
    act_vx = statistics.mean(float(r["act_vx"]) for r in rows)
    nearest = statistics.mean(float(r["nearest_obs"]) for r in rows)
    print(f"{name} unsafe={unsafe:.0f} collision={collision:.0f} fallen={fallen:.0f} act_vx={act_vx:.3f} nearest={nearest:.3f}")


if __name__ == "__main__":
    main()
