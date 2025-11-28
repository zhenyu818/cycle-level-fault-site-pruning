#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import re
import sys


def parse_csv(csv_path):
    """
    Parse the CSV file and accumulate for each register r:
      - N_r: number of injections (from the reg_names column, like "%r2:5")
      - SDC_r: number of SDCs (from the SDC column)

    Returns:
      reg_stats: dict, key is register name (e.g. "%r2"),
                 value is {"N": N_r, "SDC": SDC_r}
    """
    reg_stats = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg_field = row.get("reg_names", "").strip()
            if not reg_field:
                # For example, rows like invalid_summary have no reg_names field; skip them directly
                continue

            # reg_field looks like "%r2:5"
            try:
                reg_name, inj_count_str = reg_field.split(":")
            except ValueError:
                # Skip rows with malformed format
                continue

            reg_name = reg_name.strip()
            inj_count_str = inj_count_str.strip()
            if not inj_count_str:
                continue

            try:
                N_r_row = int(inj_count_str)
            except ValueError:
                continue

            # SDC column for this row
            sdc_str = row.get("SDC", "").strip()
            try:
                SDC_r_row = int(sdc_str) if sdc_str else 0
            except ValueError:
                SDC_r_row = 0

            if reg_name not in reg_stats:
                reg_stats[reg_name] = {"N": 0, "SDC": 0}

            reg_stats[reg_name]["N"] += N_r_row
            reg_stats[reg_name]["SDC"] += SDC_r_row

    return reg_stats


def parse_danger_log(danger_path):
    """
    Parse danger.log and compute the dangerous cycle length d_r for each register r.

    Each line looks like:
      [danger region] reg=%rs55 cycles=959-2905,3228-3547

    d_r is the sum of the lengths of each interval, computed as closed intervals:
      959-2905 -> 2905 - 959 + 1 = 1947

    Returns:
      danger_stats: dict, key is register name (e.g. "%rs55"),
                    value is d_r
    """
    danger_stats = {}

    # The regex allows digits, commas, and hyphens in cycles
    pattern = re.compile(r"reg=(%\S+)\s+cycles=([0-9,\-]+)")

    with open(danger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = pattern.search(line)
            if not m:
                continue

            reg_name = m.group(1)
            cycles_part = m.group(2)

            total_len = 0
            # There may be multiple ranges separated by commas
            for seg in cycles_part.split(","):
                seg = seg.strip()
                if not seg:
                    continue

                if "-" in seg:
                    start_str, end_str = seg.split("-")
                    start = int(start_str)
                    end = int(end_str)
                else:
                    # Single cycle, treat as length 1
                    start = end = int(seg)

                # Closed interval length
                total_len += (end - start + 1)

            danger_stats[reg_name] = total_len

    return danger_stats


def compute_p(csv_filename, T, R_user):
    """
    Compute:
      p = (1 / R) * Σ_r ( d_r * SDC_r / (T * N_r) )

    Where:
      - R is specified by the user as the third argument (R_user)
      - d_r: from danger.log
      - N_r, SDC_r: from the CSV
      - T is the total number of cycles

    Returns:
      p, R_user, used_regs, sum_terms
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_filename)
    danger_path = os.path.join(base_dir, "danger.log")

    reg_stats = parse_csv(csv_path)
    danger_stats = parse_danger_log(danger_path)

    T = float(T)
    R = float(R_user)

    print("========== Computation starts ==========")
    print(f"Reading CSV file: {csv_path}")
    print(f"Reading danger.log file: {danger_path}")
    print(f"Total cycles T = {T}")
    print(f"User specified R = {R}")
    print()
    print("=== Per-register calculation (only registers that appear in both CSV and danger.log and have N_r > 0 are summed) ===")

    sum_terms = 0.0
    used_regs = 0

    # For more aligned output, sort by register name
    for reg_name in sorted(danger_stats.keys()):
        d_r = danger_stats[reg_name]
        stats = reg_stats.get(reg_name)

        if stats is None:
            print(f"{reg_name}: No injection record found in CSV, skip (contribution considered 0)")
            continue

        N_r = stats["N"]
        SDC_r = stats["SDC"]

        if N_r <= 0:
            print(f"{reg_name}: N_r = {N_r} (<=0), cannot compute this register term, skip (contribution considered 0)")
            continue

        term = (d_r * SDC_r) / (T * N_r)
        sum_terms += term
        used_regs += 1

        print(
            f"{reg_name}: d_r = {d_r:6d}, N_r = {N_r:4d}, SDC_r = {SDC_r:4d}, "
            f"term = d_r * SDC_r / (T * N_r) = {term:.6e}"
        )

    print("\n=== Summary ===")
    print(f"Number of registers participating in sum used_regs = {used_regs}")
    print(f"Σ_r [d_r * SDC_r / (T * N_r)] = {sum_terms:.6e}")

    if R == 0:
        print("Warning: R = 0, cannot compute p, return p = 0")
        return 0.0, R, used_regs, sum_terms

    p = sum_terms / R
    print(f"R (user specified) = {R}")
    print(f"Final result p = Σ / R = {p:.6e}")
    print("========== Computation ends ==========")

    return p, R, used_regs, sum_terms


if __name__ == "__main__":
    """
    Usage example:
      python calc_p.py conv1d.csv 3548 71

    Where:
      - 1st argument: CSV filename (in the same directory as this script)
      - 2nd argument: total number of cycles T
      - 3rd argument: R (specified by yourself, e.g. 71)
    """
    if len(sys.argv) != 4:
        print(f"Usage: {os.path.basename(sys.argv[0])} <csv_filename> <T> <R>")
        sys.exit(1)

    csv_filename = sys.argv[1]
    T = sys.argv[2]
    R_user = sys.argv[3]

    compute_p(csv_filename, T, R_user)
