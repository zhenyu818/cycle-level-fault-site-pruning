#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import sys
from collections import defaultdict, Counter
from copy import deepcopy

# -----------------------------
# Core parsers and utilities (kept)
# -----------------------------


def normalize_result(s: str) -> str:
    """Normalize result category to Masked / SDC / DUE / Others"""
    x = s.strip().lower()
    if "sdc" in x:
        return "SDC"
    if "due" in x:
        return "DUE"
    if "masked" in x:
        return "Masked"
    return "Others"


def parse_log(log_path: str):
    """Parse log entries (supports inline Effects+WRITER/READER and segmented modes)."""
    # 1) Standalone Effects header (legacy format)
    re_effects_start = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*$"
    )
    # 2) Inline form (new format)
    re_effects_inline = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*(.*\S.*)$"
    )
    # 3) Writer/Reader entries
    re_writer = re.compile(
        r"^\[(?P<src>[-A-Za-z0-9_]+)_FI_WRITER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_reader = re.compile(
        r"^\[(?P<src>[-A-Za-z0-9_]+)_FI_READER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    # 4) Results and parameters
    re_result = re.compile(r"^\[Run\s+(\d+)\]\s+(tmp\.out\d+):\s*(.*?)\s*$")
    re_params = re.compile(r"^\[INJ_PARAMS\]\s+\[Run\s+(\d+)\]\s+(tmp\.out\d+)\s+(.*)$")

    latest_effects_by_pair = (
        {}
    )  # {(run_id,name): [records...]} de-duplicated latest summary
    params_by_pair = {}  # {(run_id,name): "k=v;..."}
    cur_key = None
    cur_writers, cur_readers = [], []

    occ_counter = defaultdict(int)
    effects_occ, results_occ = {}, {}

    def _merge_unique(writers, readers):
        """Merge WRITER and READER; de-duplicate by (src, kernel, line, text)."""
        seen = set()
        merged = []
        for rec in writers + readers:
            key = (
                rec.get("src"),
                rec.get("kernel"),
                rec.get("inst_line"),
                rec.get("inst_text"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(deepcopy(rec))
        if not merged:
            merged = [
                {
                    "kernel": "invalid_summary",
                    "inst_line": -1,
                    "inst_text": "",
                    "src": "invalid",
                }
            ]
        return merged

    def _merge_records(existing, add):
        if not existing:
            return _merge_unique(add, [])
        if not add:
            return _merge_unique(existing, [])
        return _merge_unique(existing + add, [])

    def flush_current_effects():
        nonlocal cur_key, cur_writers, cur_readers
        if cur_key is not None:
            new_pack = _merge_unique(cur_writers, cur_readers)
            existed = latest_effects_by_pair.get(cur_key, [])
            latest_effects_by_pair[cur_key] = _merge_records(existed, new_pack)
            cur_key = None
            cur_writers, cur_readers = [], []

    # If the log file is missing, do not exit; return empty data
    if not os.path.exists(log_path):
        print(f"Warning: log file not found: {log_path}", file=sys.stderr)
        return {}, {}, {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # (A) Inline Effects + Writer/Reader
            m = re_effects_inline.match(line)
            if m:
                run_id = int(m.group(1))
                name = m.group(2)
                rest = m.group(3).strip()
                new_key = (run_id, name)

                if cur_key != new_key:
                    flush_current_effects()
                    cur_key = new_key
                    cur_writers, cur_readers = [], []

                mw = re_writer.match(rest)
                if mw:
                    cur_writers.append(
                        {
                            "kernel": mw.group(2),
                            "inst_line": int(mw.group(4)),
                            "inst_text": mw.group(5).strip(),
                            "src": mw.group("src"),
                        }
                    )
                    continue
                mr = re_reader.match(rest)
                if mr:
                    cur_readers.append(
                        {
                            "kernel": mr.group(2),
                            "inst_line": int(mr.group(4)),
                            "inst_text": mr.group(5).strip(),
                            "src": mr.group("src"),
                        }
                    )
                    continue
                continue

            # (B) Effects header (legacy)
            m = re_effects_start.match(line)
            if m:
                flush_current_effects()
                run_id = int(m.group(1))
                name = m.group(2)
                cur_key = (run_id, name)
                cur_writers, cur_readers = [], []
                continue

            # (C) Accumulate WRITER/READER under current key (legacy)
            if cur_key is not None:
                m = re_writer.match(line)
                if m:
                    cur_writers.append(
                        {
                            "kernel": m.group(2),
                            "inst_line": int(m.group(4)),
                            "inst_text": m.group(5).strip(),
                            "src": m.group("src"),
                        }
                    )
                    continue
                m = re_reader.match(line)
                if m:
                    cur_readers.append(
                        {
                            "kernel": m.group(2),
                            "inst_line": int(m.group(4)),
                            "inst_text": m.group(5).strip(),
                            "src": m.group("src"),
                        }
                    )
                    continue

            # (D) INJ_PARAMS
            m = re_params.match(line)
            if m:
                run_id = int(m.group(1))
                name = m.group(2)
                params_by_pair[(run_id, name)] = m.group(3).strip()
                continue

            # (E) Result line: bind outcome
            m = re_result.match(line)
            if m:
                run_id = int(m.group(1))
                name = m.group(2)
                res = normalize_result(m.group(3))
                pair = (run_id, name)

                occ_counter[pair] += 1
                idx = occ_counter[pair]
                inj_key = (run_id, name, idx)

                if cur_key == pair:
                    current_pack = _merge_unique(cur_writers, cur_readers)
                    existed = latest_effects_by_pair.get(pair, [])
                    recs = _merge_records(existed, current_pack)
                    latest_effects_by_pair[pair] = deepcopy(recs)
                else:
                    recs = latest_effects_by_pair.get(
                        pair,
                        [
                            {
                                "kernel": "invalid_summary",
                                "inst_line": -1,
                                "inst_text": "",
                                "src": "invalid",
                            }
                        ],
                    )

                effects_occ[inj_key] = deepcopy(recs)
                results_occ[inj_key] = res
                continue

    flush_current_effects()
    return effects_occ, results_occ, params_by_pair


# -----------------------------
# Write test_result.csv (this run only; do not merge old files)
# -----------------------------


def write_csv(
    app: str,
    test: str,
    components: str,
    bitflip: str,
    effects_occ,
    results_occ,
    params_by_pair,
):
    """
    Generate the CSV purely from the log data parsed in this run:
      - Do not read previous CSV files and do not merge results.
      - Keep per-source count columns and overall summary columns.
      - For non-invalid rows, count reg_names (from INJ_PARAMS reg_name=...).
    """
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"test_result_{app}_{test}_{components}_{bitflip}.csv"
    )

    inst_counts = defaultdict(
        lambda: defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0, "Others": 0})
    )
    all_srcs = set()
    regname_counts = defaultdict(Counter)  # Count only non-invalid rows

    # Injection results (new data from this run only)
    for inj_key, recs in effects_occ.items():
        res_cat = results_occ.get(inj_key, "Others")
        run_id, name, _ = inj_key
        combo = params_by_pair.get((run_id, name), "") or ""

        # Extract reg_name list
        reg_names_this = []
        for part in combo.split(";"):
            part = part.strip()
            if part.startswith("reg_name="):
                raw = part.split("=", 1)[1].strip()
                if raw:
                    reg_names_this = [x.strip() for x in raw.split(":") if x.strip()]
                break

        for rec in recs:
            kernel = rec.get("kernel") or "unknown"
            inst_line = rec.get("inst_line")
            inst_line = -1 if inst_line is None else int(inst_line)
            inst_text = rec.get("inst_text") or "unknown"
            src = rec.get("src", "unknown")
            key = (kernel, inst_line, inst_text)

            inst_counts[key][src][res_cat] += 1
            all_srcs.add(src)

            if kernel == "invalid_summary" or src == "invalid":
                continue
            if reg_names_this:
                for rn in reg_names_this:
                    regname_counts[key][rn] += 1

    # Write CSV
    src_columns = []
    for src in sorted(all_srcs):
        src_columns += [f"{src}_Masked", f"{src}_SDC", f"{src}_DUE", f"{src}_Others"]

    fieldnames = (
        ["kernel", "inst_line", "inst_text", "reg_names"]
        + src_columns
        + ["Masked", "SDC", "DUE", "Others", "tot_inj"]
    )

    out_path_tmp = out_path + ".tmp"
    with open(out_path_tmp, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for kernel, inst_line, inst_text in sorted(
            inst_counts.keys(), key=lambda k: (k[0], k[1], k[2])
        ):
            src_map = inst_counts[(kernel, inst_line, inst_text)]
            row = {
                "kernel": kernel,
                "inst_line": "" if inst_line < 0 else inst_line,
                "inst_text": inst_text,
            }

            # reg_names
            if kernel == "invalid_summary":
                row["reg_names"] = ""
            else:
                rn_counts = regname_counts.get((kernel, inst_line, inst_text), {})
                if rn_counts:
                    pairs = sorted(rn_counts.items(), key=lambda x: (-x[1], x[0]))
                    row["reg_names"] = ",".join([f"{rn}:{cnt}" for rn, cnt in pairs])
                else:
                    row["reg_names"] = ""

            # per-source and totals (this run only)
            tot_m = tot_s = tot_d = tot_o = 0
            for src in sorted(all_srcs):
                m = src_map.get(src, {}).get("Masked", 0)
                s = src_map.get(src, {}).get("SDC", 0)
                d = src_map.get(src, {}).get("DUE", 0)
                o = src_map.get(src, {}).get("Others", 0)
                row[f"{src}_Masked"] = m
                row[f"{src}_SDC"] = s
                row[f"{src}_DUE"] = d
                row[f"{src}_Others"] = o
                tot_m += m
                tot_s += s
                tot_d += d
                tot_o += o

            row["Masked"] = tot_m
            row["SDC"] = tot_s
            row["DUE"] = tot_d
            row["Others"] = tot_o
            row["tot_inj"] = tot_m + tot_s + tot_d + tot_o

            writer.writerow(row)

    os.replace(out_path_tmp, out_path)
    return out_path


# -----------------------------
# Main flow (CSV generation only)
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse inst_exec.log and write test_result CSV (no merging, no stop rules, no result_info)."
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    parser.add_argument("--component", "-c", required=True, help="Component set")
    parser.add_argument(
        "--bitflip", "-b", required=True, help="Number of bit flips to inject"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "inst_exec.log")

    effects_occ, results_occ, params_by_pair = parse_log(log_path)
    total_sdc = sum(1 for v in results_occ.values() if v == "SDC")
    out_path = write_csv(
        args.app,
        args.test,
        args.component,
        args.bitflip,
        effects_occ,
        results_occ,
        params_by_pair,
    )
    print(f"Wrote CSV: {out_path}")
    print(f"Total SDC: {total_sdc}")


if __name__ == "__main__":
    main()
