import argparse
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyedflib
import torch
from scipy.signal import resample_poly


CH_ORDER = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
]

CHANNEL_ALIAS = {
    "FP1": "Fp1",
    "F3": "F3",
    "C3": "C3",
    "P3": "P3",
    "F7": "F7",
    "T3": "T3",
    "T5": "T5",
    "O1": "O1",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "FP2": "Fp2",
    "F4": "F4",
    "C4": "C4",
    "P4": "P4",
    "F8": "F8",
    "T4": "T4",
    "T6": "T6",
    "O2": "O2",
    # common newer naming to older naming used by this repo
    "T7": "T3",
    "P7": "T5",
    "T8": "T4",
    "P8": "T6",
}


@dataclass
class Segment:
    key: str
    eeg_file: str
    split: str
    fraction: float
    signal: np.ndarray


def normalize_file_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_channel_name(raw_name: str) -> Optional[str]:
    if raw_name is None:
        return None
    name = raw_name.strip().upper()
    name = re.sub(r"^EEG\s*", "", name)
    name = re.sub(r"[-_\s]*(REF|AVG|LE|RE)$", "", name)
    name = name.replace(".", "")
    name = name.replace(" ", "")
    return CHANNEL_ALIAS.get(name)


def parse_clock_to_seconds(value: str) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(".", ":")
    if not re.match(r"^\d{1,2}:\d{1,2}:\d{1,2}(:\d{1,6})?$", text):
        return None
    parts = text.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2])
    ms = int(parts[3].ljust(6, "0")) if len(parts) == 4 else 0
    return h * 3600 + m * 60 + s + ms / 1_000_000


def parse_time_seconds(value: object, ref_clock_seconds: Optional[float]) -> Optional[float]:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # numeric string in seconds
    try:
        return float(text)
    except ValueError:
        pass

    clock = parse_clock_to_seconds(text)
    if clock is None:
        return None
    if ref_clock_seconds is None:
        return clock

    delta = clock - ref_clock_seconds
    if delta < 0:
        delta += 24 * 3600
    return delta


def is_clock_like(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return parse_clock_to_seconds(str(value).strip()) is not None


def resolve_column(df: pd.DataFrame, preferred_names: List[str]) -> Optional[str]:
    by_lower = {c.lower().strip(): c for c in df.columns}
    for name in preferred_names:
        key = name.lower().strip()
        if key in by_lower:
            return by_lower[key]
    for c in df.columns:
        cl = c.lower()
        if any(name.lower() in cl for name in preferred_names):
            return c
    return None


def read_annotation_intervals(csv_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    intervals_by_file: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    csv_files = []
    for root, _, files in os.walk(csv_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    for csv_path in sorted(csv_files):
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
        except Exception:
            continue

        if df.empty:
            continue

        df.columns = [str(c).strip() for c in df.columns]
        start_col = resolve_column(df, ["Start time", "start_time", "start"])
        end_col = resolve_column(df, ["End time", "end_time", "end"])
        file_col = resolve_column(df, ["File", "File Start", "edf", "recording"])
        if start_col is None or end_col is None:
            continue

        if file_col is not None:
            df[file_col] = df[file_col].ffill()
        df[start_col] = df[start_col].ffill()
        df[end_col] = df[end_col].ffill()

        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]

        for _, row in df.iterrows():
            file_value = row[file_col] if file_col is not None else None
            base_key = csv_stem if (file_value is None or is_clock_like(file_value)) else str(file_value)
            norm_key = normalize_file_key(base_key)
            if not norm_key:
                continue

            ref_clock = parse_clock_to_seconds(str(file_value)) if (file_value is not None and is_clock_like(file_value)) else None
            start_s = parse_time_seconds(row[start_col], ref_clock)
            end_s = parse_time_seconds(row[end_col], ref_clock)
            if start_s is None or end_s is None:
                continue
            if end_s < start_s:
                start_s, end_s = end_s, start_s
            if end_s == start_s:
                end_s = start_s + 1e-3
            intervals_by_file[norm_key].append((start_s, end_s))

    # merge overlaps per file for stable overlap computation
    merged: Dict[str, List[Tuple[float, float]]] = {}
    for key, intervals in intervals_by_file.items():
        if not intervals:
            continue
        intervals = sorted(intervals, key=lambda x: x[0])
        out = [intervals[0]]
        for s, e in intervals[1:]:
            ps, pe = out[-1]
            if s <= pe:
                out[-1] = (ps, max(pe, e))
            else:
                out.append((s, e))
        merged[key] = out
    return merged


def map_intervals_to_edf(
    intervals_by_key: Dict[str, List[Tuple[float, float]]], edf_paths: List[str]
) -> Dict[str, List[Tuple[float, float]]]:
    edf_key_to_stem = {normalize_file_key(os.path.splitext(os.path.basename(p))[0]): os.path.splitext(os.path.basename(p))[0] for p in edf_paths}
    mapped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for key, intervals in intervals_by_key.items():
        if key in edf_key_to_stem:
            mapped[edf_key_to_stem[key]].extend(intervals)
            continue

        candidates = [stem for norm, stem in edf_key_to_stem.items() if key in norm or norm in key]
        if len(candidates) == 1:
            mapped[candidates[0]].extend(intervals)
    return mapped


def assign_splits(files: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, str]:
    files = sorted(files)
    if not files:
        return {}
    rng = random.Random(seed)
    rng.shuffle(files)

    if len(files) == 1:
        return {files[0]: "train"}

    train_n = max(1, int(round(len(files) * train_ratio)))
    val_n = int(round(len(files) * val_ratio))
    if train_n + val_n >= len(files):
        val_n = max(0, len(files) - train_n - 1)
    test_n = len(files) - train_n - val_n
    if test_n == 0 and len(files) > 1:
        test_n = 1
        if val_n > 0:
            val_n -= 1
        else:
            train_n = max(1, train_n - 1)

    split_map: Dict[str, str] = {}
    for idx, stem in enumerate(files):
        if idx < train_n:
            split_map[stem] = "train"
        elif idx < train_n + val_n:
            split_map[stem] = "val"
        else:
            split_map[stem] = "test"
    return split_map


def resample_1d(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return signal.astype(np.float32, copy=False)
    g = math.gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(signal, up, down).astype(np.float32, copy=False)


def read_and_prepare_edf(edf_path: str, target_fs: int) -> Tuple[np.ndarray, int]:
    reader = pyedflib.EdfReader(edf_path)
    try:
        labels = reader.getSignalLabels()
        signals_by_channel: Dict[str, np.ndarray] = {}

        for idx, raw_label in enumerate(labels):
            canonical = normalize_channel_name(raw_label)
            if canonical is None or canonical in signals_by_channel:
                continue
            fs = int(round(reader.getSampleFrequency(idx)))
            raw_signal = np.asarray(reader.readSignal(idx), dtype=np.float32)
            signals_by_channel[canonical] = resample_1d(raw_signal, fs, target_fs)

        lengths = [len(v) for v in signals_by_channel.values()]
        if not lengths:
            raise RuntimeError(f"No usable EEG channels in {edf_path}")
        n = min(lengths)

        data_19 = []
        for ch in CH_ORDER:
            if ch in signals_by_channel:
                data_19.append(signals_by_channel[ch][:n])
            else:
                data_19.append(np.zeros(n, dtype=np.float32))
        data_19 = np.stack(data_19, axis=0)
        data_20 = np.vstack([data_19, np.zeros((1, n), dtype=np.float32)])
        return data_20, target_fs
    finally:
        reader.close()


def overlap_fraction(win_start: float, win_end: float, intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    overlap = 0.0
    for s, e in intervals:
        overlap += max(0.0, min(e, win_end) - max(s, win_start))
    denom = max(1e-8, win_end - win_start)
    return float(max(0.0, min(1.0, overlap / denom)))


def build_label_row(mat_filename: str, fraction: float) -> List[object]:
    votes = np.array([[fraction]], dtype=np.float32)
    return [np.array([mat_filename]), None, None, None, votes, None, None, None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NMT EDF+CSV to ProtoEEG data format.")
    parser.add_argument("--edf-dir", required=True, help="Directory containing EDF files.")
    parser.add_argument("--csv-dir", required=True, help="Directory containing CSV annotation files.")
    parser.add_argument("--output-dir", default="../sn2_data/organized_data", help="Output directory for pth/npy files.")
    parser.add_argument("--target-fs", type=int, default=128, help="Target sampling frequency.")
    parser.add_argument("--window-samples", type=int, default=128, help="Window length in samples at target-fs.")
    parser.add_argument("--step-samples", type=int, default=128, help="Step length in samples.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="File-level train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="File-level val ratio.")
    parser.add_argument("--normal-keep-prob", type=float, default=0.1, help="Probability of keeping normal windows.")
    parser.add_argument(
        "--label-mode",
        choices=["binary", "fraction"],
        default="binary",
        help="Use binary 0/1 labels or overlap fraction labels.",
    )
    parser.add_argument(
        "--positive-overlap-threshold",
        type=float,
        default=1e-6,
        help="Minimum overlap fraction to mark a binary positive window.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--write-split-csv", default="nmt_split.csv", help="Name for generated split CSV in output-dir.")
    args = parser.parse_args()

    if not (0 <= args.normal_keep_prob <= 1):
        raise ValueError("--normal-keep-prob must be between 0 and 1.")
    if not (0 <= args.positive_overlap_threshold <= 1):
        raise ValueError("--positive-overlap-threshold must be between 0 and 1.")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train/val ratios must satisfy: train>0, val>=0, train+val<1.")
    if args.window_samples <= 0 or args.step_samples <= 0:
        raise ValueError("--window-samples and --step-samples must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    edf_paths: List[str] = []
    for root, _, files in os.walk(args.edf_dir):
        for f in files:
            if f.lower().endswith(".edf"):
                edf_paths.append(os.path.join(root, f))
    edf_paths = sorted(edf_paths)
    if not edf_paths:
        raise RuntimeError(f"No EDF files found under {args.edf_dir}")

    intervals_by_key = read_annotation_intervals(args.csv_dir)
    intervals_by_edf = map_intervals_to_edf(intervals_by_key, edf_paths)

    file_stems = [os.path.splitext(os.path.basename(p))[0] for p in edf_paths]
    split_by_file = assign_splits(file_stems, args.train_ratio, args.val_ratio, args.seed)

    train_dict: Dict[str, np.ndarray] = {}
    val_dict: Dict[str, np.ndarray] = {}
    test_dict: Dict[str, np.ndarray] = {}
    train_labels: List[List[object]] = []
    val_labels: List[List[object]] = []
    test_labels: List[List[object]] = []
    split_rows: List[Dict[str, object]] = []

    for edf_path in edf_paths:
        stem = os.path.splitext(os.path.basename(edf_path))[0]
        split = split_by_file.get(stem, "train")
        intervals = intervals_by_edf.get(stem, [])

        try:
            data, fs = read_and_prepare_edf(edf_path, args.target_fs)
        except Exception as exc:
            print(f"[WARN] Skipping {stem}: {exc}")
            continue

        total_samples = data.shape[1]
        if total_samples < args.window_samples:
            print(f"[WARN] Skipping {stem}: too short ({total_samples} samples)")
            continue

        for start in range(0, total_samples - args.window_samples + 1, args.step_samples):
            end = start + args.window_samples
            chunk = data[:, start:end]
            win_start = start / fs
            win_end = end / fs
            overlap = overlap_fraction(win_start, win_end, intervals)
            if args.label_mode == "binary":
                fraction = 1.0 if overlap >= args.positive_overlap_threshold else 0.0
            else:
                fraction = overlap

            is_normal = fraction <= 0.0
            if is_normal and np.random.rand() > args.normal_keep_prob:
                continue

            event_id = f"{stem}_{start}"
            mat_name = f"{event_id}.mat"
            label_row = build_label_row(mat_name, fraction)

            if split == "train":
                train_dict[mat_name] = chunk
                train_labels.append(label_row)
            elif split == "val":
                val_dict[mat_name] = chunk
                val_labels.append(label_row)
            else:
                test_dict[mat_name] = chunk
                test_labels.append(label_row)

            split_rows.append(
                {
                    "event_file": event_id,
                    "eeg_file": stem,
                    "total_votes_received": 1,
                    "fraction_of_yes": fraction,
                    "Split": split,
                }
            )

    torch.save(train_dict, os.path.join(args.output_dir, "train_dict.pth"))
    torch.save(val_dict, os.path.join(args.output_dir, "val_dict.pth"))
    torch.save(test_dict, os.path.join(args.output_dir, "test_dict.pth"))
    np.save(os.path.join(args.output_dir, "sn2_train_labels.npy"), np.array(train_labels, dtype=object))
    np.save(os.path.join(args.output_dir, "sn2_val_labels.npy"), np.array(val_labels, dtype=object))
    np.save(os.path.join(args.output_dir, "sn2_test_labels.npy"), np.array(test_labels, dtype=object))

    split_csv_path = os.path.join(args.output_dir, args.write_split_csv)
    pd.DataFrame(split_rows).to_csv(split_csv_path, index=False)

    print("=" * 60)
    print("NMT conversion completed")
    print(f"EDF files found: {len(edf_paths)}")
    print(f"Train segments: {len(train_dict)}")
    print(f"Val segments:   {len(val_dict)}")
    print(f"Test segments:  {len(test_dict)}")
    print(f"Output dir:     {os.path.abspath(args.output_dir)}")
    print(f"Split CSV:      {split_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
