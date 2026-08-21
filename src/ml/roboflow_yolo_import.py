import os
import re
import csv
import json
import yaml
import shutil
import hashlib
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

from src.ml.group_split import generate_safe_split, verify_no_leakage
from src.ml.dataset_prep import write_pothole_yaml, write_manifest_csv
from src.ml.dataset_validation import validate_prepared_yolo_dataset

EXPECTED_SOURCE_CLASSES = [
    "D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"
]

def get_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_safe_output_path(source_root: Path, output_dir: Path, repo_root: Path) -> Tuple[bool, str]:
    if output_dir.exists() and output_dir.is_file():
        return False, f"Output target {output_dir} is an existing regular file, not a directory"

    try:
        src = source_root.resolve()
        out = output_dir.resolve()
        repo = repo_root.resolve()
        data_dir = (repo_root / "data").resolve()
        processed_dir = (repo_root / "data" / "processed").resolve()
    except Exception as e:
        return False, f"Failed to resolve paths: {str(e)}"

    # External symlink escape check
    if repo not in data_dir.parents and data_dir != repo:
        return False, f"data/ directory resolves outside the repository root: {data_dir}"

    if repo not in processed_dir.parents:
        return False, f"data/processed/ directory resolves outside the repository root: {processed_dir}"

    if out == src:
        return False, f"Output directory cannot be equal to source root: {out}"

    if src in out.parents:
        return False, f"Output directory cannot be inside source root: {out}"

    if out in src.parents:
        return False, f"Output directory cannot be a parent of source root: {out}"

    if out == repo:
        return False, f"Output directory cannot be repository root: {out}"

    if out == data_dir:
        return False, f"Output directory cannot be repo data folder: {out}"

    if out == processed_dir:
        return False, f"Output directory cannot be data/processed directly: must be a child subdirectory"

    if processed_dir not in out.parents:
        return False, f"Output directory {out} must be a child of {processed_dir}"

    return True, ""

def parse_roboflow_data_yaml(source_root: str) -> Tuple[bool, Dict[str, Any], str]:
    data_yaml_path = os.path.join(source_root, "data.yaml")
    if not os.path.exists(data_yaml_path):
        return False, {}, f"Source data.yaml missing at {data_yaml_path}"

    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return False, {}, f"Failed to parse data.yaml: {str(e)}"

    if not isinstance(data, dict):
        return False, {}, "data.yaml is not a valid YAML mapping"

    nc = data.get("nc")
    if nc is None or isinstance(nc, bool) or not isinstance(nc, int) or nc != 10:
        return False, {}, f"data.yaml 'nc' must be exactly integer 10, got: {nc}"

    names = data.get("names")
    if names is None:
        return False, {}, "data.yaml missing 'names' field"

    names_list = []
    if isinstance(names, list):
        names_list = names
    elif isinstance(names, dict):
        norm_map = {}
        for k, v in names.items():
            try:
                int_k = int(k)
            except (ValueError, TypeError):
                return False, {}, f"data.yaml names key '{k}' cannot be parsed as integer"
            if int_k < 0 or int_k > 9:
                return False, {}, f"data.yaml names key '{k}' out of range (0-9)"
            if int_k in norm_map:
                return False, {}, f"data.yaml duplicate names key '{int_k}'"
            norm_map[int_k] = v

        if len(norm_map) != 10 or set(norm_map.keys()) != set(range(10)):
            return False, {}, f"data.yaml names dictionary must define exactly keys 0 through 9, got: {list(norm_map.keys())}"
        names_list = [norm_map[i] for i in range(10)]
    else:
        return False, {}, "data.yaml 'names' must be a list or dictionary"

    if names_list != EXPECTED_SOURCE_CLASSES:
        return False, {}, f"Expected exact 10 source classes {EXPECTED_SOURCE_CLASSES}, but got: {names_list}"

    return True, data, ""

def infer_sequence_key(filename: str) -> Optional[Tuple[str, int]]:
    base = os.path.basename(filename)
    m = re.match(r"^([a-zA-Z]+)_(\d+)", base)
    if not m:
        return None
    prefix = m.group(1)
    seq_num = int(m.group(2))
    return prefix, seq_num

def build_contiguous_sequence_groups(
    filenames: List[str],
    max_consecutive_gap: int = 1
) -> Tuple[Dict[str, str], int, int, List[str]]:
    if max_consecutive_gap < 1:
        return {}, 0, 0, ["Max consecutive gap must be >= 1"]

    items_by_prefix: Dict[str, List[Tuple[int, str]]] = {}
    errors = []

    for fname in filenames:
        key = infer_sequence_key(fname)
        if key is None:
            errors.append(f"Cannot safely infer sequence key for filename '{fname}'. Random fallback is forbidden.")
            continue
        prefix, seq_num = key
        if prefix not in items_by_prefix:
            items_by_prefix[prefix] = []
        items_by_prefix[prefix].append((seq_num, fname))

    if errors:
        return {}, 0, 0, errors

    mapping = {}
    total_runs = 0
    adjacent_numeric_pairs_total = 0

    for prefix in sorted(items_by_prefix.keys()):
        sorted_items = sorted(items_by_prefix[prefix], key=lambda x: x[0])
        run_idx = 0
        prev_num = None

        for idx, (seq_num, fname) in enumerate(sorted_items):
            if prev_num is not None:
                gap = seq_num - prev_num
                if gap == 1:
                    adjacent_numeric_pairs_total += 1
                if gap > max_consecutive_gap:
                    run_idx += 1
            mapping[fname] = f"{prefix}_run_{run_idx}"
            prev_num = seq_num

        total_runs += (run_idx + 1)

    return mapping, total_runs, adjacent_numeric_pairs_total, []

def filter_remap_yolo_label(raw_content: str, lbl_filename: str, target_class_id: int = 6) -> Tuple[bool, List[str], int, str]:
    remapped_lines = []
    d40_count = 0
    lines = raw_content.strip().splitlines()

    for idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        parts = line_str.split()
        if len(parts) != 5:
            return False, [], 0, f"Label {lbl_filename} line {idx+1}: Expected 5 tokens, got {len(parts)} ('{line_str}')"

        try:
            cid = int(parts[0])
        except ValueError:
            return False, [], 0, f"Label {lbl_filename} line {idx+1}: Non-integer class ID '{parts[0]}'"

        if cid < 0 or cid >= len(EXPECTED_SOURCE_CLASSES):
            return False, [], 0, f"Label {lbl_filename} line {idx+1}: Unknown source class ID {cid} (must be 0-9)"

        try:
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            return False, [], 0, f"Label {lbl_filename} line {idx+1}: Non-float coordinates in '{line_str}'"

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h)):
            return False, [], 0, f"Label {lbl_filename} line {idx+1}: Non-finite coordinates in '{line_str}'"

        if w <= 0 or h <= 0 or (x - w/2) < 0 or (x + w/2) > 1 or (y - h/2) < 0 or (y + h/2) > 1:
            return False, [], 0, f"Label {lbl_filename} line {idx+1}: Out of bounds coordinates (x={x}, y={y}, w={w}, h={h})"

        if cid == target_class_id:
            remapped_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            d40_count += 1

    return True, remapped_lines, d40_count, ""

def run_roboflow_yolo_import(
    source_root: str,
    output_dir: str,
    max_consecutive_gap: int = 1,
    seed: int = 42,
    dry_run: bool = False,
    overwrite: bool = False,
    _inject_promotion_failure: bool = False,
    _inject_validation_failure: bool = False
) -> Tuple[bool, Dict[str, Any], str]:

    source_path = Path(source_root)
    out_path = Path(output_dir)
    repo_root = Path(__file__).resolve().parents[2]

    if not source_path.exists():
        return False, {}, f"Source directory does not exist: {source_root}"

    valid_yaml, data_yaml, yaml_err = parse_roboflow_data_yaml(str(source_path))
    if not valid_yaml:
        return False, {}, yaml_err

    safe_path, path_err = validate_safe_output_path(source_path, out_path, repo_root)
    if not safe_path:
        return False, {}, path_err

    # Recursive scan of images and labels across source splits
    source_splits = ["train", "valid", "test"]
    discovered_images = {}  # rel_img_path -> (split, full_img_path, full_lbl_path, rel_stem)
    source_inventory = []
    quarantine = []
    seen_stems = {}

    for s in source_splits:
        img_dir = source_path / s / "images"
        lbl_dir = source_path / s / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            return False, {}, f"Source split folder missing: {img_dir} or {lbl_dir}"

        img_files_in_split = {}
        for root, _, files in os.walk(img_dir):
            for f in sorted(files):
                p = Path(root) / f
                if p.is_file() and p.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel_img = p.relative_to(img_dir).as_posix()
                    rel_stem = p.relative_to(img_dir).with_suffix("").as_posix()
                    if rel_stem in img_files_in_split:
                        return False, {}, f"Same-stem image collision in {img_dir}: '{img_files_in_split[rel_stem][1]}' vs '{rel_img}'"
                    img_files_in_split[rel_stem] = (p, rel_img)

        lbl_files_in_split = {}
        for root, _, files in os.walk(lbl_dir):
            for f in sorted(files):
                p = Path(root) / f
                if p.is_file() and p.name.endswith(".txt"):
                    rel_stem = p.relative_to(lbl_dir).with_suffix("").as_posix()
                    lbl_files_in_split[rel_stem] = p

        # Check image-only orphans
        for rel_stem, (img_p, rel_img) in img_files_in_split.items():
            if rel_stem not in lbl_files_in_split:
                quarantine.append([rel_img, f"Orphan image missing label in split {s}"])
                source_inventory.append([rel_img, s, "orphan_image", "invalid", 0, "Missing label"])

        # Check label-only orphans
        for rel_stem, lbl_p in lbl_files_in_split.items():
            if rel_stem not in img_files_in_split:
                rel_lbl = lbl_p.relative_to(lbl_dir).as_posix()
                quarantine.append([rel_lbl, f"Orphan label missing image in split {s}"])
                source_inventory.append([rel_lbl, s, "orphan_label", "invalid", 0, "Missing image"])

        if quarantine:
            return False, {}, f"Preflight failed: {len(quarantine)} orphan files detected in source dataset. First: {quarantine[0][1]}"

        # Pair valid images
        for rel_stem, (img_p, rel_img) in img_files_in_split.items():
            lbl_p = lbl_files_in_split[rel_stem]
            if rel_img in discovered_images:
                return False, {}, f"Duplicate image filename across source splits: '{rel_img}'"
            if rel_stem in seen_stems:
                return False, {}, f"Cross-split stem collision: '{rel_img}' (split {s}) collides with '{seen_stems[rel_stem]}' (split {discovered_images[seen_stems[rel_stem]][0]})"

            seen_stems[rel_stem] = rel_img
            discovered_images[rel_img] = (s, str(img_p), str(lbl_p), rel_stem)

    if not discovered_images:
        return False, {}, "No supported image files discovered in source splits"

    image_hashes = {}
    seen_hashes = {}
    valid_data = {}
    positive_counts = {}

    for rel_img in sorted(discovered_images.keys()):
        s, img_path, lbl_path, rel_stem = discovered_images[rel_img]

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                im.load()
            h = get_file_hash(img_path)
            image_hashes[rel_img] = h
        except Exception as e:
            source_inventory.append([rel_img, s, "corrupted_image", "invalid", 0, str(e)])
            quarantine.append([rel_img, f"Corrupted image: {str(e)}"])
            continue

        # Exact duplicate image detection
        if h in seen_hashes:
            return False, {}, f"Duplicate image content detected: '{rel_img}' has identical content to '{seen_hashes[h]}'"
        seen_hashes[h] = rel_img

        try:
            with open(lbl_path, "r", encoding="utf-8") as lf:
                raw_content = lf.read()
        except UnicodeDecodeError as e:
            source_inventory.append([rel_img, s, "non_utf8_label", "invalid", 0, str(e)])
            quarantine.append([rel_img, f"Non-UTF8 label file: {str(e)}"])
            continue
        except Exception as e:
            source_inventory.append([rel_img, s, "unreadable_label", "invalid", 0, str(e)])
            quarantine.append([rel_img, f"Unreadable label file: {str(e)}"])
            continue

        ok, remapped_lines, d40_count, err = filter_remap_yolo_label(raw_content, os.path.basename(lbl_path), target_class_id=6)
        if not ok:
            source_inventory.append([rel_img, s, "malformed_label", "invalid", 0, err])
            quarantine.append([rel_img, err])
            continue

        source_inventory.append([rel_img, s, "valid_pair", "valid", d40_count, ""])
        valid_data[rel_img] = remapped_lines
        positive_counts[rel_img] = d40_count

    if quarantine:
        return False, {}, f"Validation Preflight Failed: {len(quarantine)} quarantined items. First error: {quarantine[0][1]}"

    if not valid_data:
        return False, {}, "No valid paired images could be processed"

    all_valid_fnames = sorted(list(valid_data.keys()))

    # Build contiguous sequence groups
    group_mapping, contiguous_runs, adjacent_pairs_total, group_errs = build_contiguous_sequence_groups(
        all_valid_fnames,
        max_consecutive_gap=max_consecutive_gap
    )
    if group_errs:
        return False, {}, f"Grouping Preflight Failed: {'; '.join(group_errs)}"

    # Generate safe group split
    splits, split_errs = generate_safe_split(all_valid_fnames, group_mapping, positive_counts, seed=seed)
    if split_errs:
        return False, {}, f"Group Splitting Failed: {'; '.join(split_errs)}"

    # Verify no group/hash leakage
    leakage_errs = verify_no_leakage(splits, group_mapping, image_hashes)
    if leakage_errs:
        return False, {}, f"Leakage Verification Failed: {'; '.join(leakage_errs)}"

    # Verify 0 cross-split adjacent numeric pairs
    adjacent_pairs_cross_split = 0
    items_by_prefix: Dict[str, List[Tuple[int, str]]] = {}
    for rel_img in all_valid_fnames:
        key = infer_sequence_key(rel_img)
        prefix, seq_num = key
        if prefix not in items_by_prefix:
            items_by_prefix[prefix] = []
        items_by_prefix[prefix].append((seq_num, rel_img))

    for prefix, item_list in items_by_prefix.items():
        sorted_items = sorted(item_list, key=lambda x: x[0])
        for i in range(len(sorted_items) - 1):
            n1, f1 = sorted_items[i]
            n2, f2 = sorted_items[i+1]
            if n2 == n1 + 1:
                if splits[f1] != splits[f2]:
                    adjacent_pairs_cross_split += 1

    if adjacent_pairs_cross_split != 0:
        return False, {}, f"Leakage Verification Failed: {adjacent_pairs_cross_split} adjacent numeric frame pairs cross split boundaries"

    # Compute statistics
    split_dist = {
        "train": {"images": 0, "positive_images": 0, "d40_instances": 0},
        "val": {"images": 0, "positive_images": 0, "d40_instances": 0},
        "test": {"images": 0, "positive_images": 0, "d40_instances": 0}
    }
    total_d40 = 0
    valid_negatives = 0
    groups_utilized = set()
    split_manifest = []

    for rel_img in all_valid_fnames:
        split = splits[rel_img]
        gid = group_mapping[rel_img]
        d40_c = positive_counts[rel_img]
        h = image_hashes[rel_img]
        groups_utilized.add(gid)

        split_dist[split]["images"] += 1
        if d40_c > 0:
            split_dist[split]["positive_images"] += 1
        split_dist[split]["d40_instances"] += d40_c

        total_d40 += d40_c
        if d40_c == 0:
            valid_negatives += 1

        rel_stem = discovered_images[rel_img][3]
        out_img = os.path.join("images", split, rel_img)
        out_lbl = os.path.join("labels", split, f"{rel_stem}.txt")
        split_manifest.append([rel_img, gid, split, h, d40_c > 0, d40_c, out_img, out_lbl])

    grouping_method = "contiguous_sequence_runs"
    limitation_text = "Grouping is a filename-based contiguous-sequence proxy, not verified road-route or capture-session grouping."

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path.resolve()),
        "source_provenance": data_yaml.get("roboflow", {}),
        "source_names": data_yaml.get("names", []),
        "target_remap": {
            "source_class_6_D40": {
                "source_name": "D40",
                "output_class": 0,
                "output_name": "pothole"
            }
        },
        "source_splits_discarded": True,
        "seed": seed,
        "max_consecutive_gap": max_consecutive_gap,
        "grouping_method": grouping_method,
        "residual_leakage_limitation": limitation_text,
        "counts": {
            "source_images_discovered": len(discovered_images),
            "eligible_images": len(valid_data),
            "quarantined_images": len(quarantine),
            "valid_negatives": valid_negatives,
            "d40_instances": total_d40,
            "contiguous_sequence_runs": contiguous_runs,
            "adjacent_numeric_pairs_total": adjacent_pairs_total,
            "adjacent_numeric_pairs_cross_split": adjacent_pairs_cross_split,
            "groups_utilized": len(groups_utilized),
            "split_distribution": split_dist
        }
    }

    canonical_state = {
        "seed": seed,
        "max_consecutive_gap": max_consecutive_gap,
        "grouping_method": grouping_method,
        "residual_leakage_limitation": limitation_text,
        "source_provenance": data_yaml.get("roboflow", {}),
        "source_names": data_yaml.get("names", []),
        "target_remap": metadata["target_remap"],
        "items": []
    }

    for rel_img in all_valid_fnames:
        canonical_state["items"].append({
            "img": rel_img,
            "hash": image_hashes[rel_img],
            "split": splits[rel_img],
            "group_id": group_mapping[rel_img],
            "yolo": valid_data[rel_img]
        })

    canonical_json = json.dumps(canonical_state, sort_keys=True)
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    metadata["dataset_fingerprint"] = fingerprint

    summary = {
        "dry_run": dry_run,
        "source_root": str(source_path.resolve()),
        "output_dir": str(out_path.resolve()),
        "fingerprint": fingerprint,
        "counts": metadata["counts"],
        "metadata": metadata
    }

    if dry_run:
        return True, summary, ""

    # Atomic Promotion Pattern with Explicit Rollback State Machine
    out_parent = out_path.resolve().parent
    out_parent.mkdir(parents=True, exist_ok=True)

    timestamp_suffix = f"{os.getpid()}_{int(datetime.now().timestamp())}_{hashlib.md5(str(out_path).encode()).hexdigest()[:6]}"
    stage_dir = out_parent / f".tmp_stage_{out_path.name}_{timestamp_suffix}"
    backup_dir = out_parent / f".tmp_backup_{out_path.name}_{timestamp_suffix}"

    backup_created = False
    promotion_succeeded = False
    validation_succeeded = False
    rollback_succeeded = False

    try:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)

        for split in ["train", "val", "test"]:
            (stage_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (stage_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (stage_dir / "manifests").mkdir(parents=True, exist_ok=True)

        for rel_img in all_valid_fnames:
            s, src_img_path, _, rel_stem = discovered_images[rel_img]
            split = splits[rel_img]
            remapped_lines = valid_data[rel_img]

            dst_img = stage_dir / "images" / split / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img_path, dst_img)

            dst_lbl = stage_dir / "labels" / split / f"{rel_stem}.txt"
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                if remapped_lines:
                    lf.write("\n".join(remapped_lines) + "\n")

        write_manifest_csv(str(stage_dir), "source_inventory.csv", ["image", "source_split", "status", "validation", "d40_count", "reason"], source_inventory)
        groups_used_rows = [[k, v] for k, v in group_mapping.items()]
        write_manifest_csv(str(stage_dir), "group_manifest_used.csv", ["image_relpath", "group_id"], groups_used_rows)
        write_manifest_csv(str(stage_dir), "split_manifest.csv", ["image", "group_id", "split", "sha256", "is_positive", "d40_instances", "out_image", "out_label"], split_manifest)
        write_manifest_csv(str(stage_dir), "quarantine.csv", ["file", "reason"], quarantine)

        with open(stage_dir / "manifests" / "preparation_metadata.json", "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=4)

        # Staging validation with stage-path YAML
        write_pothole_yaml(str(stage_dir))
        is_stage_valid, stage_val_err = validate_prepared_yolo_dataset(str(stage_dir))
        if not is_stage_valid:
            return False, summary, f"Validation of staged dataset failed: {stage_val_err}"

        # Write final canonical pothole.yaml for destination
        final_yaml_data = {
            "path": str(out_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "pothole"}
        }
        with open(stage_dir / "pothole.yaml", "w", encoding="utf-8") as yf:
            yaml.dump(final_yaml_data, yf, sort_keys=False)

        # Overwrite Handling
        if out_path.exists():
            if not overwrite:
                return False, summary, f"Output directory {out_path} already exists. Use --overwrite."
            out_path.rename(backup_dir)
            backup_created = True

        if _inject_promotion_failure:
            raise RuntimeError("Injected promotion failure for testing")

        stage_dir.rename(out_path)
        promotion_succeeded = True

        if _inject_validation_failure:
            raise RuntimeError("Injected validation failure for testing")

        is_final_valid, final_val_err = validate_prepared_yolo_dataset(str(out_path))
        if not is_final_valid:
            raise RuntimeError(f"Final output validation failed: {final_val_err}")
        validation_succeeded = True

        # Successful promotion and validation: safe to remove backup
        if backup_created and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

    except Exception as e:
        if backup_created and backup_dir.exists():
            try:
                if out_path.exists():
                    shutil.rmtree(out_path, ignore_errors=True)
                backup_dir.rename(out_path)
                rollback_succeeded = True
            except Exception as restore_err:
                raise RuntimeError(
                    f"Dataset promotion failed ({str(e)}) and rollback failed ({str(restore_err)}). "
                    f"Original backup preserved at: {backup_dir}"
                ) from e
        return False, summary, f"Promotion/Validation Error (Rollback={rollback_succeeded}): {str(e)}"

    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        # Note: backup_dir is ONLY removed if promotion_succeeded and validation_succeeded

    return True, summary, ""
