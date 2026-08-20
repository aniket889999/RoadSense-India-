import csv
import random
import hashlib
from typing import Dict, List, Tuple

def parse_group_csv(csv_path: str, discovered_images: set) -> Tuple[Dict[str, str], List[str]]:
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return {}, ["Empty CSV"]

        if headers != ["image_relpath", "group_id"]:
            return {}, [f"Invalid headers: {headers}"]

        for row_idx, row in enumerate(reader, start=2):
            if len(row) != 2:
                return {}, [f"Row {row_idx}: must have exactly 2 columns"]

            relpath = row[0].strip()
            group_id = row[1].strip()

            if not relpath or not group_id:
                return {}, [f"Row {row_idx}: blank ID or relpath"]

            if relpath in mapping:
                return {}, [f"Row {row_idx}: duplicate path {relpath}"]

            if relpath not in discovered_images:
                return {}, [f"Row {row_idx}: Path {relpath} not found in discovered source images"]

            mapping[relpath] = group_id

    missing = discovered_images - set(mapping.keys())
    if missing:
        return {}, [f"Missing {len(missing)} discovered source images in groups CSV. Examples: {list(missing)[:3]}"]

    return mapping, []

def generate_safe_split(
    valid_images: List[str],
    group_mapping: Dict[str, str],
    positive_counts: Dict[str, int],
    seed: int = 42
) -> Tuple[Dict[str, str], List[str]]:

    groups = {}
    group_pos_img_count = {}
    group_instances = {}

    for path in valid_images:
        gid = group_mapping[path]
        if gid not in groups:
            groups[gid] = []
            group_pos_img_count[gid] = 0
            group_instances[gid] = 0

        groups[gid].append(path)
        if positive_counts.get(path, 0) > 0:
            group_pos_img_count[gid] += 1
            group_instances[gid] += positive_counts[path]

    positive_groups = [g for g in groups if group_pos_img_count[g] > 0]

    if len(positive_groups) < 3:
        return {}, [f"Failed: Only {len(positive_groups)} positive groups exist. Cannot satisfy train/val/test allocation."]

    group_ids = sorted(list(groups.keys()))
    rng = random.Random(seed)
    # Shuffle for tie-breaking
    rng.shuffle(group_ids)

    total_imgs = len(valid_images)
    total_pos_imgs = sum(group_pos_img_count.values())
    total_inst = sum(group_instances.values())

    target_p = {"train": 0.70, "val": 0.15, "test": 0.15}

    # We need at least 1 positive group per split.
    # Find the 3 smallest positive groups to minimize disruption.
    positive_groups.sort(key=lambda g: len(groups[g]))
    core_pos = positive_groups[:3]
    remaining_groups = [g for g in group_ids if g not in core_pos]

    # Process remaining largest to smallest (LPT heuristic)
    remaining_groups.sort(key=lambda g: len(groups[g]), reverse=True)

    best_loss = float('inf')
    best_split = None

    import itertools
    for p in itertools.permutations(["train", "val", "test"]):
        current_split = {
            core_pos[0]: p[0],
            core_pos[1]: p[1],
            core_pos[2]: p[2]
        }

        counts = {"train": 0, "val": 0, "test": 0}
        pos_img = {"train": 0, "val": 0, "test": 0}
        inst = {"train": 0, "val": 0, "test": 0}

        for gid, s in current_split.items():
            counts[s] += len(groups[gid])
            pos_img[s] += group_pos_img_count[gid]
            inst[s] += group_instances[gid]

        for gid in remaining_groups:
            g_len = len(groups[gid])
            g_pos = group_pos_img_count[gid]
            g_inst = group_instances[gid]

            best_s = None
            b_loss = float('inf')

            for s in ["train", "val", "test"]:
                loss = 0
                for os in ["train", "val", "test"]:
                    c = counts[os] + (g_len if os == s else 0)
                    pos = pos_img[os] + (g_pos if os == s else 0)
                    i = inst[os] + (g_inst if os == s else 0)

                    diff_count = (c / max(1, total_imgs)) - target_p[os]
                    diff_pos = (pos / max(1, total_pos_imgs)) - target_p[os]
                    diff_inst = (i / max(1, total_inst)) - target_p[os]

                    loss += (diff_count**2) + (diff_pos**2) + (diff_inst**2)

                if loss < b_loss:
                    b_loss = loss
                    best_s = s

            current_split[gid] = best_s
            counts[best_s] += g_len
            pos_img[best_s] += g_pos
            inst[best_s] += g_inst

        # Optional: Local improvement pass bounded to 5 iterations
        for _ in range(5):
            improved = False
            for gid in remaining_groups: # only move non-core groups to preserve positive guarantees safely
                old_s = current_split[gid]
                g_len = len(groups[gid])
                g_pos = group_pos_img_count[gid]
                g_inst = group_instances[gid]

                best_s = old_s
                b_loss = 0
                for s in ["train", "val", "test"]:
                    c = counts[s]
                    pos = pos_img[s]
                    i = inst[s]
                    diff_count = (c / max(1, total_imgs)) - target_p[s]
                    diff_pos = (pos / max(1, total_pos_imgs)) - target_p[s]
                    diff_inst = (i / max(1, total_inst)) - target_p[s]
                    b_loss += (diff_count**2) + (diff_pos**2) + (diff_inst**2)

                current_loss = b_loss

                for new_s in ["train", "val", "test"]:
                    if new_s == old_s: continue
                    loss = 0
                    for s in ["train", "val", "test"]:
                        c = counts[s] - (g_len if s == old_s else 0) + (g_len if s == new_s else 0)
                        pos = pos_img[s] - (g_pos if s == old_s else 0) + (g_pos if s == new_s else 0)
                        i = inst[s] - (g_inst if s == old_s else 0) + (g_inst if s == new_s else 0)
                        diff_count = (c / max(1, total_imgs)) - target_p[s]
                        diff_pos = (pos / max(1, total_pos_imgs)) - target_p[s]
                        diff_inst = (i / max(1, total_inst)) - target_p[s]
                        loss += (diff_count**2) + (diff_pos**2) + (diff_inst**2)

                    # Compare to current total loss
                    curr_total_loss = 0
                    for s in ["train", "val", "test"]:
                        diff_count = (counts[s] / max(1, total_imgs)) - target_p[s]
                        diff_pos = (pos_img[s] / max(1, total_pos_imgs)) - target_p[s]
                        diff_inst = (inst[s] / max(1, total_inst)) - target_p[s]
                        curr_total_loss += (diff_count**2) + (diff_pos**2) + (diff_inst**2)

                    if loss < curr_total_loss - 1e-9:
                        best_s = new_s

                if best_s != old_s:
                    current_split[gid] = best_s
                    counts[old_s] -= g_len
                    pos_img[old_s] -= g_pos
                    inst[old_s] -= g_inst
                    counts[best_s] += g_len
                    pos_img[best_s] += g_pos
                    inst[best_s] += g_inst
                    improved = True

            if not improved: break

        # Compute final loss for this permutation properly
        current_loss = 0
        for s in ["train", "val", "test"]:
            diff_count = (counts[s] / max(1, total_imgs)) - target_p[s]
            diff_pos = (pos_img[s] / max(1, total_pos_imgs)) - target_p[s]
            diff_inst = (inst[s] / max(1, total_inst)) - target_p[s]
            current_loss += (diff_count**2) + (diff_pos**2) + (diff_inst**2)

        if current_loss < best_loss:
            best_loss = current_loss
            best_split = current_split

    splits = {}
    for gid, s in best_split.items():
        for img in groups[gid]:
            splits[img] = s

    return splits, []

def verify_no_leakage(splits: Dict[str, str], group_mapping: Dict[str, str], image_hashes: Dict[str, str], audit_path: str = "") -> List[str]:
    errors = []

    # Group atomicity
    group_to_split = {}
    for path, split in splits.items():
        gid = group_mapping[path]
        if gid not in group_to_split:
            group_to_split[gid] = split
        elif group_to_split[gid] != split:
            errors.append(f"Leakage: Group {gid} spans {group_to_split[gid]} and {split}")

    # Exact hash leakage
    hash_to_split = {}
    for path, split in splits.items():
        h = image_hashes[path]
        if h not in hash_to_split:
            hash_to_split[h] = split
        elif hash_to_split[h] != split:
            errors.append(f"Leakage: Exact hash {h} (from {path}) spans {hash_to_split[h]} and {split}")

    return errors
