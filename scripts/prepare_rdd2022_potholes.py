import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import shutil
import hashlib
import glob
import json
import csv
from datetime import datetime, timezone
from PIL import Image

from src.ml.rdd2022 import parse_rdd_xml, convert_to_yolo
from src.ml.dataset_validation import validate_dataset_item
from src.ml.group_split import parse_group_csv, generate_safe_split, verify_no_leakage
from src.ml.dataset_prep import init_yolo_directories, write_pothole_yaml, write_manifest_csv
from src.ml.metadata import write_metadata_json

def get_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def get_ahash(im: Image.Image) -> str:
    # 8x8 grayscale
    im = im.resize((8, 8), Image.Resampling.LANCZOS).convert("L")
    pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join(['1' if p > avg else '0' for p in pixels])
    return hex(int(bits, 2))[2:].zfill(16)

def hamming_distance(h1: str, h2: str) -> int:
    b1 = bin(int(h1, 16))[2:].zfill(64)
    b2 = bin(int(h2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))

def find_data(rdd_root: str):
    img_search = os.path.join(rdd_root, "**", "India", "train", "images")
    xml_search = os.path.join(rdd_root, "**", "India", "train", "annotations", "xmls")

    img_dirs = [p for p in glob.glob(img_search, recursive=True) if os.path.isdir(p)]
    xml_dirs = [p for p in glob.glob(xml_search, recursive=True) if os.path.isdir(p)]

    if len(img_dirs) > 1 or len(xml_dirs) > 1:
        print("Multiple India/train roots found. Ambiguous dataset.")
        sys.exit(1)

    images_dir = img_dirs[0] if img_dirs else os.path.join(rdd_root, "India", "train", "images")
    xml_dir = xml_dirs[0] if xml_dirs else os.path.join(rdd_root, "India", "train", "annotations", "xmls")

    return images_dir, xml_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-root", required=True)
    parser.add_argument("--groups-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-near-duplicates", action="store_true")
    args = parser.parse_args()

    # Overwrite safety check
    if args.overwrite:
        abs_out = os.path.realpath(args.output_dir)
        repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        forbidden = [
            os.path.realpath("/"),
            os.path.realpath(os.path.expanduser("~")),
            repo_root,
            os.path.realpath(os.path.join(repo_root, "data")),
            os.path.realpath(os.path.join(repo_root, "outputs"))
        ]
        if abs_out in forbidden:
            print(f"Safety constraint: Will not overwrite protected path {abs_out}")
            sys.exit(1)

    images_dir, xml_dir = find_data(args.rdd_root)

    images_found = set()
    if os.path.exists(images_dir):
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(".jpg"):
                    rel = os.path.relpath(os.path.join(root, f), images_dir)
                    images_found.add(rel)

    xmls_found = set()
    if os.path.exists(xml_dir):
        for root, _, files in os.walk(xml_dir):
            for f in files:
                if f.lower().endswith(".xml"):
                    rel = os.path.relpath(os.path.join(root, f), xml_dir)
                    xmls_found.add(rel)

    if not images_found:
        print("No images found.")
        sys.exit(1)

    group_mapping, mapping_errs = parse_group_csv(args.groups_csv, images_found)
    if mapping_errs:
        for err in mapping_errs: print(err)
        sys.exit(1)

    source_inventory = []
    exclusions = []
    image_hashes = {}
    ahashes = {}
    valid_data = {}
    positive_counts = {}

    for img in sorted(list(images_found)):
        img_path = os.path.join(images_dir, img)
        xml_rel = os.path.splitext(img)[0] + ".xml"
        xml_path = os.path.join(xml_dir, xml_rel)

        has_xml = xml_rel in xmls_found

        if not has_xml:
            source_inventory.append([img, xml_rel, "missing_xml", "invalid", 0, "Missing XML"])
            exclusions.append([img, "Missing XML"])
            continue

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                im.load()
                decoded_width, decoded_height = im.size
                image_hashes[img] = get_file_hash(img_path)
                if args.audit_near_duplicates:
                    ahashes[img] = get_ahash(im)
        except Exception as e:
            source_inventory.append([img, xml_rel, "paired", "invalid", 0, f"Corrupted image: {str(e)}"])
            exclusions.append([img, f"Corrupted image: {str(e)}"])
            continue

        xml_w, xml_h, boxes, parse_err = parse_rdd_xml(xml_path)
        if parse_err:
            source_inventory.append([img, xml_rel, "paired", "invalid", 0, parse_err])
            exclusions.append([img, parse_err])
            continue

        is_valid, val_err = validate_dataset_item(xml_w, xml_h, decoded_width, decoded_height, boxes)
        if not is_valid:
            source_inventory.append([img, xml_rel, "paired", "invalid", 0, val_err])
            exclusions.append([img, val_err])
            continue

        try:
            yolo_lines = convert_to_yolo(boxes, decoded_width, decoded_height)
        except ValueError as e:
            source_inventory.append([img, xml_rel, "paired", "invalid", 0, str(e)])
            exclusions.append([img, str(e)])
            continue

        source_inventory.append([img, xml_rel, "paired", "valid", len(boxes), ""])
        valid_data[img] = yolo_lines
        positive_counts[img] = len(boxes)

    # Check for orphan XMLs
    for xml in xmls_found:
        img_name = os.path.splitext(xml)[0] + ".jpg"
        if img_name not in images_found:
            source_inventory.append(["missing_img", xml, "orphan_xml", "invalid", 0, "Orphan XML"])
            exclusions.append([xml, "Orphan XML"])

    if not valid_data:
        print("No valid data parsed.")
        sys.exit(1)

    splits, split_errors = generate_safe_split(list(valid_data.keys()), group_mapping, positive_counts, args.seed)
    if split_errors:
        for err in split_errors: print(err)
        sys.exit(1)

    leakage_errors = verify_no_leakage(splits, group_mapping, image_hashes, "")
    if leakage_errors:
        for err in leakage_errors: print(err)
        sys.exit(1)

    audit_rows = []
    if args.audit_near_duplicates:
        valid_imgs = sorted(list(valid_data.keys()))
        for i in range(len(valid_imgs)):
            for j in range(i + 1, len(valid_imgs)):
                im1 = valid_imgs[i]
                im2 = valid_imgs[j]
                # Compare cross-split only
                if splits[im1] != splits[im2]:
                    dist = hamming_distance(ahashes[im1], ahashes[im2])
                    if dist <= 10: # threshold
                        audit_rows.append([im1, splits[im1], ahashes[im1], im2, splits[im2], ahashes[im2], dist])

    # Attempt to read sidecar metadata for group quality
    grouping_method = "user_supplied_grouping_unverified"
    limitation = "user_supplied_grouping_unverified"
    sidecar_meta = {}
    meta_path = os.path.splitext(args.groups_csv)[0] + "_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            sidecar_meta = json.load(f)
            grouping_method = sidecar_meta.get("grouping_quality", grouping_method)
            limitation = sidecar_meta.get("limitation", grouping_method)

    # Canonical fingerprint state
    canonical_state = {
        "seed": args.seed,
        "grouping_method": grouping_method,
        "limitation": limitation,
        "sidecar": sidecar_meta,
        "items": []
    }

    # Pre-populate canonical state to hash before writing directories
    for img in sorted(list(valid_data.keys())):
        split = splits[img]
        yolo_lines = valid_data[img]
        gid = group_mapping[img]
        h = image_hashes[img]
        canonical_state["items"].append({
            "img": img, "hash": h, "split": split, "group_id": gid, "yolo": yolo_lines
        })

    canonical_json = json.dumps(canonical_state, sort_keys=True)
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    # PREFLIGHT PASSED. Create directories
    if not init_yolo_directories(args.output_dir, args.overwrite):
        print(f"Output directory {args.output_dir} is not empty. Use --overwrite.")
        sys.exit(1)

    split_manifest = []
    group_counts = set()

    split_dist = {
        "train": {"images": 0, "positive_images": 0, "d40_instances": 0},
        "val": {"images": 0, "positive_images": 0, "d40_instances": 0},
        "test": {"images": 0, "positive_images": 0, "d40_instances": 0}
    }

    total_d40 = 0
    valid_negatives = 0

    for img in sorted(list(valid_data.keys())):
        split = splits[img]
        yolo_lines = valid_data[img]
        count = positive_counts[img]
        gid = group_mapping[img]
        h = image_hashes[img]

        # PRESERVE relative path
        src_img = os.path.join(images_dir, img)
        dst_img = os.path.join(args.output_dir, "images", split, img)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(src_img, dst_img)

        rel_label = os.path.splitext(img)[0] + ".txt"
        dst_label = os.path.join(args.output_dir, "labels", split, rel_label)
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        with open(dst_label, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        # Write output relative paths to split manifest
        out_img_rel = os.path.join("images", split, img)
        out_lbl_rel = os.path.join("labels", split, rel_label)
        split_manifest.append([img, gid, split, h, count > 0, count, out_img_rel, out_lbl_rel])

        group_counts.add(gid)
        split_dist[split]["images"] += 1
        if count > 0:
            split_dist[split]["positive_images"] += 1
        split_dist[split]["d40_instances"] += count

        total_d40 += count
        if count == 0:
            valid_negatives += 1

    if args.audit_near_duplicates:
        audit_path = os.path.join(args.output_dir, "manifests", "near_duplicates_audit.csv")
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        with open(audit_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["img1", "split1", "ahash1", "img2", "split2", "ahash2", "hamming_dist"])
            if audit_rows:
                writer.writerows(audit_rows)
            else:
                writer.writerow(["no_near_duplicates_found", "", "", "", "", "", ""])

    write_pothole_yaml(args.output_dir)

    write_manifest_csv(args.output_dir, "source_inventory.csv", ["image", "xml", "pair_status", "validation_status", "d40_count", "reason"], source_inventory)

    # Snapshot groups used
    groups_used = [[k, v] for k, v in group_mapping.items()]
    write_manifest_csv(args.output_dir, "group_manifest_used.csv", ["image_relpath", "group_id"], groups_used)
    write_manifest_csv(args.output_dir, "exclusions.csv", ["file", "reason"], exclusions)
    write_manifest_csv(args.output_dir, "split_manifest.csv", ["image_relpath", "group_id", "split", "sha256", "is_positive", "d40_instances", "out_image", "out_label"], split_manifest)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_fingerprint": fingerprint,
        "grouping_method": grouping_method,
        "residual_leakage_limitation": limitation,
        "counts": {
            "raw_images": len(images_found),
            "raw_xmls": len(xmls_found),
            "eligible_images": len(valid_data),
            "exclusions": len(exclusions),
            "valid_negatives": valid_negatives,
            "d40_instances": total_d40,
            "groups_utilized": len(group_counts),
            "split_distribution": split_dist
        }
    }
    write_metadata_json(os.path.join(args.output_dir, "manifests"), "preparation_metadata.json", metadata)
    print("Dataset prepared successfully.")

if __name__ == "__main__":
    main()
