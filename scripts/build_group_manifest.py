import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import glob
import csv
import json

def find_data(rdd_root: str):
    img_search = os.path.join(rdd_root, "**", "India", "train", "images")
    img_dirs = [p for p in glob.glob(img_search, recursive=True) if os.path.isdir(p)]

    if len(img_dirs) > 1:
        print("Multiple India/train/images roots found. Ambiguous dataset.")
        sys.exit(1)

    return img_dirs[0] if img_dirs else os.path.join(rdd_root, "India", "train", "images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-block-size", type=int, default=0)
    args = parser.parse_args()

    if args.sequence_block_size < 0:
        print("Error: sequence-block-size must be non-negative")
        sys.exit(1)

    images_dir = find_data(args.rdd_root)

    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} does not exist.")
        sys.exit(1)

    images = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                rel = os.path.relpath(os.path.join(root, f), images_dir)
                images.append(rel)

    if not images:
        print(f"Error: Directory {images_dir} contains no supported .jpg images.")
        sys.exit(1)

    images.sort()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "group_id"])

        if args.sequence_block_size > 0:
            for idx, img in enumerate(images):
                group_id = f"group_{idx // args.sequence_block_size}"
                writer.writerow([img, group_id])
        else:
            for img in images:
                writer.writerow([img, "TODO_ASSIGN_GROUP"])

    if args.sequence_block_size > 0:
        meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "grouping_quality": "sequence_proxy_not_verified_route",
                "limitation": f"Unverified sequence blocks (size {args.sequence_block_size}); may span multiple physical routes or suffer leakage.",
                "sequence_block_size": args.sequence_block_size,
                "is_proxy": True
            }, f, indent=4)

    print(f"Manifest written to {args.output}")

if __name__ == "__main__":
    main()
