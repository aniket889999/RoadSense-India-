import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import sys as sys_module
from src.ml.dataset_validation import validate_prepared_yolo_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    is_valid, err = validate_prepared_yolo_dataset(args.dataset)
    if not is_valid:
        print(f"Validation Error: {err}")
        sys_module.exit(1)

    print("Dataset validation passed.")

if __name__ == "__main__":
    main()
