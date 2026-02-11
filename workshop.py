# Importing all necessary libraries and modules
from ultralytics import YOLO
from pathlib import Path
import argparse
import yaml

# Define supported image extensions for filtering
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Function to load data.yaml file and return its content as a dictionary
def load_data_yaml(data_yaml_path: Path) -> dict:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Function to resolve the dataset root directory based on the "path" key in data.yaml or default to the YAML's parent directory
def resolve_dataset_root(data_yaml_path: Path, data: dict) -> Path:
    root = data.get("path", None)
    if root:
        rootp = Path(root).expanduser()
        # IMPORTANT: if relative, resolve relative to data.yaml location
        if not rootp.is_absolute():
            rootp = (data_yaml_path.parent / rootp)
        return rootp.resolve()
    return data_yaml_path.parent.resolve()

# Function to get a sorted list of test image paths based on the "test" key in data.yaml, which can be a directory or a .txt file
def get_test_images_list(root: Path, data: dict, split_key: str = "test") -> list[Path]:
    split_value = data.get(split_key, None)
    if not split_value:
        raise ValueError(f"'{split_key}' was not found in data.yaml")

    split_raw = Path(split_value).expanduser()
    split_path = split_raw if split_raw.is_absolute() else (root / split_raw).resolve()

    # If split is a directory (e.g. test or test/images)
    if split_path.is_dir():
        # If directory has no images directly but has an "images" subfolder, use it
        candidate = split_path
        imgs = [p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

        if len(imgs) == 0 and (split_path / "images").is_dir():
            candidate = split_path / "images"
            imgs = [p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

        return sorted(imgs, key=lambda p: p.name.lower())

    # If split is a txt file with image paths
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in split_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        imgs = []
        for ln in lines:
            p = Path(ln).expanduser()
            if not p.is_absolute():
                p = (root / p).resolve()
            imgs.append(p)
        return sorted(imgs, key=lambda p: p.name.lower())

    raise ValueError(f"Test split path is neither a directory nor a .txt list: {split_path}")


# Main function to execute the validation workflow
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=-1,
                        help="How many test images to evaluate (first N sorted by filename). Use -1 for ALL.")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help='e.g. "0" or "cpu"')
    args = parser.parse_args()

    # Load data.yaml and resolve dataset root
    data_yaml_path = Path(args.data).expanduser().resolve()
    data = load_data_yaml(data_yaml_path)
    root = resolve_dataset_root(data_yaml_path, data)
    print("data.yaml:", data_yaml_path)
    print("dataset root resolved:", root)
    print("data['test']:", data.get("test"))
    
    
    # Collect & sort test images
    test_imgs = get_test_images_list(root, data, split_key="test")

    # Take first N (if requested)
    if args.n is not None and args.n > 0:
        test_imgs = test_imgs[:args.n]

    # Ensure we have test images to evaluate
    if len(test_imgs) == 0:
        raise SystemExit("No test images found for the given configuration.")

    # Write subset list file
    subset_list_path = data_yaml_path.parent / f"subset_test_{len(test_imgs)}.txt"
    subset_list_path.write_text("\n".join(str(p) for p in test_imgs) + "\n", encoding="utf-8")

    # Create a modified data.yaml for this subset
    data_subset = dict(data)
    data_subset["test"] = str(subset_list_path)

    # Write a temporary yaml file for this subset
    subset_yaml_path = data_yaml_path.parent / f"subset_data_{len(test_imgs)}.yaml"
    subset_yaml_path.write_text(
        yaml.safe_dump(data_subset, sort_keys=False),
        encoding="utf-8"
    )

    print("Using subset yaml:", subset_yaml_path)

    # Run validation with the subset yaml
    model = YOLO(args.weights)
    
    results = model.val(
        data=str(subset_yaml_path),  
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        save_txt=True,
        save_json=True
    )

    # Print validation metrics
    print("Validation Metrics (subset):")
    print(f"Images evaluated: {len(test_imgs)}")
    print(f"mAP@0.5: {results.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")
    print("*****************Program completed*****************")


if __name__ == "__main__":
    main()
