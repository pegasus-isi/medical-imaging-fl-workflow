#!/usr/bin/env python3
"""Download real medical images for FL workflow (v2).

TCIA: Uses tcia_utils to download DICOM series, converts middle slice to PNG.
NIH:  Uses HuggingFace datasets (streaming) for ChestX-ray14.

Outputs a single tar.gz archive of PNG images.

Usage:
  python download_datav2.py --dataset tcia --config configs/default.yml
  python download_datav2.py --dataset nih  --config configs/default.yml
"""

import argparse
import shutil
import tarfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# TCIA download
# ---------------------------------------------------------------------------

# Map config names (hyphenated) to actual TCIA API names (some use spaces)
TCIA_COLLECTION_NAMES = {
    "NSCLC-Radiomics": "NSCLC-Radiomics",
    "TCGA-LUAD": "TCGA-LUAD",
    "LIDC-IDRI": "LIDC-IDRI",
    "NSCLC-Radiogenomics": "NSCLC Radiogenomics",
    "RIDER-Lung-CT": "RIDER Lung CT",
}


def download_tcia(collections: list, output_dir: Path, max_series: int = 50):
    """Download TCIA collections via tcia_utils, convert DICOM -> PNG."""
    from tcia_utils import nbia
    import pydicom

    output_dir.mkdir(parents=True, exist_ok=True)

    for collection_cfg_name in collections:
        # Resolve the actual API name
        api_name = TCIA_COLLECTION_NAMES.get(collection_cfg_name, collection_cfg_name)
        print(f"\n=== TCIA collection: {collection_cfg_name} (API: {api_name}) ===")

        # Use the config name (hyphenated) for directory naming
        collection_dir = output_dir / collection_cfg_name
        collection_dir.mkdir(exist_ok=True)

        # Get series list
        try:
            series_result = nbia.getSeries(collection=api_name)
        except Exception as e:
            print(f"  ERROR fetching series list: {e}")
            continue

        # getSeries returns a list of dicts
        if series_result is None:
            print(f"  No series found for {api_name}")
            continue

        if isinstance(series_result, list):
            series_list = series_result
        elif hasattr(series_result, "to_dict"):
            # DataFrame fallback
            series_list = series_result.to_dict("records")
        else:
            print(f"  Unexpected getSeries return type: {type(series_result)}")
            continue

        if not series_list:
            print(f"  No series found for {api_name}")
            continue

        total = len(series_list)
        to_download = series_list[:max_series]
        print(f"  Found {total} series, downloading up to {len(to_download)}")

        converted = 0
        for idx, series_row in enumerate(to_download):
            series_uid = series_row.get("SeriesInstanceUID", str(idx))
            short_uid = series_uid[-12:]
            print(f"  [{idx+1}/{len(to_download)}] Series ...{short_uid}", end=" ", flush=True)

            # Download DICOM files for this series into a temp dir
            dl_dir = output_dir / "_dicom_tmp"
            dl_dir.mkdir(exist_ok=True)
            try:
                nbia.downloadSeries(
                    series_data=[series_uid],
                    path=str(dl_dir),
                    input_type="list",
                )
            except Exception as e:
                print(f"DOWNLOAD FAILED: {e}")
                shutil.rmtree(dl_dir, ignore_errors=True)
                continue

            # Find all DICOM files recursively
            dcm_files = sorted(dl_dir.rglob("*.dcm"))
            if not dcm_files:
                # Some TCIA downloads don't use .dcm extension
                dcm_files = sorted(
                    f for f in dl_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() not in {
                        ".json", ".xml", ".txt", ".csv", ".html", ".log",
                    }
                )

            if not dcm_files:
                print("no DICOM files found")
                shutil.rmtree(dl_dir, ignore_errors=True)
                continue

            # Pick the middle slice (most informative for CT volumes)
            mid_idx = len(dcm_files) // 2
            dcm_path = dcm_files[mid_idx]

            try:
                ds = pydicom.dcmread(dcm_path)
                arr = ds.pixel_array.astype(np.float64)

                # Apply rescale if present (CT Hounsfield units)
                slope = float(getattr(ds, "RescaleSlope", 1))
                intercept = float(getattr(ds, "RescaleIntercept", 0))
                arr = arr * slope + intercept

                # Window/level normalization
                wc_attr = getattr(ds, "WindowCenter", 0)
                ww_attr = getattr(ds, "WindowWidth", 2000)
                # These can be a single value or a list
                wc = float(wc_attr[0]) if hasattr(wc_attr, "__getitem__") and not isinstance(wc_attr, (int, float, str)) else float(wc_attr)
                ww = float(ww_attr[0]) if hasattr(ww_attr, "__getitem__") and not isinstance(ww_attr, (int, float, str)) else float(ww_attr)
                if ww == 0:
                    ww = 2000
                low = wc - ww / 2
                high = wc + ww / 2
                arr = np.clip(arr, low, high)

                # Normalize to 0-255
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
                else:
                    arr = np.zeros_like(arr)

                img = Image.fromarray(arr.astype(np.uint8))
                img = img.resize((224, 224), Image.LANCZOS)
                img = img.convert("RGB")

                safe_name = series_uid.replace(".", "_")
                out_path = collection_dir / f"{safe_name}.png"
                img.save(out_path)
                converted += 1
                print(f"OK ({len(dcm_files)} slices, saved middle)")

            except Exception as e:
                print(f"CONVERT FAILED: {e}")

            shutil.rmtree(dl_dir, ignore_errors=True)

        print(f"  {collection_cfg_name}: {converted}/{len(to_download)} series converted to PNG")


# ---------------------------------------------------------------------------
# NIH ChestX-ray14 download via HuggingFace
# ---------------------------------------------------------------------------

# HuggingFace dataset label index -> disease name
NIH_LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "No Finding", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def download_nih_huggingface(output_dir: Path, max_samples: int = 2000):
    """Download NIH ChestX-ray14 via HuggingFace datasets (streaming).

    Uses BahaaEldin0/NIH-Chest-Xray-14 which provides:
      - image: PIL Image (1024x1024)
      - label: list of int (multi-label disease indices)
      - Patient ID: int
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"\n=== NIH ChestX-ray14 via HuggingFace (max {max_samples} samples) ===")

    ds = load_dataset(
        "BahaaEldin0/NIH-Chest-Xray-14",
        split="train",
        streaming=True,
    )

    # Build a labels CSV as we go
    label_rows = []
    count = 0

    # Reverse map: label name -> index (handles datasets that return strings)
    name_to_idx = {name: i for i, name in enumerate(NIH_LABEL_NAMES)}

    for sample in ds:
        if count >= max_samples:
            break

        img = sample["image"]
        raw_labels = sample["label"]
        # Handle both int indices and string label names
        if isinstance(raw_labels, (list, tuple)):
            labels = []
            for l in raw_labels:
                if isinstance(l, str) and not l.isdigit():
                    idx = name_to_idx.get(l, name_to_idx.get(l.replace(" ", "_"), -1))
                    if idx >= 0:
                        labels.append(idx)
                else:
                    labels.append(int(l))
        elif isinstance(raw_labels, str):
            # Single string label or pipe-delimited
            for part in raw_labels.split("|"):
                part = part.strip()
                idx = name_to_idx.get(part, name_to_idx.get(part.replace(" ", "_"), -1))
                labels = [idx] if idx >= 0 else [10]  # default to "No Finding"
        else:
            labels = [int(raw_labels)]

        patient_id = sample.get("Patient ID", count)

        fname = f"img_{count:06d}.png"
        img_path = images_dir / fname

        # Resize and save
        img = img.resize((224, 224), Image.LANCZOS).convert("RGB")
        img.save(img_path)

        # Store all labels as pipe-delimited names + raw indices
        label_names = "|".join(NIH_LABEL_NAMES[i] for i in labels if 0 <= i < len(NIH_LABEL_NAMES))
        label_indices = "|".join(str(i) for i in labels)
        # Binary label: 0 = "No Finding" only, 1 = any pathology
        # "No Finding" is index 10 in NIH_LABEL_NAMES
        binary_label = 0 if labels == [10] else 1

        label_rows.append(f"{fname},{label_indices},{label_names},{binary_label},{patient_id}")
        count += 1

        if count % 200 == 0:
            print(f"  Downloaded {count}/{max_samples} images...")

    # Write labels CSV
    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("filename,label_indices,label_names,binary_label,patient_id\n")
        f.write("\n".join(label_rows) + "\n")

    print(f"  Done: {count} images saved to {images_dir}")
    print(f"  Labels CSV: {labels_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download real medical images for FL workflow (v2)"
    )
    parser.add_argument("--dataset", required=True, choices=["tcia", "nih"])
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: <dataset>_raw_data)")
    parser.add_argument("--no-tar", action="store_true",
                        help="Skip tar.gz packaging (keep directory for inspection)")
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{args.dataset}_raw_data")

    if args.dataset == "tcia":
        collections = ds_cfg["collections"]
        max_series = ds_cfg.get("max_series_per_collection", 50)
        download_tcia(collections, output_dir, max_series=max_series)
    else:
        max_samples = ds_cfg.get("max_samples", 2000)
        source = ds_cfg.get("download_source", "huggingface")
        if source == "nih_box":
            print("nih_box source not yet implemented in v2; use huggingface")
            return
        download_nih_huggingface(output_dir, max_samples=max_samples)

    # Report what we got
    png_files = list(output_dir.rglob("*.png"))
    print(f"\n=== Summary ===")
    print(f"  Output directory: {output_dir}")
    print(f"  Total PNG files:  {len(png_files)}")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            sub_pngs = list(subdir.rglob("*.png"))
            print(f"    {subdir.name}/: {len(sub_pngs)} PNGs")

    if args.no_tar:
        print(f"\n  --no-tar: skipping tar.gz, inspect {output_dir}/ directly")
        return

    # Package into tar.gz (matches downstream expectation)
    output_tar = f"{args.dataset}_raw_data.tar.gz"
    print(f"\n  Packaging to {output_tar} ...")
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(output_dir, arcname=args.dataset)

    shutil.rmtree(output_dir)
    print(f"  Done. Output: {output_tar}")


if __name__ == "__main__":
    main()
