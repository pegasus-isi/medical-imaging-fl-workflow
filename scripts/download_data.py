#!/usr/bin/env python3
"""Download dataset from source.

For TCIA: uses tcia_utils to download DICOM series, converts middle slice to PNG.
For NIH:  uses HuggingFace datasets (streaming) for ChestX-ray14.

Outputs a single tar.gz archive of real PNG images.
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
        api_name = TCIA_COLLECTION_NAMES.get(collection_cfg_name, collection_cfg_name)
        print(f"\n=== TCIA collection: {collection_cfg_name} (API: {api_name}) ===")

        collection_dir = output_dir / collection_cfg_name
        collection_dir.mkdir(exist_ok=True)

        try:
            series_result = nbia.getSeries(collection=api_name)
        except Exception as e:
            print(f"  ERROR fetching series list: {e}")
            continue

        if series_result is None:
            print(f"  No series found for {api_name}")
            continue

        if isinstance(series_result, list):
            series_list = series_result
        elif hasattr(series_result, "to_dict"):
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

            # Find DICOM files
            dcm_files = sorted(dl_dir.rglob("*.dcm"))
            if not dcm_files:
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

            # Pick the middle slice
            mid_idx = len(dcm_files) // 2
            dcm_path = dcm_files[mid_idx]

            try:
                ds = pydicom.dcmread(dcm_path)
                arr = ds.pixel_array.astype(np.float64)

                slope = float(getattr(ds, "RescaleSlope", 1))
                intercept = float(getattr(ds, "RescaleIntercept", 0))
                arr = arr * slope + intercept

                wc_attr = getattr(ds, "WindowCenter", 0)
                ww_attr = getattr(ds, "WindowWidth", 2000)
                wc = float(wc_attr[0]) if hasattr(wc_attr, "__getitem__") and not isinstance(wc_attr, (int, float, str)) else float(wc_attr)
                ww = float(ww_attr[0]) if hasattr(ww_attr, "__getitem__") and not isinstance(ww_attr, (int, float, str)) else float(ww_attr)
                if ww == 0:
                    ww = 2000
                low = wc - ww / 2
                high = wc + ww / 2
                arr = np.clip(arr, low, high)

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

NIH_LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "No Finding", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def download_nih(output_dir: Path, max_samples: int = 2000):
    """Download NIH ChestX-ray14 via HuggingFace datasets (streaming)."""
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

    # Reverse map: label name -> index (handles datasets that return strings)
    name_to_idx = {name: i for i, name in enumerate(NIH_LABEL_NAMES)}

    label_rows = []
    count = 0

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
            for part in raw_labels.split("|"):
                part = part.strip()
                idx = name_to_idx.get(part, name_to_idx.get(part.replace(" ", "_"), -1))
                labels = [idx] if idx >= 0 else [10]
        else:
            labels = [int(raw_labels)]

        patient_id = sample.get("Patient ID", count)

        fname = f"img_{count:06d}.png"
        img_path = images_dir / fname

        img = img.resize((224, 224), Image.LANCZOS).convert("RGB")
        img.save(img_path)

        label_names = "|".join(NIH_LABEL_NAMES[i] for i in labels if 0 <= i < len(NIH_LABEL_NAMES))
        label_indices = "|".join(str(i) for i in labels)
        # Binary label: 0 = "No Finding" only, 1 = any pathology
        binary_label = 0 if labels == [10] else 1

        label_rows.append(f"{fname},{label_indices},{label_names},{binary_label},{patient_id}")
        count += 1

        if count % 200 == 0:
            print(f"  Downloaded {count}/{max_samples} images...")

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("filename,label_indices,label_names,binary_label,patient_id\n")
        f.write("\n".join(label_rows) + "\n")

    print(f"  Done: {count} images saved to {images_dir}")
    print(f"  Labels CSV: {labels_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tcia", "nih"])
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    output_dir = Path(f"{args.dataset}_raw_data")

    if args.dataset == "tcia":
        collections = ds_cfg["collections"]
        max_series = ds_cfg.get("max_series_per_collection", 50)
        download_tcia(collections, output_dir, max_series=max_series)
    else:
        max_samples = ds_cfg.get("max_samples", 2000)
        download_nih(output_dir, max_samples=max_samples)

    # Package into tar.gz
    output_tar = f"{args.dataset}_raw_data.tar.gz"
    print(f"Packaging to {output_tar}")
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(output_dir, arcname=args.dataset)

    # Cleanup
    shutil.rmtree(output_dir)
    print(f"Done. Output: {output_tar}")


if __name__ == "__main__":
    main()
