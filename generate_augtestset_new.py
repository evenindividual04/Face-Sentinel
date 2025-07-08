import os
import pickle
from pathlib import Path
import random
import cv2
from tqdm import tqdm
import albumentations as A
import pandas as pd
import warnings
import logging
import json
from joblib import Parallel, delayed

# Ignore UserWarnings from albumentations about arguments
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    filename="augmentation_errors.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s:%(message)s"
)

# Define augmentations with updated parameters for recent albumentations
AUGMENTATIONS = [
    A.ImageCompression(quality_range=(30, 50), compression_type='jpeg', p=1.0),
    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    A.GaussNoise(var_limit=(10, 50), p=1.0),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, mask_fill_value=0, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    A.Rotate(limit=15, border_mode=0, p=1.0),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
]

def load_df(df_path):
    if df_path.endswith('.pkl'):
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(df_path)
    return df

def get_file_paths(df, column):
    if column in df.columns:
        return df[column].tolist()
    elif column == df.index.name or column == 'index':
        return df.index.tolist()
    else:
        raise AssertionError(f"'{column}' not in dataframe columns or as index")

def process_and_save(
    file_path, input_dir, output_dir, augs, output_format, quality, dry_run, save_original
):
    full_path = input_dir / file_path
    save_path = output_dir / file_path
    save_path = save_path.with_suffix(f".{output_format}")

    if save_path.exists():
        return None, None  # Resume support: skip if already exists

    if not full_path.exists():
        msg = f"File not found: {full_path}"
        logging.warning(msg)
        return file_path, msg

    if dry_run:
        print(f"[DRY RUN] Would process: {full_path} -> {save_path}")
        return None, None

    image = cv2.imread(str(full_path))
    if image is None:
        msg = f"Could not read: {full_path}"
        logging.warning(msg)
        return file_path, msg

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original if requested
    if save_original and not (output_dir / file_path).exists():
        orig_save_path = output_dir / file_path
        orig_save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(orig_save_path.with_suffix(f".{output_format}")),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Apply augmentations
    composed = A.Compose(augs)
    aug_image = composed(image=image)['image']
    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == 'jpg':
        cv2.imwrite(str(save_path), aug_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(str(save_path), aug_image_bgr)

    # Return metadata for logging
    aug_names = [aug.__class__.__name__ for aug in augs]
    return str(file_path), aug_names

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory with original images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save augmented images')
    parser.add_argument('--test_df', type=str, required=True, help='DataFrame (.pkl or .csv) with image paths')
    parser.add_argument('--column', type=str, required=True, help='Column name or "index" for image paths')
    parser.add_argument('--n_aug', type=int, default=2, help='Number of random augmentations per image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (if saving as jpg)')
    parser.add_argument('--dry_run', action='store_true', help='If set, only print what would be done')
    parser.add_argument('--save_original', action='store_true', help='Also save original images in output_dir')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = load_df(args.test_df)
    file_paths = get_file_paths(df, args.column)

    print(f"Processing {len(file_paths)} images with {args.n_aug} random augmentations each...")

    # Prepare augmentations for each image (deterministic with seed)
    all_augs = [random.sample(AUGMENTATIONS, args.n_aug) for _ in range(len(file_paths))]

    # Metadata dict for augmentation logging
    metadata = {}

    # Parallel processing
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_and_save)(
            file_path, input_dir, output_dir, augs, args.format, args.quality, args.dry_run, args.save_original
        )
        for file_path, augs in tqdm(zip(file_paths, all_augs), total=len(file_paths))
    )

    # Collect and save metadata
    for file_path, aug_names in results:
        if file_path and aug_names:
            metadata[file_path] = aug_names

    if not args.dry_run:
        with open(output_dir / "augmentation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✅ Augmented images and metadata saved to: {output_dir}")

    print("Done.")

if __name__ == '__main__':
    main()
