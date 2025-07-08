#!/usr/bin/env python3
"""
Index Celeb-DF v2
Image and Sound Processing Lab - Politecnico di Milano
Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import os
os.environ['TORCHVISION_IO_DISABLE_IMAGE_EXTENSION'] = '1'  # Suppress libjpeg warning

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
import numpy as np

# Import with fallback for Apple Silicon compatibility
try:
    from isplutils.utils import extract_meta_av, extract_meta_cv
except ImportError:
    print("isplutils not found. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "git+https://github.com/polimi-ispl/deepfake_detection_utils.git"])
    from isplutils.utils import extract_meta_av, extract_meta_cv

def main():
    parser = argparse.ArgumentParser(
        description='Index Celeb-DF v2 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source', type=Path, required=True,
                        help='Root directory of Celeb-DF v2 dataset')
    parser.add_argument('--videodataset', type=Path, default='index/celebdf_videos.pkl',
                        help='Output path for video DataFrame')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help='Number of parallel workers')
    
    args = parser.parse_args()

    # Validate source directory
    source_dir = args.source.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create output directory if needed
    args.videodataset.parent.mkdir(parents=True, exist_ok=True)

    # Load or create DataFrame
    if args.videodataset.exists():
        print(f'Loading existing video DataFrame from {args.videodataset}')
        df_videos = pd.read_pickle(args.videodataset)
    else:
        print('Creating new video DataFrame')
        
        # Load test split
        test_file = source_dir / 'List_of_testing_videos.txt'
        if not test_file.exists():
            raise FileNotFoundError(f'Test split file not found: {test_file}. '
                                   'Download from: https://github.com/yuezunli/celeb-deepfakeforensics')
        test_videos = pd.read_csv(test_file, delim_whitespace=True, header=0, index_col=1)
        
        # Collect video paths
        video_paths = list(source_dir.rglob('*.mp4'))
        if not video_paths:
            raise FileNotFoundError(f"No MP4 videos found in {source_dir}")
            
        df_videos = pd.DataFrame({
            'path': [p.relative_to(source_dir) for p in video_paths],
            'height': 0,
            'width': 0,
            'frames': 0,
            'label': False,
            'test': False,
            'original': -1
        }).astype({
            'height': np.uint16,
            'width': np.uint16,
            'frames': np.uint16,
            'original': np.int16
        })
        
        # Extract metadata in parallel
        print(f'Extracting metadata using {args.workers} workers...')
        with Pool(args.workers) as pool:
            meta_data = pool.starmap(extract_meta, 
                                    [(source_dir, p) for p in df_videos['path']])
        
        # Assign metadata
        df_videos[['height', 'width', 'frames']] = meta_data
        
        # Set labels
        df_videos['class'] = df_videos['path'].apply(lambda x: x.parts[0])
        df_videos['label'] = df_videos['class'].eq('Celeb-synthesis')  # True=Fake, False=Real
        df_videos['name'] = df_videos['path'].apply(lambda x: x.stem)
        
        # Map originals for fake videos
        real_videos = df_videos[~df_videos['label']]
        fake_mask = df_videos['label']
        
        def find_original(name):
            prefix = name.split('_')[0]
            matches = real_videos[real_videos['name'].str.startswith(prefix)]
            return matches.index[0] if not matches.empty else -1
        
        df_videos.loc[fake_mask, 'original'] = df_videos[fake_mask]['name'].apply(find_original)
        
        # Mark test videos
        df_videos['test'] = df_videos['path'].astype(str).isin(test_videos.index)
        
        # Save results
        print(f'Saving video DataFrame to {args.videodataset}')
        df_videos.to_pickle(args.videodataset)

    # Print summary
    print(f"\n{'='*40}\nDataset Summary\n{'='*40}")
    print(f"Total videos: {len(df_videos)}")
    print(f"Real videos: {sum(~df_videos['label'])}")
    print(f"Fake videos: {sum(df_videos['label'])}")
    print(f"Test videos: {sum(df_videos['test'])}")
    print(f"Videos with metadata issues: {sum((df_videos['height'] == 0) | (df_videos['width'] == 0))}")
    print('='*40)

def extract_meta(source_dir: Path, rel_path: Path) -> tuple:
    """Extract video metadata with fallback handling"""
    abs_path = source_dir / rel_path
    try:
        # Try with PyAV first
        return extract_meta_av(str(abs_path))
    except Exception as e:
        # Fallback to OpenCV
        try:
            print(f"AV failed for {rel_path}: {str(e)[:50]}... Using OpenCV fallback")
            return extract_meta_cv(str(abs_path))
        except Exception as cv_e:
            print(f"Both methods failed for {rel_path}: {str(cv_e)[:50]}... Returning zeros")
            return (0, 0, 0)

if __name__ == '__main__':
    main()
