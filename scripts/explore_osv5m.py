#!/usr/bin/env python3
"""Script to explore and display sample data from OSV5M dataset."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
from PIL import Image
import json

def explore_osv5m_samples():
    """Load and display sample entries from the OSV5M dataset."""

    # Path to cached dataset
    cache_dir = Path.home() / ".cache/huggingface/datasets/osv5m"

    # Check if dataset exists
    test_csv = cache_dir / "test.csv"
    if not test_csv.exists():
        print(f"Dataset not found at {test_csv}")
        print("The dataset will be downloaded when you first run the environment.")
        return

    # Load metadata
    print("Loading OSV5M test dataset metadata...")
    df = pd.read_csv(test_csv, dtype={"id": str})

    print(f"\n{'='*80}")
    print(f"Dataset Overview:")
    print(f"{'='*80}")
    print(f"Total samples in test.csv: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    # Show first 10 samples
    print(f"\n{'='*80}")
    print(f"First 10 Sample Entries:")
    print(f"{'='*80}")

    for idx, row in df.head(10).iterrows():
        print(f"\n[Sample {idx + 1}]")
        print(f"  ID: {row['id']}")
        print(f"  Location: {row['city']}, {row['region']}, {row['country']}")
        print(f"  Coordinates: ({row['latitude']:.6f}, {row['longitude']:.6f})")
        print(f"  Sub-region: {row['sub-region'] if pd.notna(row['sub-region']) else 'N/A'}")

    # Geographic distribution
    print(f"\n{'='*80}")
    print(f"Geographic Distribution (Top 20):")
    print(f"{'='*80}")

    print("\nTop Countries:")
    country_counts = df['country'].value_counts().head(20)
    for country, count in country_counts.items():
        print(f"  {country}: {count} samples ({count/len(df)*100:.1f}%)")

    print("\nTop Regions (first 15):")
    region_counts = df['region'].value_counts().head(15)
    for region, count in region_counts.items():
        print(f"  {region}: {count} samples")

    print("\nTop Cities (first 15):")
    city_counts = df['city'].value_counts().head(15)
    for city, count in city_counts.items():
        print(f"  {city}: {count} samples")

    # Check for available images
    images_dir = cache_dir / "images" / "test"
    if images_dir.exists():
        print(f"\n{'='*80}")
        print(f"Image Files:")
        print(f"{'='*80}")

        # Check how many images are actually downloaded
        available_images = []
        for idx, row in df.head(100).iterrows():
            img_id = row["id"]
            img_path = images_dir / "00" / f"{img_id}.jpg"
            if img_path.exists():
                available_images.append((idx, row, img_path))

        print(f"Found {len(available_images)} downloaded images (checked first 100 entries)")

        if available_images:
            print("\nSample images with full metadata:")
            for idx, (df_idx, row, img_path) in enumerate(available_images[:5]):
                print(f"\n[Image Sample {idx + 1}]")
                print(f"  File: {img_path.name}")
                print(f"  Size: {img_path.stat().st_size / 1024:.1f} KB")

                # Try to get image dimensions
                try:
                    img = Image.open(img_path)
                    print(f"  Dimensions: {img.width} x {img.height}")
                    img.close()
                except Exception as e:
                    print(f"  (Could not read image: {e})")

                print(f"  Ground Truth:")
                print(f"    Country: {row['country']}")
                print(f"    Region: {row['region']}")
                print(f"    Sub-region: {row['sub-region'] if pd.notna(row['sub-region']) else 'N/A'}")
                print(f"    City: {row['city']}")
                print(f"    Coordinates: ({row['latitude']:.6f}, {row['longitude']:.6f})")
    else:
        print(f"\nNo downloaded images found at {images_dir}")
        print("Images will be downloaded when you first run the environment.")

    # Show some interesting statistics
    print(f"\n{'='*80}")
    print(f"Dataset Statistics:")
    print(f"{'='*80}")

    # Count missing values
    print("\nMissing Values:")
    for col in ['country', 'region', 'sub-region', 'city']:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")

    # Coordinate ranges
    print(f"\nCoordinate Ranges:")
    print(f"  Latitude: [{df['latitude'].min():.2f}, {df['latitude'].max():.2f}]")
    print(f"  Longitude: [{df['longitude'].min():.2f}, {df['longitude'].max():.2f}]")

    # Unique counts
    print(f"\nUnique Values:")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Regions: {df['region'].nunique()}")
    print(f"  Cities: {df['city'].nunique()}")

if __name__ == "__main__":
    explore_osv5m_samples()