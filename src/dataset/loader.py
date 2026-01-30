"""Dataset loading utilities for NCLT."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def load_scan(file_path: str) -> np.ndarray:
    """Load a single NCLT LiDAR scan from a binary file.
    
    NCLT Format:
    - x, y, z: uint16 (2 bytes), scaled by 0.005, offset -100.0
    - intensity: uint8 (1 byte)
    - label: uint8 (1 byte)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {file_path}")
    
    # Define the custom binary structure
    # <u2 = Little Endian Unsigned Short (2 bytes)
    # u1 = Unsigned Char (1 byte)
    dtype_nclt = np.dtype([
        ('x', '<u2'),
        ('y', '<u2'),
        ('z', '<u2'),
        ('i', 'u1'),
        ('l', 'u1')
    ])
    
    # Read data directly from disk using the structure
    try:
        data = np.fromfile(path, dtype=dtype_nclt)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)
    
    # Apply NCLT scaling to get meters
    # Formula: value * 0.005 - 100
    x = data['x'].astype(np.float32) * 0.005 - 100.0
    y = data['y'].astype(np.float32) * 0.005 - 100.0
    z = data['z'].astype(np.float32) * 0.005 - 100.0
    intensity = data['i'].astype(np.float32)
    
    # Stack into (N, 4) array
    points = np.stack([x, y, z, intensity], axis=1)
    
    return points

def load_ground_truth_csv(csv_path: str) -> pd.DataFrame:
    """Load the NCLT ground truth CSV."""
    # NCLT GT format: timestamp, x, y, z, roll, pitch, yaw
    # We explicitly name columns because sometimes headers are messy
    df = pd.read_csv(csv_path)
    return df

def get_scan_files(velodyne_dir: str) -> List[Path]:
    """Get all .bin files sorted by timestamp (filename)."""
    p = Path(velodyne_dir)
    # NCLT filenames are timestamps (e.g., 1326031000123456.bin)
    files = sorted(list(p.glob("*.bin")))
    return files
