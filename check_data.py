import numpy as np
import matplotlib.pyplot as plt
from src.dataset.loader import load_scan, get_scan_files

# PATH TO YOUR EXTRACTED DATA (Adjust if necessary)
VEL_DIR = "data/raw/velodyne_data/velodyne_sync" 

def visualize_scan():
    files = get_scan_files(VEL_DIR)
    if not files:
        print("No .bin files found! Did you extract the archive?")
        return

    # Load the first scan
    print(f"Loading: {files[0]}")
    points = load_scan(str(files[0]))
    print(f"Raw Point Cloud Shape: {points.shape}") 
    
    # --- DEBUGGING TELEMETRY ---
    print(f"Max X: {points[:, 0].max():.2f}")
    print(f"Min X: {points[:, 0].min():.2f}")

    # --- THE FILTER ---
    # Keep only points within 200 meters of the robot
    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = distances < 200  # 200 meters is plenty for a campus
    points = points[mask]

    print(f"Filtered Point Cloud Shape: {points.shape}")

    if points.shape[0] == 0:
        print("WARNING: All points were filtered out! Data reading might be wrong.")
        return

    # Simple 2D Top-Down View (X vs Y)
    plt.figure(figsize=(10, 10))
    # Plot every 5th point
    plt.scatter(points[::5, 0], points[::5, 1], s=1, c=points[::5, 3], cmap='gray') 
    plt.title(f"NCLT Scan: Top Down View (Filtered)\n{files[0].name}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualize_scan()