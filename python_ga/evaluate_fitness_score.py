import sys
import open3d as o3d
import numpy as np

# Adjust the path so we can import ga_optimizer
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ga_optimizer import load_fragments, calculate_inliers

def main():
    export_dir = "../Dataset/SfS_pp/Export"
    
    # If not running from within python_ga dir, try another path
    if not os.path.exists(export_dir):
        export_dir = "/home/taha/Desktop/CI_PROJECT/structure-from-sherds-pp/Dataset/SfS_pp/Export"
        
    fragments = load_fragments(export_dir)
    
    total_pts = sum([len(f.base_edge.points) for f in fragments])
    print(f"\nTotal break-line points across {len(fragments)} fragments: {total_pts}")
    
    # --- Realistic Maximum Fitness Calculation ---
    # A single point on the break-line of Fragment A can physically only connect
    # to ONE other fragment's break-line (its true neighbor). 
    # For its true neighbor, if perfectly aligned:
    #   distance = 0 -> reward = 50
    #   distance < 1.5 -> bonus = 100
    #   Total = 150 points for that match.
    # 
    # For ALL OTHER fragments (the non-neighbors), the nearest point will likely be
    # > 50mm away across the pot, yielding 0 reward.
    # 
    # Therefore, the maximum realistic fitness score if the pot is perfectly 
    # assembled with zero collisions is roughly:
    # 150 * (Number of edge points)
    
    realistic_max_reward = total_pts * 150.0
    
    print(f"Realistic Max possible reward (perfect assembly): {realistic_max_reward}")

if __name__ == "__main__":
    main()
