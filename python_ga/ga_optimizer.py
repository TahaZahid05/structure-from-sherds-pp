import open3d as o3d
import numpy as np
import os
import glob
from deap import base, creator, tools, algorithms
import random

class Fragment:
    def __init__(self, frag_id, edge_cloud, in_cloud, out_cloud):
        self.frag_id = frag_id
        # Raw point clouds
        self.base_edge = edge_cloud
        if self.base_edge and not self.base_edge.has_normals():
            self.base_edge.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
            
        self.base_in = in_cloud
        self.base_out = out_cloud
        
        # Working copies for transforms
        self.edge = type(self.base_edge)(self.base_edge)
        self.sur_in = type(self.base_in)(self.base_in) if self.base_in else None
        self.sur_out = type(self.base_out)(self.base_out) if self.base_out else None
        
    def get_obb(self):
        """Get an Oriented Bounding Box of the inner surface for fast overlap checks"""
        if self.sur_in:
            return self.sur_in.get_oriented_bounding_box()
        return None

def load_fragments(export_dir):
    fragments = []
    # Count the number of unique fragments by looking at edge files
    edge_files = glob.glob(os.path.join(export_dir, "*_edge.xyz"))
    
    for i in range(1, len(edge_files) + 1):
        base_name = os.path.join(export_dir, f"shard_{i}")
        
        edge_path = f"{base_name}_edge.xyz"
        in_path = f"{base_name}_in.xyz"
        out_path = f"{base_name}_out.xyz"
        
        if not os.path.exists(edge_path):
            continue
            
        edge_pc = o3d.io.read_point_cloud(edge_path)
        
        in_pc = None
        if os.path.exists(in_path):
            in_pc = o3d.io.read_point_cloud(in_path)
            
        out_pc = None
        if os.path.exists(out_path):
            out_pc = o3d.io.read_point_cloud(out_path)
            
        fragments.append(Fragment(i, edge_pc, in_pc, out_pc))
        print(f"Loaded Fragment {i}: Edge pts: {len(edge_pc.points)}")
        
    return fragments

def get_z_rotation_matrix(theta):
    """Returns a 4x4 transformation matrix for a rotation around Z-axis."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0, 0],
        [sin_t,  cos_t, 0, 0],
        [    0,      0, 1, 0],
        [    0,      0, 0, 1]
    ])

def get_z_translation_matrix(z):
    """Returns a 4x4 transformation matrix for a translation along Z-axis."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def apply_chromosome_transforms(fragments, individual):
    """
    Applies the (theta, z) transforms from the GA chromosome to the fragments.
    Fragment 0 is considered the base and remains fixed at origin.
    The chromosome contains [theta1, z1, theta2, z2, ..., thetaN, zN]
    """
    # Reset all fragment copies back to their original state first
    for frag in fragments:
        frag.edge.points = frag.base_edge.points
        if frag.sur_in:
            frag.sur_in.points = frag.base_in.points
        if frag.sur_out:
            frag.sur_out.points = frag.base_out.points

    # Fragment 0 doesn't move. Move Fragments 1 to N.
    for i in range(1, len(fragments)):
        # Calculate indices in the chromosome
        idx = (i - 1) * 2
        theta = individual[idx]
        z = individual[idx + 1]
        
        T_rot = get_z_rotation_matrix(theta)
        T_trans = get_z_translation_matrix(z)
        T_final = T_trans @ T_rot
        
        # Apply transforms using Open3D's built-in fast matrix multiplication
        fragments[i].edge.transform(T_final)
        if fragments[i].sur_in:
            fragments[i].sur_in.transform(T_final)
        if fragments[i].sur_out:
            fragments[i].sur_out.transform(T_final)

def check_overlap(fragments, overlap_dist_threshold=2.0):
    """
    Volumetric penetration check using Oriented Bounding Boxes (OBB) 
    and fast point cloud distance queries between inner and outer surfaces.
    """
    collision_penalty = 0
    for i in range(len(fragments)):
        obb_i = fragments[i].get_obb()
        if not obb_i: continue
            
        for j in range(i + 1, len(fragments)):
            obb_j = fragments[j].get_obb()
            if not obb_j: continue
            
            # 1. Fast preliminary check: Do the OBBs even intersect/come close?
            # If the centers of the bounding boxes are far apart, skip detailed check
            dist_centers = np.linalg.norm((obb_i.center - obb_j.center))
            if dist_centers > (np.max(obb_i.extent) + np.max(obb_j.extent)):
                continue
                
            # 2. Detailed check: Point-to-point distance between surfaces
            if fragments[i].sur_in and fragments[j].sur_in:
                # Calculate minimum distances from points in fragment i to fragment j
                # sur_in points should NOT heavily overlap perfectly over the face.
                # However, near the break-lines, they will naturally be close (< 2mm). 
                # So we only penalize if a MASSIVE amount of points are colliding, indicating true penetration.
                distances = fragments[i].sur_in.compute_point_cloud_distance(fragments[j].sur_in)
                overlap_pts = np.sum(np.asarray(distances) < overlap_dist_threshold)
                
                if overlap_pts > 30: # Lower threshold to catch thinner clips
                    collision_penalty += (overlap_pts * 100) # Massive multiplier: overlapping is strictly forbidden
                
    return collision_penalty

def calculate_inliers(fragments, angle_th=0.523): # ~30 degrees
    """
    Provides a continuous reward gradient for edge adjacency to guide the GA.
    Matches the C++ logic: strictly enforces that normals must align (angle < threshold)
    before giving distance-based inlier rewards.
    """
    total_score = 0
    
    # To ensure pieces don't form disconnected clusters (floating in space), we track connectivity
    adjacency_graph = {i: [] for i in range(len(fragments))}
    
    for i in range(len(fragments)):
        edge_i = fragments[i].edge
        if not edge_i or len(edge_i.points) == 0: continue
            
        for j in range(i + 1, len(fragments)):
            edge_j = fragments[j].edge
            if not edge_j or len(edge_j.points) == 0: continue
            
            # Sub-function to calculate unidirectional distance from source to target
            def calc_directed_score(source_pc, target_pc):
                pcd_tree = o3d.geometry.KDTreeFlann(target_pc)
                pts_src = np.asarray(source_pc.points)
                normals_src = np.asarray(source_pc.normals)
                normals_tgt = np.asarray(target_pc.normals)
                
                valid_dists = []
                for pt_idx in range(len(pts_src)):
                    [k, idx, dist_squared] = pcd_tree.search_knn_vector_3d(pts_src[pt_idx], 1)
                    if k == 0: continue
                    
                    dist = np.sqrt(dist_squared[0])
                    n_a = normals_src[pt_idx]
                    n_b = normals_tgt[idx[0]]
                    
                    dot_prod = np.clip(np.dot(n_a, n_b), -1.0, 1.0)
                    angle = np.arccos(dot_prod)
                    angle = min(angle, np.pi - angle)
                    
                    if angle < angle_th:
                        valid_dists.append(dist)
                        
                score_increment = 0
                inliers_count = 0
                if len(valid_dists) > 0:
                    valid_dists = np.array(valid_dists)
                    continuous_reward = np.sum(np.maximum(0, 50.0 - valid_dists))
                    inliers_count = np.sum(valid_dists < 1.5) 
                    inlier_bonus = inliers_count * 100.0
                    score_increment = continuous_reward + inlier_bonus
                
                return score_increment, inliers_count
                
            # Calculate symmetrically: i -> j AND j -> i
            score_ij, inliers_ij = calc_directed_score(edge_i, edge_j)
            score_ji, inliers_ji = calc_directed_score(edge_j, edge_i)
            
            total_score += (score_ij + score_ji)
                
            # Mark these two fragments as connected if they share at least 5 inlier points symmetrically
            if (inliers_ij + inliers_ji) > 5:
                adjacency_graph[i].append(j)
                adjacency_graph[j].append(i)
            
    return total_score, adjacency_graph

def check_connected_components(adjacency_graph, fragments):
    """
    Checks if all fragments are connected to each other (form a single connected component).
    If they are split into separate floating islands, we apply a continuous penalty based
    on the distance of disconnected fragments to the base fragment.
    """
    num_fragments = len(fragments)
    if num_fragments <= 1:
        return 0
        
    visited = set()
    
    # Simple BFS/DFS to find all connected nodes starting from fragment 0
    def dfs(node):
        visited.add(node)
        for neighbor in adjacency_graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
                
    dfs(0)
    
    penalty = 0
    # For any fragment not connected to fragment 0, penalize based on distance to fragment 0
    if len(visited) < num_fragments:
        center_0 = fragments[0].get_obb().center if fragments[0].get_obb() else np.zeros(3)
        for i in range(num_fragments):
            if i not in visited:
                center_i = fragments[i].get_obb().center if fragments[i].get_obb() else np.zeros(3)
                dist = np.linalg.norm(center_i - center_0)
                # Apply a very gentle continuous penalty so pieces don't float into the void,
                # but NOT so strong that it acts like a black hole pulling them into the center instead of edges.
                penalty += dist * 5.0
                
    return penalty


# =========================================================
# DEAP Genetic Algorithm Setup
# =========================================================

# Fitness: maximize inliers, minimize (penalize) overlap
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate_fitness(individual, fragments):
    """
    The core fitness function for the GA.
    A valid individual maximizes edge adjacency while minimizing volumetric overlap.
    """
    # 1. Apply the 2-DoF transforms to the clouds based on genes
    apply_chromosome_transforms(fragments, individual)
    
    # 2. Check for physical collisions (Penetration Penalty)
    overlap_penalty = check_overlap(fragments, overlap_dist_threshold=2.0)
    
    # 3. Calculate distance reward (Inliers Reward) and Adjacency Graph
    # This now returns a continuous score rather than just a binary count,
    # solving the issue where max fitness gets stuck at 0.0.
    adjacency_score, adjacency_graph = calculate_inliers(fragments)
    
    # 4. Connected Components Penalty
    # Ensure all pieces form a single contiguous pot, not floating islands
    disconnected_penalty = check_connected_components(adjacency_graph, fragments)
    
    # Calculate final fitness score. Overlap and Disconnections serve as massive penalties.
    score = adjacency_score - overlap_penalty - disconnected_penalty
    
    # Return as a tuple (DEAP requirement)
    return (score,)

def setup_toolbox(num_genes, fragments):
    """
    Configures the GA toolbox with genetic operators.
    Each variable is a continuous float representing either a rotation (theta) or translation (z).
    """
    toolbox = base.Toolbox()
    
    # Gene generators
    # Theta: random angle between 0 and 2*PI radians
    toolbox.register("attr_theta", random.uniform, 0, 2 * np.pi)
    # Z translation: random value between -50 and 50 mm (tighter bound around alignment)
    toolbox.register("attr_z", random.uniform, -50, 50)
    
    # Initialize individual with alternating theta and z values
    def init_individual(icls, n_frags):
        genes = []
        for _ in range(n_frags):
            genes.append(toolbox.attr_theta())
            genes.append(toolbox.attr_z())
        return icls(genes)
        
    toolbox.register("individual", init_individual, creator.Individual, num_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators suitable for continuous variables
    toolbox.register("evaluate", evaluate_fitness, fragments=fragments)
    
    # Custom pair-wise crossover to avoid destroying rigid-body relationships
    def cx_pairs(ind1, ind2):
        for i in range(0, len(ind1), 2):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
                ind1[i+1], ind2[i+1] = ind2[i+1], ind1[i+1]
        return ind1, ind2
        
    toolbox.register("mate", cx_pairs) 
    # Gaussian mutation - increased sigma to allow jumping out of local minima
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2.0, indpb=0.3)
    # Tournament selection
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def main():
    print("Loading fragments...")
    export_dir = "../Dataset/SfS_pp/Export"
    
    # If not running from within python_ga dir, try another path
    if not os.path.exists(export_dir):
        export_dir = "/home/taha/Desktop/CI_PROJECT/structure-from-sherds-pp/Dataset/SfS_pp/Export"
        
    fragments = load_fragments(export_dir)
    num_fragments = len(fragments)
    
    if num_fragments < 2:
        print("Need at least 2 fragments to run GA.")
        return
        
    print(f"Loaded {num_fragments} fragments.")
    
    # Number of mobile fragments is N-1 (Fragment 0 is fixed)
    num_mobile = num_fragments - 1
    toolbox = setup_toolbox(num_mobile, fragments)
    
    # GA Parameters
    pop_size = 200
    num_generations = 30
    crossover_prob = 0.7
    mutation_prob = 0.2
    
    print(f"Initializing GA Population of {pop_size}...")
    pop = toolbox.population(n=pop_size)
    
    # Keep track of the best individuals
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Starting Evolution...")
    # Run the Simple GA
    # We pass the fragments implicitly bound in the evaluate function arguments
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, 
                                   ngen=num_generations, stats=stats, halloffame=hof, verbose=True)
                                   
    best_ind = hof[0]
    print(f"\nBest Individual Fitness: {best_ind.fitness.values[0]}")
    print(f"Best Chromosome (theta, z pairs): {best_ind}")
    
    # Apply the best transform to visualize
    print("\nVisualizing Best Assembly...")
    apply_chromosome_transforms(fragments, best_ind)
    
    # Gather point clouds for visualization
    vis_clouds = []
    # Use distinct colors for each fragment
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    
    for i, frag in enumerate(fragments):
        c = colors[i % len(colors)]
        if frag.edge:
            frag.edge.paint_uniform_color(c)
            vis_clouds.append(frag.edge)
        if frag.sur_in:
            frag.sur_in.paint_uniform_color([c[0]*0.5, c[1]*0.5, c[2]*0.5]) # Darker color for surface
            vis_clouds.append(frag.sur_in)
            
    o3d.visualization.draw_geometries(vis_clouds, window_name="GA Reconstruction Result")

if __name__ == "__main__":
    main()
