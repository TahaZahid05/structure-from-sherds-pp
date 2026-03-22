#include <iostream>
#include "glog/logging.h"
#include <time.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <queue>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h> 
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/impl/vtk_lib_io.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include "ceres/ceres.h"
#include "class/data_path.h"
#include "class/data_structure.h"
#include "class/visualize.h"
#include "class/reconstruction.h"
#include "class/feature_matching.h"			
#include "class/ranking_system.h"
#include "class/genetic_algorithm.h"
// test

//#define NO_RIM_INFO
#define NO_BASE_INFO

using namespace std;
using namespace Eigen;

vector<Geom> shard(SHARD_NUMBER); // DONE!
vector<Trans> GT_trans(SHARD_NUMBER);
MatrixXd GT_graph(SHARD_NUMBER, SHARD_NUMBER);

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Pot reconstruction"));

VisSwitchVariables vis; 


void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
	pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*> (nothing);
	std::string key_string = event.getKeySym();
	bool key_down = event.keyDown();
	vis.KeyEvent(key_string, key_down);
}


int main(int argc, char** argv)
{
	//#################### PCL viewer setting ####################//
	double calculation_time(0);
	int s_time(0), e_time(0);

	//#################### PCL viewer setting ####################//
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	// Register keyboard callback :
	viewer->registerKeyboardCallback(&keyboardEventOccurred, (void*)viewer.get());
	 
	cout << "#################### Pottery Data load ####################" << endl;
	//#################### Pottery Data load ####################//
	int max_breakline_points(0);
	for (int i = 0; i < SHARD_NUMBER; i++) {
		shard[i].edge_line_.ReadAxis(axis_path[i]);
		if (shard[i].edge_line_.axis_point_.empty()) {
			shard_on_off[i] = false;
			continue;
		}
		// Consider multi-axis shards at the same time
		if(shard_on_off[i])	{
			shard[i].edge_line_.ReadPCDFileWithInfo(file_path[i]);
			if (shard[i].edge_line_.point_.cols() < 50) {
				shard_on_off[i] = false;
				shard[i].edge_line_.Remove();
				continue;
			}
			shard[i].edge_line_.CalculateLineNormal();
			int breakline_points = shard[i].edge_line_.point_.cols();
			max_breakline_points = max(max_breakline_points, breakline_points);
			shard[i].LoadSurface(surface_in[i], surface_out[i], surface_fr[i]);
			shard[i].is_matching_ = true;
			shard[i].sur_frac_.CalculateLineNormal();
		}
	}

#ifdef NO_RIM_INFO 
	for (int i = 0; i < SHARD_NUMBER; i++) {
		shard[i].edge_line_.is_seg_rim_ = false;
	}
#endif

#ifdef NO_BASE_INFO
	for (int i = 0; i < SHARD_NUMBER; i++) {
		shard[i].edge_line_.is_seg_base_ = false;
	}
#endif

	cout << "#################### Ground Truth data load ####################" << endl;
	for (int i = 0; i < SHARD_NUMBER; i++) {
		for (int j = 0; j < SHARD_NUMBER; j++)
			GT_graph(i, j) = 0;
	}
	int start_index(0);
	for (int i = 0; i < NUM_MIXED_SHERD; i++) {
		MatrixXd single_graph;
		ifstream myfile(gt_graph_path[i]);
		string str;
		vector<string> fileContents;
		stringstream ss;
		while (getline(myfile, str)) {
			fileContents.push_back(str);
		}
		int num_raw = fileContents.size();
		single_graph.resize(num_raw, num_raw);
		for (int j = 0; j < num_raw; j++) {
			ss << fileContents[j];
			for (int k = 0; k < num_raw; k++) {
				ss >> single_graph(j, k);
			}
			ss.clear();
		}
		//cout << single_graph << endl;

		for (int j = start_index; j < num_raw + start_index; j++) {
			GT_trans[j].Read(gt_T_path[j]);
			for (int k = start_index; k < num_raw + start_index; k++) {
				GT_graph(j, k) = single_graph(j - start_index, k - start_index);
			}
		}
		start_index += num_raw;
	}

	//########## Remove excluded sherd information 
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) {
			for (int j = 0; j < SHARD_NUMBER; j++) {
				GT_graph(i, j) = 0;
				GT_graph(j, i) = 0;
			}
		}
	}
	cout << GT_graph << endl;

	cout << "#################### Save initial state ####################" << endl;
	//#################### Save initial state ####################//
	vector<Visualize> pc_origin(SHARD_NUMBER); 
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (shard_on_off[i]) {
			std::string pointName = "origin_" + std::to_string(i + 1);
			pc_origin[i].MakePointCloud(shard[i].edge_line_.point_, shard[i].edge_line_.normal_, pointName);
			pointName = "o_Mesh" + std::to_string(i + 1);
			pc_origin[i].MakeMesh(obj_path[i], pointName);
		}
	}

	s_time = clock();
	
	cout << "#################### Change Axis symmetrix to z axis ####################" << endl;
	//#################### Change Axis symmetrix to z axis ####################//
	vector<Trans> T_axis(SHARD_NUMBER);
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (shard[i].is_matching_) {
			Matrix3d R_d = Matrix3d::Identity();
			Vector3d t_d = { 0, 0, 0 };
			T_axis[i].Set(R_d, t_d, i + 1, i + 1);

			// Align symmetric axis to z-axis
			AxisAlignment(shard[i].edge_line_, R_d, t_d);	
			shard[i].SurMove(R_d, t_d, true);

			pc_origin[i].UpdateData(viewer, shard[i].edge_line_.point_, shard[i].edge_line_.normal_);
			pc_origin[i].AddPointCloud(viewer);
			pc_origin[i].MeshTransform(R_d, t_d, viewer);
			T_axis[i].Input(R_d, t_d);		// Save transformation matrix to z-axis

			CalculateFeatureAxisless(shard[i]);

			//######## Multi axis
			int num_axis = shard[i].edge_line_.axis_norm_.size();
			if (num_axis > 1) {
				for (int j = 1; j < num_axis; j++) {
					Matrix3d R_a, R_i;
					Vector3d t_a, t_i;
					AxisAlignment(shard[i].edge_line_, R_a, t_a, j);
					CalculateFeatureAxisless(shard[i], j);
					R_i = R_a.inverse();
					t_i = -R_i * t_a;
					shard[i].MoveWOSurface(R_i, t_i);
				}
			}
		}
	}

	cout << "#################### Feature matching ####################" << endl;
	////#################### Feature matching ####################//
	list<LCSIndex> LCS_out;
	FeatureComp(shard, LCS_out, 25, MINIMUM_NUMBER, 0);
	cout << "Total number : " << LCS_out.size() << endl;

	cout << "#################### Pairwise pruning ####################" << endl;
	PairwisePruning(shard, LCS_out);

	// Save matches computed on original axis-aligned positions
	// (LCS_out gets overwritten during iterative GA, so preserve it here)
	list<LCSIndex> LCS_original = LCS_out;

	cout << "#################### Genetic Algorithm search ####################" << endl;

	// Iterative GA parameters
	const int kMaxGAIterations = 5;
	const double kConvergenceThreshold = 5.0; // minimum fitness improvement to continue

	vector<Trans> T_ga;
	MatrixXd graph_ga;
	vector<Trans> T_ga_eval = T_axis;
	double prev_best_fitness = -1e9;
	int ga_iteration = 0;

	// Keep a copy of original axis-aligned shard positions
	// so we can reset between iterations cleanly
	vector<Geom> shard_original = shard;

	for (ga_iteration = 0; ga_iteration < kMaxGAIterations; ++ga_iteration) {
		cout << "=== GA Iteration " << ga_iteration + 1
			<< " / " << kMaxGAIterations << " ===" << endl;

		// Run GA on current match list
		GeneticAssembler ga_iter(shard, LCS_out, SHARD_NUMBER);
		ga_iter.Run();
		T_ga = ga_iter.GetTransforms();
		graph_ga = ga_iter.GetGraph();

		// Get best fitness from this run
		double current_fitness = ga_iter.GetBestFitness();
		double improvement = current_fitness - prev_best_fitness;

		// only accumulate if this iteration improved fitness
		if (current_fitness >= prev_best_fitness || ga_iteration == 0) {
			for (int i = 0; i < SHARD_NUMBER; i++) {
				if (!shard_on_off[i]) continue;
				Matrix3d R = Matrix3d::Identity();
				Vector3d t = Vector3d::Zero();
				T_ga[i].Output(R, t);
				T_ga_eval[i].Input(R, t);
			}
		}

		cout << "[GA Iter " << ga_iteration + 1 << "] "
			<< "Best fitness: " << current_fitness
			<< " (improvement: " << improvement << ")" << endl;

		// Check convergence
		if (ga_iteration > 0 && improvement < kConvergenceThreshold) {
			cout << "[GA Iter " << ga_iteration + 1
				<< "] Converged. Stopping." << endl;
			break;
		}
		prev_best_fitness = current_fitness;

		// If this is the last iteration, don't recompute matches
		if (ga_iteration == kMaxGAIterations - 1) {
			break;
		}

		// Apply GA transforms to sherds to get new assembled positions
		// Reset to original positions first, then apply new transforms
		shard = shard_original;
		for (int i = 0; i < SHARD_NUMBER; i++) {
			if (!shard_on_off[i]) continue;
			Matrix3d R = Matrix3d::Identity();
			Vector3d t = Vector3d::Zero();
			T_ga[i].Output(R, t);
			shard[i].Move(R, t, true);
		}

		// Recompute matches on newly assembled positions
		cout << "[GA Iter " << ga_iteration + 1
			<< "] Recomputing matches on assembled positions..." << endl;
		LCS_out.clear();
		FeatureComp(shard, LCS_out, 25, MINIMUM_NUMBER, 0);
		cout << "[GA Iter " << ga_iteration + 1
			<< "] New match count: " << LCS_out.size() << endl;
		PairwisePruning(shard, LCS_out);
		cout << "[GA Iter " << ga_iteration + 1
			<< "] Pruned match count: " << LCS_out.size() << endl;
	}

	cout << "GA converged after " << ga_iteration + 1
		<< " iteration(s)." << endl;

	pair<int, int> sherd_acc, edge_acc;
	vector<bool> right_sherd_ga(SHARD_NUMBER, true);
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i])
			right_sherd_ga[i] = false;
	}
	auto [k_sherd, t_sherd, k_edge, t_edge] = CountResult(
		GT_graph, GT_trans, graph_ga, T_ga_eval, right_sherd_ga);

    // ---- Post-GA Incremental ICP Refinement (replicating BAISER flow) ----
    cout << "#################### Post-GA Incremental ICP Refinement ####################" << endl;

    // Use the RankingSubgraph constructor that sets lcs_reference_ = LCS_original
    // so MakeHierarchyPriorityList can pick and group edges correctly
    RankingSubgraph graph_icp(LCS_original, SHARD_NUMBER);
    graph_icp.node_[0] = true;
    graph_icp.root_node_ = 1;
    graph_icp.ResetMatchedIndex(shard_original);

    // Work from axis-aligned original positions
    vector<Geom> shard_icp = shard_original;

    // Determine BFS placement order from graph_ga, starting from sherd 0 (root)
    vector<int> placement_order;
    vector<bool> bfs_placed(SHARD_NUMBER, false);
    bfs_placed[0] = true;
    placement_order.push_back(0);
    queue<int> bfs_queue;
    bfs_queue.push(0);
    while (!bfs_queue.empty()) {
        int curr = bfs_queue.front(); bfs_queue.pop();
        for (int j = 0; j < SHARD_NUMBER; j++) {
            if (!bfs_placed[j] && (graph_ga(curr, j) > 0 || graph_ga(j, curr) > 0)
                && shard_on_off[j]) {
                bfs_placed[j] = true;
                placement_order.push_back(j);
                bfs_queue.push(j);
            }
        }
    }

    // graph_icp.T_[k] accumulates total transform per sherd
    // Root (sherd 0) stays identity throughout

    for (int pi = 1; pi < (int)placement_order.size(); pi++) {
        int current = placement_order[pi]; // 0-indexed

        // STEP 1: Restore shard_icp to axis-aligned positions
        shard_icp = shard_original;

        // STEP 2: Re-apply accumulated transforms to all already-placed sherds
        for (int k = 0; k < SHARD_NUMBER; k++) {
            if (!graph_icp.node_[k]) continue;
            Matrix3d R_k = Matrix3d::Identity();
            Vector3d t_k = Vector3d::Zero();
            graph_icp.T_[k].Output(R_k, t_k);
            shard_icp[k].Move(R_k, t_k);
        }

        // STEP 3: Build priority list using full BAISER machinery
        // Pass graph_icp by reference inside the vector so node_ state
        // is current when MakeHierarchyPriorityList runs
        vector<RankingSubgraph> single_graph_vec;
        single_graph_vec.push_back(graph_icp);
        single_graph_vec[0].MakeHierarchyPriorityList(0, single_graph_vec);
        graph_icp = single_graph_vec[0];

        // Filter priority_list_ to only keep chunks whose edges connect
        // to the root (sherd 1). lcs.trans_ is only valid for root edges
        // since non-root sherds have been moved by ICP and their
        // precomputed transforms are stale.
        vector<Chunk> filtered_priority;
        for (auto& chunk : graph_icp.priority_list_) {
            bool has_root_edge = false;
            for (int ei : chunk.i_edge) {
                int sx = graph_icp.sub_graph_[ei].shard_x_;
                int sy = graph_icp.sub_graph_[ei].shard_y_;
                if (sx == 1 || sy == 1) {
                    has_root_edge = true;
                    break;
                }
            }
            if (has_root_edge)
                filtered_priority.push_back(chunk);
        }
        if (!filtered_priority.empty())
            graph_icp.priority_list_ = filtered_priority;

        if (graph_icp.priority_list_.empty()) {
            graph_icp.node_[current] = true;
            continue;
        }

        // STEP 4: Set priority_index_ to the chunk targeting current sherd
        // Find the chunk whose node matches current+1
        bool found = false;
        for (int ci = 0; ci < (int)graph_icp.priority_list_.size(); ci++) {
            if (graph_icp.priority_list_[ci].node == current + 1) {
                graph_icp.priority_index_ = ci;
                found = true;
                break;
            }
        }
        if (!found) {
            graph_icp.node_[current] = true;
            continue;
        }

        // STEP 5: PrepareGraphBuilding — runs TransAverage + Move
        // exactly as BAISER does in BuildState
        vector<TransHistory> history;
        PrepareGraphBuilding(shard_icp, graph_icp, history, 0);

        // STEP 6: IcpIncGraphAxis — fresh R/t per call, same as BAISER BuildState
        // Build pregraph to lock all already-placed sherds as fixed references
        // so IcpIncGraphAxis cannot move them during optimization
        RankingSubgraph pregraph_step(SHARD_NUMBER);
        for (int k = 0; k < SHARD_NUMBER; k++) {
            if (k == current) continue; // new sherd is the mover, not fixed
            if (graph_icp.node_[k])
                pregraph_step.node_[k] = true;
        }
        vector<RankingSubgraph> pregraph_vec = { pregraph_step };

        vector<Matrix3d> R_step(SHARD_NUMBER, Matrix3d::Identity());
        vector<Vector3d> t_step(SHARD_NUMBER, Vector3d::Zero());
        int inlier_step = 0;
        IcpIncGraphAxis(shard_icp, R_step, t_step, graph_icp,
                        pregraph_vec, inlier_step, true, true);

        // STEP 7: Store ICP delta into graph_icp.T_ for all placed sherds
        // same as BAISER BuildState: graph.T_[i].Input(R[i], t[i])
        for (int k = 0; k < SHARD_NUMBER; k++) {
            if (!graph_icp.node_[k]) continue;
            graph_icp.T_[k].Input(R_step[k], t_step[k]);
        }


    }

    // Final evaluation: T_axis composed with graph_icp.T_[k]
    vector<Trans> T_ga_eval_icp = T_axis;
    for (int k = 0; k < SHARD_NUMBER; k++) {
        if (!shard_on_off[k]) continue;
        if (k == 0) continue;
        Matrix3d R_k = Matrix3d::Identity();
        Vector3d t_k = Vector3d::Zero();
        graph_icp.T_[k].Output(R_k, t_k);
        T_ga_eval_icp[k].Input(R_k, t_k);
    }

    // Evaluate
    vector<bool> right_sherd_icp(SHARD_NUMBER, true);
    for (int k = 0; k < SHARD_NUMBER; k++)
        if (!shard_on_off[k]) right_sherd_icp[k] = false;

    auto [k_sherd_icp, t_sherd_icp, k_edge_icp, t_edge_icp] = CountResult(
        GT_graph, GT_trans, graph_ga, T_ga_eval_icp, right_sherd_icp);

    cout << "########## Post-ICP Results ##########" << endl;
    cout << "ICP Sherd Accuracy : " << k_sherd_icp << " / " << t_sherd_icp << endl;
    cout << "ICP Edge Accuracy  : " << k_edge_icp << " / " << t_edge_icp << endl;
    cout << "######################################" << endl;

	// Apply GA transforms to get assembled positions for IcpFine
	vector<Geom> shard_fine = shard_original;
	vector<Matrix3d> R_fine(SHARD_NUMBER, Matrix3d::Identity());
	vector<Vector3d> t_fine(SHARD_NUMBER, Vector3d::Zero());
	vector<bool> true_node_ga(SHARD_NUMBER, false);

	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) continue;
		true_node_ga[i] = true;
		Matrix3d R = Matrix3d::Identity();
		Vector3d t = Vector3d::Zero();
		T_ga_eval[i].Output(R, t);
		shard_fine[i].Move(R, t, true);
		// R_fine and t_fine stay as Identity/Zero
		// IcpFine will output the delta refinement only
	}

	MatrixXd graph_ga_fine = graph_ga;
	IcpFine(shard_fine, R_fine, t_fine, true_node_ga, graph_ga_fine);

	// Compose delta on top of T_ga_eval
	vector<Trans> T_ga_eval_fine = T_ga_eval;
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!true_node_ga[i]) continue;
		T_ga_eval_fine[i].Input(R_fine[i], t_fine[i]);
	}


	// Evaluate fine result separately
	vector<bool> right_sherd_fine(SHARD_NUMBER, true);
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) right_sherd_fine[i] = false;
	}
	// Final result summary for refined assembly
	sherd_acc = { k_sherd_icp, t_sherd_icp };
	edge_acc = { k_edge_icp, t_edge_icp };
	double time_total = (clock() - s_time) / 1000.0;

	cout << "########## Final Refined Results ##########" << endl;
	cout << "Final Sherd Accuracy : " << k_sherd_icp << " / " << t_sherd_icp << endl;
	cout << "Final Edge Accuracy  : " << k_edge_icp << " / " << t_edge_icp << endl;
	cout << "Total Runtime        : " << time_total << " sec" << endl;
	cout << "###########################################" << endl;

	string path_result_refined = path + "Result/Refined_";
	string refined_mkdir_cmd = "mkdir -p \"" + path_result_refined + "\"";
	std::system(refined_mkdir_cmd.c_str());
	SaveAcc(path_result_refined, sherd_acc, edge_acc, time_total);

	for (int i = 0; i < SHARD_NUMBER; i++) {
		pc_origin[i].TurnOffData(viewer);
	}

	// Apply ICP-refined relative transforms and show result.
	// shard_original contains axis-aligned pieces.
	// graph_icp.T_[i] contains the refinement relative to the root piece.
	// Applying T_[i] to shard_original[i] results in the final assembly.
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) continue;
		Matrix3d R_vis = Matrix3d::Identity();
		Vector3d t_vis = Vector3d::Zero();
		graph_icp.T_[i].Output(R_vis, t_vis);
		pc_origin[i].UpdateData(viewer,
			shard_original[i].edge_line_.point_,
			shard_original[i].edge_line_.normal_);
		pc_origin[i].MeshTransform(R_vis, t_vis, viewer);
		pc_origin[i].AddPointCloud(viewer);
		pc_origin[i].AddMesh(viewer);
	}
	viewer->resetCamera();

	// Simple viewer loop — press q to quit
	cout << "Showing Refined assembly result. Press Q to quit." << endl;
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
	}

	return 0;
}


