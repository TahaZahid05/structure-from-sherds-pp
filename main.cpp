#include <iostream>
#include "glog/logging.h"
#include <time.h>
#include <vector>
#include <fstream>
#include <algorithm>
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

	cout << "#################### Genetic Algorithm search ####################" << endl;
	GeneticAssembler ga(shard, LCS_out, SHARD_NUMBER);
	ga.Run();
	vector<Trans> T_ga = ga.GetTransforms();
	MatrixXd graph_ga = ga.GetGraph();
	vector<Trans> T_ga_eval = T_axis;
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i])
			continue;
		Matrix3d R_ga = Matrix3d::Identity();
		Vector3d t_ga = { 0, 0, 0 };
		T_ga[i].Output(R_ga, t_ga);
		T_ga_eval[i].Input(R_ga, t_ga);
	}

	pair<int, int> sherd_acc, edge_acc;
	vector<bool> right_sherd_ga(SHARD_NUMBER, true);
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i])
			right_sherd_ga[i] = false;
	}
	auto [k_sherd, t_sherd, k_edge, t_edge] = CountResult(
		GT_graph, GT_trans, graph_ga, T_ga_eval, right_sherd_ga);
	sherd_acc = { k_sherd, t_sherd };
	edge_acc = { k_edge, t_edge };
	// Note: time_ga includes preprocessing (axis alignment, LCS,
	// pruning) + GA. For pure GA time, move s_time start to just
	// before ga.Run().
	double time_ga = (clock() - s_time) / 1000.0;
	cout << "########## GA Results ##########" << endl;
	cout << "GA Sherd Accuracy : " << k_sherd << " / " << t_sherd << endl;
	cout << "GA Edge Accuracy  : " << k_edge << " / " << t_edge << endl;
	cout << "GA Time           : " << time_ga << " sec" << endl;
	cout << "################################" << endl;
	string path_result_ga = path + "Result/GA_";
	string ga_mkdir_cmd = "mkdir -p \"" + path_result_ga + "\"";
	std::system(ga_mkdir_cmd.c_str());
	SaveAcc(path_result_ga, sherd_acc, edge_acc, time_ga);

	for (int i = 0; i < SHARD_NUMBER; i++) {
		pc_origin[i].TurnOffData(viewer);
	}

	// Apply GA transforms and show result
	for (int i = 0; i < SHARD_NUMBER; i++) {
		if (!shard_on_off[i]) continue;
		Matrix3d R_vis = Matrix3d::Identity();
		Vector3d t_vis = Vector3d::Zero();
		T_ga[i].Output(R_vis, t_vis);
		pc_origin[i].UpdateData(viewer,
			shard[i].edge_line_.point_,
			shard[i].edge_line_.normal_);
		pc_origin[i].MeshTransform(R_vis, t_vis, viewer);
		pc_origin[i].AddPointCloud(viewer);
		pc_origin[i].AddMesh(viewer);
	}
	viewer->resetCamera();

	// Simple viewer loop — press q to quit
	cout << "Showing GA assembly result. Press Q to quit." << endl;
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
	}

	return 0;
}


