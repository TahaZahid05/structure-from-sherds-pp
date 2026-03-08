#include "genetic_algorithm.h"
#include "../class/reconstruction.h"
#include <iostream>
#include <algorithm>
#include <numeric>

GeneticAlgorithm::GeneticAlgorithm(int pop_size, int max_generations, int tournament_size, 
                                   double crossover_prob, double swap_mut_prob, double rev_mut_prob, 
                                   int elitism_count, int candidate_limit,
                                   vector<Geom>& shard, list<LCSIndex>& LCS_out, 
                                   int step_size, const string& log_path)
    : pop_size_(pop_size), max_generations_(max_generations), tournament_size_(tournament_size),
      crossover_prob_(crossover_prob), swap_mut_prob_(swap_mut_prob), rev_mut_prob_(rev_mut_prob),
      elitism_count_(elitism_count), candidate_limit_(candidate_limit), 
      num_shards_(shard.size()), step_size_(step_size), log_path_(log_path), lcs_out_(LCS_out)
{
    for (int i = 0; i < num_shards_; ++i) {
        shard_.push_back(shard[i]);
    }
    match_scores_ = MatrixXd::Zero(num_shards_, num_shards_);
    CalculatePairwiseMatchScores();
}

void GeneticAlgorithm::CalculatePairwiseMatchScores() {
    for (auto& lcs : lcs_out_) {
        int u = lcs.shard_x_ - 1;
        int v = lcs.shard_y_ - 1;
        if (u >= 0 && u < num_shards_ && v >= 0 && v < num_shards_) {
            match_scores_(u, v) += lcs.inliner_;
            match_scores_(v, u) += lcs.inliner_;
        }
    }
}

void GeneticAlgorithm::InitializePopulation() {
    cout << "  InitializePopulation: Starting... (num_shards=" << num_shards_ << ", pop_size=" << pop_size_ << ")" << endl;
    population_.clear();
    std::random_device rd;
    std::mt19937 g(rd());
    
    int num_heuristic = static_cast<int>(pop_size_ * 0.7);
    int num_random = pop_size_ - num_heuristic;

    cout << "  InitializePopulation: Building " << num_heuristic << " heuristic individuals..." << endl;
    // Build heuristic chromosomes
    for (int i = 0; i < num_heuristic; ++i) {
        cout << "    Heuristic " << i << "/" << num_heuristic << "..." << flush;
        Chromosome chrom;
        vector<int> candidates(num_shards_);
        iota(candidates.begin(), candidates.end(), 0);
        
        // Randomly pick a starting root to maintain some diversity even in heuristics
        std::uniform_int_distribution<int> root_dist(0, num_shards_ - 1);
        int root = root_dist(g);
        chrom.sequence_.push_back(root);
        candidates.erase(remove(candidates.begin(), candidates.end(), root), candidates.end());

        while (!candidates.empty()) {
            // Find the candidate with highest connectivity to already selected nodes
            int best_candidate = candidates[0];
            double max_score = -1.0;
            
            for (int curr : candidates) {
                double score = 0;
                for (int selected : chrom.sequence_) {
                    if (curr >= 0 && curr < num_shards_ && selected >= 0 && selected < num_shards_) {
                        score += match_scores_(curr, selected);
                    }
                }
                // Add tiny random noise to score to break ties and ensure diversity
                std::uniform_real_distribution<double> noise(0.0, 1.0);
                score += noise(g);
                
                if (score > max_score) {
                    max_score = score;
                    best_candidate = curr;
                }
            }
            chrom.sequence_.push_back(best_candidate);
            candidates.erase(remove(candidates.begin(), candidates.end(), best_candidate), candidates.end());
        }
        population_.push_back(chrom);
        cout << " done." << endl;
    }

    cout << "  InitializePopulation: Building " << num_random << " random individuals..." << endl;
    // Build random chromosomes
    for (int i = 0; i < num_random; ++i) {
        cout << "    Random " << i << "/" << num_random << "..." << flush;
        Chromosome chrom;
        chrom.sequence_.resize(num_shards_);
        iota(chrom.sequence_.begin(), chrom.sequence_.end(), 0);
        shuffle(chrom.sequence_.begin(), chrom.sequence_.end(), g);
        population_.push_back(chrom);
        cout << " done." << endl;
    }
    cout << "  InitializePopulation: Finished." << endl;
}

void GeneticAlgorithm::EvaluateChromosome(Chromosome& chrom) {
    cout << "    EvaluateChromosome: Sequence [";
    for (int gene : chrom.sequence_) cout << gene << " ";
    cout << "] Starting..." << endl;

    // We must work on a local copy of shards to avoid modifying base data for other chromosomes
    vector<Geom> shards_local = shard_;
    
    State current_state(num_shards_);
    int root = chrom.sequence_[0]; // 0-indexed
    current_state.true_node_[root] = true;

    RankingSubgraph graph(lcs_out_, num_shards_);
    graph.root_node_ = root + 1; // 1-indexed
    graph.node_[root] = true;
    current_state.graph_.push_back(graph);
    
    // CRITICAL: Initialize matched_index_ to avoid segfault in FillMatchedPoints (called by ICP)
    current_state.graph_[0].ResetMatchedIndex(shards_local);
    
    // Add history for root
    InputHistory(current_state.history_, root + 1, 0, Matrix3d::Identity(), Vector3d::Zero());

    double total_penalty = 0.0;
    
    // Evaluate sequence
    for (int i = 1; i < num_shards_; ++i) {
        int c_node = chrom.sequence_[i]; // 0-indexed
        cout << "      Step " << i << ": Adding shard " << c_node << "..." << flush;
        
        // Find best edges connecting c_node to existing subgraph
        vector<LCSIndex> connecting_edges;
        for (auto& lcs : lcs_out_) {
            int u = lcs.shard_x_ - 1;
            int v = lcs.shard_y_ - 1;
            if ((u == c_node && current_state.graph_[0].node_[v]) || 
                (v == c_node && current_state.graph_[0].node_[u])) {
                connecting_edges.push_back(lcs);
            }
        }

        // Sort by inliers descending
        sort(connecting_edges.begin(), connecting_edges.end(), [](const LCSIndex& a, const LCSIndex& b){
            return a.inliner_ > b.inliner_;
        });

        // Limit the number of candidates 
        if(connecting_edges.size() > candidate_limit_) {
             connecting_edges.resize(candidate_limit_);
        }

        // Compute initial pose
        Matrix3d R_p = Matrix3d::Identity();
        Vector3d t_p = Vector3d::Zero();
        if (!connecting_edges.empty()) {
            // Get the relative pose using TransAverage
            Matrix3d R_rel; Vector3d t_rel;
            TransAverage(c_node + 1, connecting_edges, R_rel, t_rel);
            
            // Get current pose of the best neighbor to anchor correctly
            int neighbor_idx = (connecting_edges[0].shard_x_ == c_node + 1) ? 
                              (connecting_edges[0].shard_y_ - 1) : (connecting_edges[0].shard_x_ - 1);
            
            Matrix3d R_n; Vector3d t_n;
            current_state.graph_[0].T_[neighbor_idx].Output(R_n, t_n);
            
            // T_p = T_n * T_rel (World pose of the new shard)
            R_p = R_n * R_rel;
            t_p = R_n * t_rel + t_n;
        } else {
             total_penalty += 500.0;
        }

        // Apply initial pose to local shard for correspondence search
        shards_local[c_node].Move(R_p, t_p);
        current_state.graph_[0].T_[c_node].Input(R_p, t_p);
        InputHistory(current_state.history_, c_node + 1, 0, R_p, t_p);

        // Update tracking
        current_state.graph_[0].node_[c_node] = true;
        current_state.true_node_[c_node] = true;

        // Populate priority_list_ for this node so graph.EdgeOut() works inside ICP
        Chunk chunk;
        chunk.node = c_node + 1;
        for (size_t k = 0; k < current_state.graph_[0].sub_graph_.size(); ++k) {
             const auto& lcs = current_state.graph_[0].sub_graph_[k];
             int u = lcs.shard_x_ - 1;
             int v = lcs.shard_y_ - 1;
             if ((u == c_node && current_state.graph_[0].node_[v]) ||
                 (v == c_node && current_state.graph_[0].node_[u])) {
                 chunk.i_edge.push_back(k);
             }
        }
        current_state.graph_[0].priority_list_.clear();
        current_state.graph_[0].priority_list_.push_back(chunk);
        current_state.graph_[0].priority_index_ = 0;

        for (auto& edge : connecting_edges) {
             current_state.graph_[0].edge_.push_back(edge);
             current_state.graph_[0].simple_graph_(edge.shard_x_ - 1, edge.shard_y_ - 1) = 1;
             current_state.graph_[0].simple_graph_(edge.shard_y_ - 1, edge.shard_x_ - 1) = 1;
        }

        // Local Refinement (ICP)
        // CRITICAL: Initialize R_icp and t_icp from CURRENT T_ so UpdateTrans accumulates correctly
        vector<Matrix3d> R_icp(num_shards_);
        vector<Vector3d> t_icp(num_shards_);
        for(int j=0; j<num_shards_; ++j) {
            current_state.graph_[0].T_[j].Output(R_icp[j], t_icp[j]);
        }

        vector<RankingSubgraph> pregraph;
        // Fix all pieces currently in the graph except the one we just added? 
        // Actually, BAISER fixes everything from the "pregraph".
        RankingSubgraph anchor_graph = current_state.graph_[0];
        anchor_graph.node_[c_node] = false; // Don't fix the newly added node
        pregraph.push_back(anchor_graph);
        
        bool rim_restrain = true; 
        bool axis_restrain = true;

        cout << " (ICP)..." << flush;
        IcpIncGraphAxis(shards_local, R_icp, t_icp, current_state.graph_[0], pregraph, current_state.graph_[0].graph_score_, rim_restrain, axis_restrain);
        
        // Update transforms from ICP - R_icp now contains the cumulative world pose
        for (int j = 0; j < num_shards_; j++) {
            if (current_state.graph_[0].node_[j]) {
                current_state.graph_[0].T_[j].Input(R_icp[j], t_icp[j]);
                InputHistory(current_state.history_, j + 1, 0, R_icp[j], t_icp[j]);
            }
        }
        
        // Optional: Global Refinement (Axis alignment)
        GraphAxisRefinement(shards_local, current_state.graph_[0], current_state.history_, 0);

        // Verification
        bool is_valid = CheckGraphPlausibility(shards_local, current_state.graph_[0], log_path_, 0, 0, 0, -1);
        if (!is_valid || current_state.graph_[0].graph_score_ <= 0) {
            total_penalty += 1000.0;
            cout << " [Fail: score=" << current_state.graph_[0].graph_score_ << "]" << flush;
        } else {
            cout << " [Pass: score=" << current_state.graph_[0].graph_score_ << "]" << flush;
        }
        cout << " done." << endl;
    }

    current_state.UpdateMatchedMatrix(shards_local);
    current_state.UpdateStateScore();

    chrom.state_ = current_state;
    chrom.fitness_ = current_state.state_score_ - total_penalty;
    chrom.penalty_ = total_penalty;
    
    cout << "    EvaluateChromosome: Finished. Fitness: " << chrom.fitness_ << endl;
}

void GeneticAlgorithm::TournamentSelection(const vector<Chromosome>& population, vector<Chromosome>& selected) {
    selected.clear();
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dist(0, population.size() - 1);

    for (int i = 0; i < pop_size_; ++i) {
        int best_idx = dist(g);
        for (int j = 1; j < tournament_size_; ++j) {
            int contender_idx = dist(g);
            // Higher is better
            if (population[contender_idx] < population[best_idx]) { 
                best_idx = contender_idx;
            }
        }
        selected.push_back(population[best_idx]);
    }
}

void GeneticAlgorithm::OrderCrossover(Chromosome& parent1, Chromosome& parent2) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    
    if (p_dist(g) > crossover_prob_) return;

    std::uniform_int_distribution<int> cut_dist(0, num_shards_ - 1);
    int cut1 = cut_dist(g);
    int cut2 = cut_dist(g);
    if (cut1 > cut2) swap(cut1, cut2);

    vector<int> child1(num_shards_, -1);
    vector<int> child2(num_shards_, -1);

    // Copy middle segment
    for (int i = cut1; i <= cut2; ++i) {
        child1[i] = parent1.sequence_[i];
        child2[i] = parent2.sequence_[i];
    }

    // Fill remaining from the other parent
    auto fill_remaining = [&](vector<int>& child, const vector<int>& other_parent) {
        int current_pos = (cut2 + 1) % num_shards_;
        for (int i = 0; i < num_shards_; ++i) {
            int candidate = other_parent[(cut2 + 1 + i) % num_shards_];
            if (find(child.begin(), child.end(), candidate) == child.end()) {
                child[current_pos] = candidate;
                current_pos = (current_pos + 1) % num_shards_;
            }
        }
    };

    fill_remaining(child1, parent2.sequence_);
    fill_remaining(child2, parent1.sequence_);

    parent1.sequence_ = child1;
    parent2.sequence_ = child2;
}

void GeneticAlgorithm::SwapMutation(Chromosome& chrom) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    
    if (p_dist(g) < swap_mut_prob_) {
        std::uniform_int_distribution<int> pos_dist(0, num_shards_ - 1);
        int p1 = pos_dist(g);
        int p2 = pos_dist(g);
        swap(chrom.sequence_[p1], chrom.sequence_[p2]);
    }
}

void GeneticAlgorithm::SegmentReversalMutation(Chromosome& chrom) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    
    if (p_dist(g) < rev_mut_prob_) {
        std::uniform_int_distribution<int> pos_dist(0, num_shards_ - 1);
        int p1 = pos_dist(g);
        int p2 = pos_dist(g);
        if (p1 > p2) swap(p1, p2);
        reverse(chrom.sequence_.begin() + p1, chrom.sequence_.begin() + p2 + 1);
    }
}

void GeneticAlgorithm::Run() {
    cout << "Initializing Genetic Algorithm Population..." << endl;
    InitializePopulation();
    
    for (int i = 0; i < pop_size_; ++i) {
        cout << "  Evaluating initial individual " << i << "/" << pop_size_ << "..." << endl;
        EvaluateChromosome(population_[i]);
    }
    sort(population_.begin(), population_.end());
    
    double best_fitness = population_[0].fitness_ - population_[0].penalty_;
    int stagnant_gens = 0;

    for (int gen = 0; gen < max_generations_; ++gen) {
        cout << "Generation " << gen + 1 << "/" << max_generations_ << " - Best Fitness: " << best_fitness << endl;
        
        vector<Chromosome> selected;
        TournamentSelection(population_, selected);

        vector<Chromosome> next_generation;
        
        // Elitism
        for(int i = 0; i < elitism_count_; ++i) {
            next_generation.push_back(population_[i]);
        }

        // Crossover
        for (int i = elitism_count_; i < pop_size_ - 1; i += 2) {
            Chromosome c1 = selected[i];
            Chromosome c2 = selected[i + 1];
            OrderCrossover(c1, c2);
            next_generation.push_back(c1);
            next_generation.push_back(c2);
        }
        if (next_generation.size() < pop_size_) {
             next_generation.push_back(selected.back());
        }

        // Mutation & Evaluation
        for (int i = elitism_count_; i < pop_size_; ++i) {
            cout << "  Gen " << gen + 1 << " - Evaluating individual " << i << "/" << pop_size_ << "..." << endl;
            SwapMutation(next_generation[i]);
            SegmentReversalMutation(next_generation[i]);
            EvaluateChromosome(next_generation[i]);
        }

        population_ = next_generation;
        sort(population_.begin(), population_.end());

        double current_best = population_[0].fitness_ - population_[0].penalty_;
        if (current_best > best_fitness) {
            best_fitness = current_best;
            stagnant_gens = 0;
        } else {
            stagnant_gens++;
        }

        // Early stopping rule (configurable or hardcoded here)
        if (stagnant_gens > 20) {
            cout << "Early stopping: Fitness hasn't improved for 20 generations." << endl;
            break;
        }
    }

    out_state_.clear();
    // Return top unique results
    for (int i = 0; i < std::min(5, (int)population_.size()); ++i) {
        out_state_.push_back(population_[i].state_);
    }
    
    // Apply best state to shard_ for external visualization pipeline
    State& best_state = population_[0].state_;
    for (int i = 0; i < num_shards_; ++i) {
         if (best_state.graph_[0].node_[i]) {
             Matrix3d R; Vector3d t;
             best_state.graph_[0].T_[i].Output(R, t);
             shard_[i].Move(R, t);
         }
    }
    cout << "Genetic Algorithm Finished! Best Score: " << best_fitness << endl;
}
