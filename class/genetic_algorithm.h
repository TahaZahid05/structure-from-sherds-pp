#pragma once
#ifndef _GENETIC_ALGORITHM_H_
#define _GENETIC_ALGORITHM_H_

#include <vector>
#include <list>
#include <string>
#include <random>
#include "../class/data_structure.h"
#include "../class/ranking_system.h"

using namespace std;

struct Chromosome {
    vector<int> sequence_; 
    State state_;
    double fitness_;
    double penalty_;

    Chromosome() : fitness_(0.0), penalty_(0.0) {}

    bool operator<(const Chromosome& other) const {
        return (fitness_ - penalty_) > (other.fitness_ - other.penalty_); // Descending sort
    }
};

class GeneticAlgorithm {
public:
    GeneticAlgorithm(int pop_size, 
                     int max_generations, 
                     int tournament_size, 
                     double crossover_prob, 
                     double swap_mut_prob, 
                     double rev_mut_prob, 
                     int elitism_count, 
                     int candidate_limit,
                     vector<Geom>& shard, 
                     list<LCSIndex>& LCS_out, 
                     int step_size, 
                     const string& log_path);
    
    ~GeneticAlgorithm() {}

    void Run();

    vector<State> out_state_;
    vector<Geom> shard_;
    string log_path_;

private:
    void InitializePopulation();
    void EvaluateChromosome(Chromosome& chrom);
    
    void TournamentSelection(const vector<Chromosome>& population, vector<Chromosome>& selected);
    void OrderCrossover(Chromosome& parent1, Chromosome& parent2);
    void SwapMutation(Chromosome& chrom);
    void SegmentReversalMutation(Chromosome& chrom);

    // To compute heuristic seed probabilities
    void CalculatePairwiseMatchScores();

    int pop_size_;
    int max_generations_;
    int tournament_size_;
    double crossover_prob_;
    double swap_mut_prob_;
    double rev_mut_prob_;
    int elitism_count_;
    int candidate_limit_;
    
    int num_shards_;
    int step_size_;

    list<LCSIndex> lcs_out_;
    
    vector<Chromosome> population_;
    
    MatrixXd match_scores_; // Precomputed heuristic pairwise match scores
};

#endif // _GENETIC_ALGORITHM_H_
