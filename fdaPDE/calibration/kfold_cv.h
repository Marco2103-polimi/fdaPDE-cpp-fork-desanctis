// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __K_FOLD_CV__
#define __K_FOLD_CV__

#include <fdaPDE/utils.h>
#include <fdaPDE/linear_algebra.h>

#include <algorithm>
#include <vector>
using fdapde::core::BlockFrame;
using fdapde::core::BinaryVector;
using fdapde::Dynamic;

#include "calibration_base.h"

namespace fdapde {
namespace calibration {
  
// general implementation of KFold Cross Validation
class KCV : public CalibratorBase<KCV> {
   private:
    // algorithm's parameters
    int K_;          // number of folds
    int seed_;       // seed used for BlockFrame shuffling
    bool shuffle_;   // whether to shuffle data before splitting into folds

    DVector<double> avg_scores_;   // mean CV score for each \lambda
    DVector<double> std_scores_;   // CV score standard deviation for each \lambda
    DMatrix<double> scores_;       // matrix of CV scores
    DVector<double> optimum_;      // optimal smoothing parameter

    // produces a bit_mask identifying the i-th fold to be supplied to the model via mask_obs
    using TrainTestPartition = std::pair<BinaryVector<Dynamic>, BinaryVector<Dynamic>>;
    TrainTestPartition split(const BlockFrame<double, int>& data, int i) {
        int n = data.rows();          // number of data points
        int m = std::floor(n / K_);   // number of data per fold

        // create test-train index sets, as function of test fold i
        BinaryVector<Dynamic> test_mask(n);
	BinaryVector<Dynamic> train_mask(n);
        for (int j = 0; j < n; ++j) {
            if (j >= m * i && j < m * (i + 1)) {
                test_mask.set(j);
            } else {
                train_mask.set(j);
            }
        }
        return std::make_pair(train_mask, test_mask);
    }
   public:
    // constructor
    KCV() = default;
    KCV(int K, bool shuffle = true) : K_(K), seed_(std::random_device()()), shuffle_(shuffle) {};
    KCV(int K, int seed, bool shuffle = true) :
        K_(K), seed_((seed == fdapde::random_seed) ? std::random_device()() : seed), shuffle_(shuffle) {};

    // selects best smoothing parameter of model according to a K-fold cross validation strategy
    template <typename ModelType, typename ScoreType>
    DVector<double> fit(ModelType& model, const DMatrix<double>& lambdas, ScoreType cv_score, 
                        DMatrix<double> time_locs, DMatrix<double> space_locs, DMatrix<double> y, 
                        bool has_fix_cov = false, bool has_rnd_cov = false) {  // ATT M aggiunti time_locs, space_locs, y
        
        std::cout << "dim y = " << y.rows() << ";" << y.cols() << std::endl;

        fdapde_assert(lambdas.cols() == 1 || lambdas.cols() == 2);
        // reserve space for CV scores
        scores_.resize(K_, lambdas.rows());

        if (shuffle_) {   // perform a first shuffling of the data if required

            // model.set_data(model.data().shuffle(seed_));   // prima 


            // M ATT aggiunto shuffle anche per le locs
            // NOTA: in space-time non posso fare 
            // model.set_spatial_locations(model.locs().shuffle(seed_));
            // model.set_temporal_locations(time_locs.shuffle(seed_));
            // perchè anche se stesso seed, le lunghezze dei vettori sono diverse quindi farebbe un diverso shuffle.
            // Quindi faccio prima shuffle su space_locs, poi su time_locs, e poi per y l'ordinamento sarà il prodotto cartesiano dei due
            

            // store room 
            DMatrix<double> space_locs_shuffle;
            DMatrix<double> time_locs_shuffle;
            DMatrix<double> y_shuffle;
            space_locs_shuffle.resize(model.n_spatial_locs(), 2);
            time_locs_shuffle.resize(model.n_temporal_locs(), 1);
            y_shuffle.resize(model.n_spatial_locs(), model.n_temporal_locs());

            std::cout << "model.n_spatial_locs() = " << model.n_spatial_locs() << std::endl;
            std::cout << "model.n_temporal_locs() = " << model.n_temporal_locs() << std::endl;
            std::cout << "model.n_locs() = " << model.n_locs() << std::endl;

            // generate random permutation for space locs shuffle
            auto rng_space_locs = std::default_random_engine(seed_);
            std::vector<int> indexes_space_locs;
            indexes_space_locs.resize(model.n_spatial_locs());
            std::iota(indexes_space_locs.begin(), indexes_space_locs.end(), 0);
            std::shuffle(std::begin(indexes_space_locs), std::end(indexes_space_locs), rng_space_locs);

            for(int i = 0; i < model.n_spatial_locs(); i++) {
                space_locs_shuffle(i,0) = space_locs(indexes_space_locs[i],0);
                space_locs_shuffle(i,1) = space_locs(indexes_space_locs[i],1);
            }


            // generate random permutation for time locs shuffle
            auto rng_time_locs = std::default_random_engine(seed_);
            std::vector<int> indexes_time_locs;
            indexes_time_locs.resize(model.n_temporal_locs());
            std::iota(indexes_time_locs.begin(), indexes_time_locs.end(), 0);
            std::shuffle(std::begin(indexes_time_locs), std::end(indexes_time_locs), rng_time_locs);

            for(int i = 0; i < model.n_temporal_locs(); i++) {
                time_locs_shuffle(i,0) = time_locs(indexes_time_locs[i],0);
            }


            // generate random permutation for y shuffle accordingly
            for(int j=0; j<model.n_temporal_locs(); ++j){
                for(int i=0; i<model.n_spatial_locs(); ++i){                    
                    y_shuffle(i, j) = y(indexes_space_locs[i], indexes_time_locs[j]);
                }    
            }

            std::cout << "dim y_shuffle = " << y_shuffle.rows() << ";" << y_shuffle.cols() << std::endl;
            std::cout << "dim space_locs_shuffle = " << space_locs_shuffle.rows() << " , " << space_locs_shuffle.cols() << std::endl;

            // set the shuffled data
            BlockFrame<double, int> df_shuffle;
            df_shuffle.stack(OBSERVATIONS_BLK, y_shuffle);  // ATT stack for space-time models

            DMatrix<double> X_shuffle; 
            if(has_fix_cov){
                X_shuffle.resize(model.n_locs(), model.X().cols());
                for(int k=0; k<model.X().cols();++k){
                    for(int j=0; j<model.n_temporal_locs(); ++j){
                        for(int i=0; i<model.n_spatial_locs(); ++i){                    
                            X_shuffle(j*model.n_temporal_locs()+i, k) = model.X()(indexes_time_locs[j]*model.n_temporal_locs()+indexes_space_locs[i], k);
                        }    
                    }
                }

                df_shuffle.insert(DESIGN_MATRIX_BLK, X_shuffle);
            }


            DMatrix<double> Z_shuffle; 
            if(has_rnd_cov){
                Z_shuffle.resize(model.n_locs(), model.Z().cols());
                for(int k=0; k<model.Z().cols();++k){
                    for(int j=0; j<model.n_temporal_locs(); ++j){
                        for(int i=0; i<model.n_spatial_locs(); ++i){                    
                            Z_shuffle(j*model.n_temporal_locs()+i, k) = model.Z()(indexes_time_locs[j]*model.n_temporal_locs()+indexes_space_locs[i], k);
                        }    
                    }
                }

                df_shuffle.insert(DESIGN_MATRIX_RANDOM_BLK, Z_shuffle);

            }

            // set shuffle data 
            model.set_data(df_shuffle);
            model.set_temporal_locations(time_locs_shuffle); 
            model.set_spatial_locations(space_locs_shuffle); 

            std::cout << "end shuffle" << std::endl;


    }
        // cycle over all tuning parameters
        for (int j = 0; j < lambdas.rows(); ++j) {
            for (int fold = 0; fold < K_; ++fold) {   // fixed a tuning parameter, cycle over all data splits
                // compute train-test partition and evaluate CV score
                TrainTestPartition partition_mask = split(model.data(), fold);
                scores_.coeffRef(fold, j) = cv_score(lambdas.row(j), partition_mask.first, partition_mask.second);
            }
        }
        // reserve space for storing results
        avg_scores_ = DVector<double>::Zero(lambdas.rows());
        std_scores_ = DVector<double>::Zero(lambdas.rows());
        for (int j = 0; j < lambdas.rows(); ++j) {
            // record the average score and its standard deviation computed among the K_ runs
            double avg_score = 0;
            for (int i = 0; i < K_; ++i) avg_score += scores_(i, j);
            avg_score /= K_;
            double std_score = 0;
            for (int i = 0; i < K_; ++i) std_score += std::pow(scores_(i, j) - avg_score, 2);
            std_score = std::sqrt(std_score / (K_ - 1));   // use unbiased sample estimator for standard deviation
            // store results
            avg_scores_[j] = avg_score;
            std_scores_[j] = std_score;
        }

        // store optimal lambda according to given metric F
        Eigen::Index opt_score;
        avg_scores_.minCoeff(&opt_score);
        optimum_ = lambdas.row(opt_score);
	return optimum_;
    }

    // getters
    const DVector<double>& avg_scores() const { return avg_scores_; }   // mean CV score vector
    const DVector<double>& std_scores() const { return std_scores_; }   // CV score standard deviation vector
    const DMatrix<double>& scores() const { return scores_; }           // CV scores
    const DVector<double>& optimum() const { return optimum_; }         // optimal tuning parameter
    // setters
    void set_n_folds(int K) { K_ = K; }
};

}   // namespace calibration
}   // namespace fdapde

#endif   // __K_FOLD_CV__
