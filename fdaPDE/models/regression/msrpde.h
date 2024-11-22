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

#ifndef __MSRPDE_H__
#define __MSRPDE_H__

#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "fpirls.h"
#include "regression_base.h"

namespace fdapde {
namespace models {

template <typename RegularizationType_>
class MSRPDE : public RegressionBase<MSRPDE<RegularizationType_>, RegularizationType_> {
   public:
    using RegularizationType = std::decay_t<RegularizationType_>;
    using This = MSRPDE<RegularizationType>;
    using Base = RegressionBase<MSRPDE<RegularizationType>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS
    using Base::invXtWX_;   // LU factorization of X^T*W*X
    using Base::lambda_D;   // smoothing parameter in space
    using Base::P;          // discretized penalty matrix
    using Base::W_;         // weight matrix
    using Base::XtWX_;      // q x q matrix X^T*W*X

    // constructor
    MSRPDE() = default;
    // space-only constructor
    MSRPDE(const Base::PDE& pde, Sampling s)
        requires(is_space_only<This>::value)
        : Base(pde, s) {
        std::cout << "msrpde constructor here 0" << std::endl; 
        fpirls_ = FPIRLS<This>(this, tol_, max_iter_);
        std::cout << "msrpde constructor here 1" << std::endl;
    }
    // setter
    void set_fpirls_tolerance(double tol) { tol_ = tol; }
    void set_fpirls_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }

    void init_model() { 
        fpirls_.init();
    }
    void solve() {
        
        std::cout << "in solve msrpde here 0" << std::endl; 
        fdapde_assert(y().rows() != 0);
        std::cout << "in solve msrpde here 1" << std::endl; 

        // execute FPIRLS_ for minimization of functional \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2
        fpirls_.compute();
        std::cout << "in solve msrpde here 2" << std::endl; 

        // fpirls_ converged: store solution estimates  
        W_ = fpirls_.solver().W();
        f_ = fpirls_.solver().f();
        g_ = fpirls_.solver().g();
        // parametric part, if problem was semi-parametric
        if (has_covariates()) {
            beta_ = fpirls_.solver().beta();
            XtWX_ = fpirls_.solver().XtWX();
            invXtWX_ = fpirls_.solver().invXtWX();
            U_ = fpirls_.solver().U();
            V_ = fpirls_.solver().V();
        }
        invA_ = fpirls_.solver().invA();
        // TODO compute Sigma_b_ matrix at fpirls convergence 
        Sigma_b_ = Delta_;
        for(auto k=0; k < p(); ++k){
            Sigma_b_(k) *= Delta_(k);
            Sigma_b_(k) = sigma_sq_hat_/Sigma_b_(k);   // ATT: assumes independence between random components
        }

        return;
    }

    // required by FPIRLS_ (see fpirls_.h for details)
    void fpirls_init() {

        // Pre-allocate memory for all quatities
        ZTZ_.resize(n_groups_);
        for(int i=0; i<n_groups_; ++i){
            ZTZ_(i).resize(p(), p()); 
        }
        ZtildeTZtilde_.resize(n_groups_);
        Delta_.resize(p());
        Sigma_b_.resize(p());

        // Compute ZTZ_ for each group
        for(int i=0; i < n_groups_; ++i){
            DMatrix<double> Z_i = blockmatrix_indexing(Z(), i, group_sizes_); 
            ZTZ_(i) = Z_i.transpose() * Z_i;
        }
        
        // initialize Delta_
        for(int k=0; k < p(); ++k){
            for(int i=0; i < n_groups_; ++i){
                DMatrix<double> Z_i = blockmatrix_indexing(Z(), i, group_sizes_); 
                for(int j=0; j < n_groups_; ++j){
                    Delta_(k) += Z_i(j,k)*Z_i(j,k); 
                }
            }
            Delta_(k) = 3/8*std::sqrt( Delta_(k)/n_groups_ ); 
        }

    }
    
    // computes W^k
    void fpirls_compute_step() {

        // compute ZtildeTZtilde_
        for(auto i=0; i < n_groups_; ++i){
            
            // For each group, consider Z^T Z
            DMatrix<double> ZtildeTZtilde_temp = ZTZ_(i);
            
            // Then add the diagonal elements stored in D_
            for(auto k=0; k < p(); ++k){
                ZtildeTZtilde_temp(k,k) += Delta_(k) * Delta_(k);
            }	
            
            ZtildeTZtilde_(i).compute(ZtildeTZtilde_temp);
            
        }

        // compute weights
        for(int i=0; i < n_groups_; i++){
            DMatrix<double> Z_i = blockmatrix_indexing(Z(), i, group_sizes_); 
            // Compute the current block with the Woodbury identity
            pW_(i) = - Z_i * ZtildeTZtilde_(i).solve( Z_i.transpose() ); // I_ni - Z_i * (ZtildeTZtilde_i_)^(-1) * Z_i^T
            
            // Add the identity matrix (1 on the diagonal)
            for(int k=0; k < group_sizes_[i]; ++k){
                pW_(i)(k,k) += 1;	
            }
	    }
        
    }
    // updates mean vector \mu after WLS solution
    void fpirls_update_step(const DMatrix<double>& hat_f, [[maybe_unused]] const DMatrix<double>& hat_beta) {
        mu_ = hat_f;

        compute_bhat();
	    compute_sigma_sq_hat();
        build_LTL();
        compute_C();

        for(auto i=0; i<p(); ++i){
            Delta_(i) = C_(i,i) * std::sqrt(n_groups_);
        }

    }
    // returns the data loss (J_parametric)
    double data_loss() const { 
        // ATT: capire se vogliamo usare la loss approssimata di fpirls o quella esatta. 
        // Per ora implementiamo quella esatta per eventuali confronti con Melchionda

        double data_loss_value = 0;

        data_loss_value -= ( n_groups_*p() - n_obs() ) * std::log(sigma_sq_hat_);
        
        for(auto i=0; i < n_groups_; ++i){
            // log-likelihood of random effects	(completed outside the for cycle)
            DVector<double> Deltab_i = Delta_.asDiagonal() * b_hat_[i];
            data_loss_value -= ( Deltab_i ).dot( Deltab_i ) / sigma_sq_hat_;
        }
        
        // Compute the determinant of Delta (NOTE: Delta is a diagonal matrix stored in a vector!)
        double detDelta = 1;
        for(auto k=0; k < p(); ++k){
            detDelta *= Delta_[k];
        }
        data_loss_value += 2 * n_groups_ * std::log(detDelta);

        return data_loss_value/2 ;
    }
    

    const DVector<double>& py() const { return py_; }
    const SpMatrix<double>& pW() {   // ATT: tolto const perchè devo convertire pW_ in matrice Sparsa
        // Convert pW_ in a SpMatrix (needed for fpirls temporary implementation)

        // Compute the total size of the sparse matrix
        int total_size = 0;
        for (const auto& mat : pW_) {
            total_size += mat.rows();
        }

        // Resize sparse_mat_weights_
        sparse_mat_weights_.resize(total_size, total_size);

        // Fill the sparse matrix with blocks from pW_
        int current_offset = 0;
        for (const auto& block : pW_) {
            int block_size = block.rows();
            
            // Insert the block at the correct diagonal position in the sparse matrix
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    if (block(i, j) != 0) {
                        sparse_mat_weights_.insert(current_offset + i, current_offset + j) = block(i, j);
                    }
                }
            }
            
            // Update the offset for the next block diagonal
            current_offset += block_size;
        }

        return sparse_mat_weights_;
    }

        
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    const std::vector<DVector<double>>& b_hat() const { return b_hat_; };

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }
   
   private:
    DVector<double> py_ = y(); 
    SpMatrix<double> sparse_mat_weights_ {}; 
    DVector<DMatrix<double>> pW_ {};   // diagonal blocks of W^k (one block for each group)
    // ATT in gsrpde e qsrpde pW_ è un vettore, qui è un vettore di matrici --> da capire cosa accade in fpirls

    DVector<double> mu_;      // \mu^k = [ \mu^k_1, ..., \mu^k_n ] : fitted vector at step k
    fdapde::SparseLU<SpMatrix<double>> invA_;

    unsigned int n_groups_; 
    std::vector<unsigned int> group_sizes_; 
    std::vector<std::vector<int>> ids_perm_; 

    DVector<DMatrix<double>> ZTZ_;  // vectors of blocks storing the random effect matrices Z_i^T * Z_i
    DVector<Eigen::LLT<DMatrix<double>>> ZtildeTZtilde_; 
    std::vector<DVector<double>> b_hat_;  // vector containing the estimates of the random effects for each group
    Eigen::LLT<DMatrix<double>> LTL_; 
    DMatrix<double> C_;       // Cholesky factor of LTL_
    DVector<double> Delta_;   // ATT: we are assuming independent random effects  
    DVector<double> Sigma_b_; // ATT: we are assuming independent random effects  
    double sigma_sq_hat_;     // estimate of the variance of the errors 

    FPIRLS<This> fpirls_;   // fpirls algorithm
    int max_iter_ = 200;    // maximum number of iterations in fpirls before forced stop
    double tol_ = 1e-6;     // fprils convergence tolerance



    // helper functions
    void set_ids_groups(std::vector<unsigned int> Rgroup_ids){  // Rgroup_ids: vector storing the id (integer) that identifies each observation to a specific group
        
        // M: compute the number of groups as the number of unique values in Rgroup_ids
        std::unordered_set<unsigned int> unique_ids(Rgroup_ids.begin(), Rgroup_ids.end()); 
        n_groups_ = unique_ids.size();
        std::cout << "Number of groups =" << n_groups_ << std::endl; 

        // Extract the size of each group
        group_sizes_.resize(n_groups_);
        ids_perm_.resize(n_groups_);

        for(int i=0; i < n_groups_; ++i){
            int i_loc = Rgroup_ids[i];
            group_sizes_[i_loc] += 1;	// update the counter of the correct group
            ids_perm_[i_loc].push_back(i); // map the global index to the local one
        }
    }

    DVector<double> vector_indexing(const DVector<double>& big_vector, const std::vector<int> ids){

        DVector<double> small_vector(ids.size());
        for(int k=0; k < ids.size(); ++k){
            small_vector(k) = big_vector(ids[k]);
        }
        
        return small_vector;
    }

    DMatrix<double> blockmatrix_indexing(const DMatrix<double>& big_matrix, const unsigned int id_block, 
                                         const std::vector<unsigned int> rows_blocks){
        
        // Ensure the id_block is within bounds
        assert(id_block < rows_blocks.size() && "Block index is out of range.");

        // Compute the total number of rows in big_matrix
        unsigned int total_rows = 0;
        for (auto rows : rows_blocks) total_rows += rows;

        // Ensure the matrix dimensions are valid
        assert(total_rows == big_matrix.rows() && "Matrix rows must match rows_blocks.");
        assert(big_matrix.cols() % rows_blocks.size() == 0 && "Column count must be divisible by blocks.");

        // Compute the number of columns in each block
        unsigned int block_columns = big_matrix.cols() / rows_blocks.size();

        // Calculate the starting row for the requested block
        unsigned int start_row = 0;
        for (unsigned int i = 0; i < id_block; ++i) {
            start_row += rows_blocks[i];
        }

        // Get the number of rows for the requested block
        unsigned int block_rows = rows_blocks[id_block];

        // Ensure the block lies within the matrix
        assert(start_row + block_rows <= big_matrix.rows() && "Block exceeds matrix bounds.");

        // Extract the block as a new matrix
        DMatrix<double> block = big_matrix.block(start_row, id_block * block_columns, block_rows, block_columns);

        return block;

    }

    void compute_bhat() { 
        for(auto i=0; i < n_groups_; ++i){
            DMatrix<double> Z_i = blockmatrix_indexing(Z(), i, group_sizes_); 
            DVector<double> res_i = vector_indexing(y(), ids_perm_[i]);
            res_i -= vector_indexing(mu_, ids_perm_[i]);
            b_hat_[i] = ZtildeTZtilde_(i).solve( Z_i.transpose() * res_i );
        }
    }
    void compute_sigma_sq_hat() {

        sigma_sq_hat_ = 0.;	
        for(auto i=0; i < n_groups_; i++){
            
            DMatrix<double> Z_i = blockmatrix_indexing(Z(), i, group_sizes_); 
            DVector<double> mu_i = vector_indexing(mu_, ids_perm_[i]);
            DVector<double> res_i = vector_indexing(y(), ids_perm_[i]);
            res_i -= ( mu_i + Z_i*b_hat_[i] );   // note: here the residual contains the random effect too
            sigma_sq_hat_ += res_i.dot(res_i);
        }
        
        sigma_sq_hat_ /= n_obs();  // ATT: dubbio: Melchionda mette solo n, ma la formula dice n-(q+tr(S))
    }
    void build_LTL(){

        DMatrix<double> LTL_temp = DMatrix<double>::Zero(p(), p());
        
        for(auto i=0; i < n_groups_; ++i){
            LTL_temp += b_hat_[i]*(b_hat_[i]).transpose() / sigma_sq_hat_;
            LTL_temp += ZtildeTZtilde_(i).solve(DMatrix<double>::Identity(p(), p()));
        }
        
        LTL_.compute(LTL_temp);
    }
    void compute_C(){
        DMatrix<double> C_temp = LTL_.matrixL();
        C_ = C_temp.triangularView<Eigen::Lower>().solve( DMatrix<double>::Identity(p(), p()) );
    }
 
};

}   // namespace models
}   // namespace fdapde

#endif   // __MSRPDE_H__
