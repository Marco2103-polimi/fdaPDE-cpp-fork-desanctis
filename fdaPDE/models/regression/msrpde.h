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
    using Base::set_random_part; 

    // constructor
    MSRPDE() = default;
    // space-only constructor
    MSRPDE(const Base::PDE& pde, Sampling s)
        requires(is_space_only<This>::value || is_space_time_parabolic<This>::value)
        : Base(pde, s) {
        fpirls_ = FPIRLS<This>(this, tol_, max_iter_);
    }
    // space-time separable constructor
    MSRPDE(const Base::PDE& space_penalty, const Base::PDE& time_penalty, Sampling s)
        requires(is_space_time_separable<This>::value)
        : Base(space_penalty, time_penalty, s) {
        fpirls_ = FPIRLS<This>(this, tol_, max_iter_);
    }

    // setter
    void set_fpirls_tolerance(double tol) { tol_ = tol; }

    void init_model() { 
        fpirls_.init();
    }
    void solve() {
        
        fdapde_assert(y().rows() != 0);

        // execute FPIRLS_ for minimization of functional \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2
        fpirls_.compute();

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

        // compute Sigma_b_ matrix at fpirls convergence 
        Sigma_b_ = Delta_;
        for(auto k=0; k < p(); ++k){
            Sigma_b_(k) *= Delta_(k);
            Sigma_b_(k) = sigma_sq_hat_/Sigma_b_(k);   // ATT: assumes independence between random components
        }
        // set random effect in Base class for gcv computations at fpirls convergence 
        for(int i=0; i<n_groups_; ++i){
            DVector<double> temp = Z_(i)*b_hat_[i]; 
            for(int j=0; j<group_sizes_[i]; ++j){
                set_random_part(temp(j), loc_to_glob_map_[i][j]); 
            }
        } 
        // compute sigma_sq_hat_ at fpirls convergence. 
        // Nota: sto seguendo il metodo di Melchionda, ovvero sigma_sq_hat_ non ha gli edf nelle iter di fpirls, ma a convergenza viene restituito calcolandolo con gli edf
        // Nota2: questo viene fatto DOPO il calcolo di Sigma_b_ che usa quindi il sigma_sq_hat_ SENZA edf.
        compute_sigma_sq_hat(true);

        // debug 
        n_iter_ = fpirls_.n_iter(); 
        min_J_ = fpirls_.min_functional(); 
        pW_init_ = fpirls_.weights_init(); 

        return;
    }

    // required by FPIRLS_ (see fpirls_.h for details)
    void fpirls_init() {

        // Pre-allocate memory for all quatities
        Z_.resize(n_groups_);
        for(int i=0; i<n_groups_; ++i){
            Z_(i).resize(group_sizes_[i], p()); 
        }
        ZTZ_.resize(n_groups_);
        for(int i=0; i<n_groups_; ++i){
            ZTZ_(i).resize(p(), p()); 
        }
        ZtildeTZtilde_.resize(n_groups_);
        Delta_.resize(p());
        Sigma_b_.resize(p());

        pW_.resize(n_groups_); 
        for(int i=0; i<n_groups_; ++i){
            pW_(i).resize(group_sizes_[i], group_sizes_[i]);
        }

        // Compute Z_ and ZTZ_ for each group
        for(int i=0; i < n_groups_; ++i){

            Z_(i) = matrix_indexing(Z(), loc_to_glob_map_[i]);

            if(set_miss_rows_Z_to_zero_){
                // metto a zero le righe di Z che hanno missing data -> questo serve per calcolo di Delta_, Ztilde e quindi b_i, sigma_sq_hat_ etc..
                for(int glob_idx : loc_to_glob_map_[i]){
                    if(Base::nan_mask()[glob_idx]){
                        //std::cout << "set to zero rows of Z missing" << std::endl;
                        std::vector<unsigned int> glob_idxs_of_block = loc_to_glob_map_[i]; 
                        unsigned int block_row_idx; 
                        for(int idx=0; idx < glob_idxs_of_block.size(); ++idx){
                            if(glob_idxs_of_block[idx] == glob_idx){
                                block_row_idx = idx; 
                            }
                        }

                        (Z_(i).row(block_row_idx)).setZero();
                    }
                }  
            } 

            ZTZ_(i) = Z_(i).transpose() * Z_(i);
        }

        
        // initialize Delta_
        for(int k=0; k < p(); ++k){
            Delta_(k) = 0.; 
            for(int i=0; i < n_groups_; i++){
                for(int j=0; j < group_sizes_[i]; j++){
                    Delta_(k) += Z_(i)(j,k) * Z_(i)(j,k); 
                }
            }
            Delta_(k) = std::sqrt( Delta_(k)/n_groups_ ) * 3 / 8;  // ATT 3/8 fa zero (o metti il punto o metti 3/8 dopo)
        }

        // debug
        Delta_init_ = Delta_;

    }
    
    // computes W^k
    void fpirls_compute_step() {

        // compute pseudo observations
        py_ = y(); 

        // compute ZtildeTZtilde_ and invert it 
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
        for(int i=0; i < n_groups_; ++i){

            // Compute the current block with the Woodbury identity
            pW_(i) = - Z_(i) * ZtildeTZtilde_(i).solve( Z_(i).transpose() );  // I_ni - Z_i * (ZtildeTZtilde_i_)^(-1) * Z_i^T

            // Add the identity matrix (1 on the diagonal)
            for(int k=0; k < group_sizes_[i]; ++k){
                pW_(i)(k,k) = 1 + pW_(i)(k,k);	
            }
	    }

        // set weights and pseudo-observations to zero where there are missing values
        for(std::size_t i=0; i<n_locs(); ++i){   // qui voglio loopare su tutto il vettore => n_locs()
        
            if(Base::nan_mask()[i]){
                
                py_(i)=0.; 
                
                int block_idx = group_ids_(i); 
                std::vector<unsigned int> glob_idxs_of_block = loc_to_glob_map_[block_idx]; 
                unsigned int block_row_idx; 
                for(int idx=0; idx < glob_idxs_of_block.size(); ++idx){
                    if(glob_idxs_of_block[idx] == i){
                        block_row_idx = idx; 
                    }
                }
                
                for(int col_idx=0; col_idx<group_sizes_[block_idx]; ++col_idx){
                    pW_(block_idx)(block_row_idx, col_idx) = 0.;   // NOTA: se ho già settato le righe di Z a zero, questo set è inutile perchè questi sono già zeri
                }

            }
        
        }


    }

    // updates mean vector \mu after WLS solution
    void fpirls_update_step(const DMatrix<double>& hat_f, [[maybe_unused]] const DMatrix<double>& hat_beta) {

        mu_ = hat_f;   // fn + X%*%beta (no random part here!)

        compute_bhat();
	    compute_sigma_sq_hat();
        build_LTL();
        compute_C();

        for(auto k=0; k<p(); ++k){
            Delta_(k) = C_(k,k) * std::sqrt(n_groups_);
        }

    }

    // returns the data loss (J_parametric)
    double data_loss() const { 

        // J_parametric = -0.5*(mp-n)log(sigma^2) - 0.5*(|| Delta*b_i ||/ sigma)^2 + m*log(det(Delta))

        double data_loss_value = 0.;

        // cast to int to avoid overflow
        int signed_int = n_groups_*p() - n_obs();  
        
        data_loss_value -= signed_int * std::log(sigma_sq_hat_);
        
        for(auto i=0; i < n_groups_; ++i){
            // log-likelihood of random effects	(completed outside the for cycle)
            DVector<double> Deltab_i = Delta_.asDiagonal() * b_hat_[i];
            data_loss_value -= ( Deltab_i ).dot( Deltab_i ) / sigma_sq_hat_;
        }
        
        // Compute the determinant of Delta (NOTE: Delta is a diagonal matrix stored in a vector!)
        double detDelta = 1.;
        for(auto k=0; k < p(); ++k){
            detDelta *= Delta_(k);
        }
        data_loss_value += 2 * n_groups_ * std::log(detDelta);

        return data_loss_value/2;    
    }
    

    const DVector<double>& py() const { return py_; }
    const SpMatrix<double>& pW() {   // ATT: tolto const perchè devo convertire pW_ in matrice Sparsa

        // Convert pW_ in a SpMatrix (needed for fpirls temporary implementation)
        // ATT: pW_ stores the single blocks, so to build W we should use the global to local map

        // Compute the total size of the sparse matrix
        int total_size = 0;
        for(const auto& mat : pW_) {
            total_size += mat.rows();
        }
        // Resize sparse_mat_weights_
        sparse_mat_weights_.resize(total_size, total_size);
        sparse_mat_weights_.setZero();  // clean the object 

        // Store room for triplets
        std::vector<fdapde::Triplet<double>> triplet_list;
        unsigned int size_triplet = 0; 
        for(const auto& pw_block : pW_) {
            size_triplet += pw_block.rows()*pw_block.cols();
        }
        triplet_list.reserve(size_triplet);

        // Fill the sparse matrix using the local to global map
        for(auto k=0; k<n_groups_; ++k){
            const auto pw_block = pW_(k); 
            for(auto i=0; i<group_sizes_[k]; ++i){
                for(auto j=0; j<group_sizes_[k]; ++j){
                    triplet_list.emplace_back(loc_to_glob_map_[k][i], loc_to_glob_map_[k][j], pw_block(i, j)); 
                }
            }
	    }

        // finalize construction
        sparse_mat_weights_.setFromTriplets(triplet_list.begin(), triplet_list.end());
        sparse_mat_weights_.makeCompressed();

        return sparse_mat_weights_;
    }

        
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    const std::vector<DVector<double>>& b_hat() const { return b_hat_; }
    const double& sigma_sq_hat() const { return sigma_sq_hat_; };
    const DVector<double>& Sigma_b() const { return Sigma_b_; }; 
    
    // debug 
    const unsigned int& n_inter_fpirls() const { return n_iter_; }; 
    const double& min_J() const { return min_J_; }; 
    const DVector<DMatrix<double>>& Z_debug() const { return Z_; };   // ATT e' solo per debug, Z() e' un metodo di RegressionBase
    const DVector<DMatrix<double>>& ZTZ() const { return ZTZ_; }; 
    const SpMatrix<double>& pW_init() const{ return pW_init_; };
    const DVector<double>& Delta0_debug() const { return Delta_init_; };
    const SpMatrix<double>& Psi_debug() const { return Psi(); }; 
    const SpMatrix<double>& R0_debug() const { return Base::R0(); };
    const SpMatrix<double>& R1_debug() const { return Base::R1(); };
    const SpMatrix<double>& PT_debug() const { return Base::PT(); };
    const SpMatrix<double>& P0_debug() const { return Base::P0(); };
    const SpMatrix<double>& P1_debug() const { return Base::P1(); };
    const SpMatrix<double>& R0_space_debug() const { return Base::R0_space(); };
    const SpMatrix<double>& R1_space_debug() const { return Base::R1_space(); };
 
    // setter 
    void set_ids_groups(DVector<unsigned int> Rgroup_ids){  // Rgroup_ids: vector storing the id (integer) that identifies each observation to a specific group
        
        // M: compute the number of groups as the number of unique values in Rgroup_ids
        std::unordered_set<unsigned int> unique_ids(Rgroup_ids.begin(), Rgroup_ids.end()); 
        n_groups_ = unique_ids.size();
        std::cout << "Number of groups = " << n_groups_ << std::endl; 

        group_ids_ = Rgroup_ids - DVector<unsigned int>::Ones(Rgroup_ids.size()); // ATT: aggiunto -1 perchè come input dò Rgroup_ids che parte a contare da 1 (anche Melchionda lo fa, ma nel wrapper)

        // Extract the size of each group
        group_sizes_.resize(n_groups_);
        loc_to_glob_map_.resize(n_groups_);

        for(int i=0; i < Rgroup_ids.size(); ++i){  // ATT: Melchionda lo chiama m_ che però non è il #gruppi ma l'ampiezza campionaria, quindi OK
            int i_loc = Rgroup_ids(i)-1;           // ATT: aggiunto -1 per stesso motivo di sopra 
            group_sizes_[i_loc] += 1;	           // update the counter of the correct group
            loc_to_glob_map_[i_loc].push_back(i);  // map the local index to the global one
        }

        // // debug: print loc_to_glob_map_
        // std::cout << "printing loc_to_glob_map_" << std::endl; 
        // for(int k=0; k<loc_to_glob_map_.size(); ++k){
        //     std::cout << std::endl; 
        //     for(int j=0; j<loc_to_glob_map_[k].size(); ++j){
        //         std::cout << loc_to_glob_map_[k][j] << "; ";  
        //     }    
        // }

        // std::cout << "printing group_sizes_" << std::endl; 
        // for(int k=0; k<group_sizes_.size(); ++k){
        //     std::cout << group_sizes_[k] << "; ";    
        // }
        // std::cout << std::endl; 
        

    }

    void set_fpirls_max_iter(int max_iter) { 
        std::cout << "setting max_iter fpirls to " << max_iter << std::endl; 
        max_iter_ = max_iter; 
        fpirls_.set_max_iter(max_iter);   // M: update variable in fpirls object
    }

    void set_miss_rows_Z_to_zero(bool flag) { set_miss_rows_Z_to_zero_ = flag; } 

    // GCV support -> NON sommo le osservazioni mascherate !
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {

        double result = 0;
        for (int i = 0; i < n_locs(); ++i) {
            if (!Base::masked_obs()[i]) result += (op2.coeff(i, 0) - op1.coeff(i, 0))*(op2.coeff(i, 0) - op1.coeff(i, 0));
        }
        return result;  

    }      


   private:
    DVector<double> py_; 
    SpMatrix<double> sparse_mat_weights_ {}; 
    DVector<DMatrix<double>> pW_ {};   // diagonal blocks of W^k (one block for each group)
    // ATT in gsrpde e qsrpde pW_ è un vettore, qui è un vettore di matrici 

    DVector<double> mu_;      // \mu^k = [ \mu^k_1, ..., \mu^k_n ] : fitted vector at step k
    fdapde::SparseLU<SpMatrix<double>> invA_;

    unsigned int n_groups_; 
    std::vector<unsigned int> group_sizes_; 
    std::vector<std::vector<unsigned int>> loc_to_glob_map_; 
    DVector<unsigned int> group_ids_; 

    DVector<DMatrix<double>> Z_;  // vectors of blocks storing the random effect matrices Z_i
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

    // debug 
    unsigned int n_iter_;    // number of iterations in fpirls at convergence 
    double min_J_; 
    SpMatrix<double> pW_init_; 
    DVector<double> Delta_init_; 
    bool set_miss_rows_Z_to_zero_ = true; 

    // helper functions
    DVector<double> vector_indexing(const DVector<double>& big_vector, const std::vector<unsigned int> ids){

        // ids: vector of global indexes 

        DVector<double> small_vector(ids.size());
        for(int k=0; k < ids.size(); ++k){
            small_vector(k) = big_vector(ids[k]);
        }
        
        return small_vector;
    }
    DMatrix<double> matrix_indexing(const DMatrix<double>& big_matrix, const std::vector<unsigned int> row_ids){

        // row_ids: vector of global row indexes
        
        DMatrix<double> small_matrix(row_ids.size(), big_matrix.cols());

        // Reconstruct the block using loc_to_glob_map
        for (unsigned int i = 0; i < row_ids.size(); ++i) {
            small_matrix.row(i) = big_matrix.row(row_ids[i]);
        }
        return small_matrix;

    }

    void compute_bhat() { 
        b_hat_.resize(n_groups_); 
        for(auto i=0; i < n_groups_; ++i){
            DVector<double> res_i = vector_indexing(y(), loc_to_glob_map_[i]);
            res_i -= vector_indexing(mu_, loc_to_glob_map_[i]);   // here mu_ does NOT contain the random effect
            
            // M: for missing (ATT aggiunto rispetto a Melchionda)
            for(int j=0; j<res_i.size(); ++j){
                if(Base::nan_mask()[loc_to_glob_map_[i][j]]){
                    res_i(j) = 0.; 
                }
            }

            
            b_hat_[i] = ZtildeTZtilde_(i).solve( Z_(i).transpose() * res_i );
        }
    }
    
    DMatrix<double> lmbQ_model(const DMatrix<double>& x) const {

        SpMatrix<double> W = fpirls_.solver().W(); // we need to take it from fpirls_ since we are not at convergence yet, so RegressionBase does not store the correct matrices!

        if (!has_covariates()) return W * x;
        DMatrix<double> v = X().transpose() * W * x;   // X^\top*W*x
        DMatrix<double> XtWX = X().transpose() * W * X();
        Eigen::PartialPivLU<DMatrix<double>> invXtWX = XtWX.partialPivLu();
        
        DMatrix<double> z = invXtWX.solve(v);          // (X^\top*W*X)^{-1}*X^\top*W*x
        // compute W*x - W*X*z = W*x - (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(I - H)*x = Q*x
        return W * x - W * X() * z;
    }   
    double compute_edf(){

        // NOTA: Se scegliamo la strategia di Melchionda, per cui gli edf ci servono solo a convergenza fpirls, 
        //       si puo' modificare questo codice chiamando T, lmbQ direttamente da Base (che è più efficiente 
        //       in quanto qui per semplicità stiamo trasformando tutto in matrici dense). 
        //       Se vogliamo la strategia di Pigani, ci servono gli edf ad ogni iterazione di fpirls, quindi
        //       bisogna implementare lmbQ con i pesi di quell'iterazione corrente (vedi lmbQ_model).

        // M: nota: very inefficient computation !! 
        
        DMatrix<double> T = PsiTD() * lmbQ_model(Psi()) + Base::P();  // nota: non posso chiamare direttamente Base::T perchè ci serve il nostro lmbQ !

        Eigen::PartialPivLU<DMatrix<double>> invT = T.partialPivLu();
        DMatrix<double> E = PsiTD();          
        DMatrix<double> S = lmbQ_model(Psi() * invT.solve(E));   // \Psi*T^{-1}*\Psi^T*Q
        double ret_edf = 0.; 
        
        for(int i=0; i<S.rows(); ++i){
            ret_edf += S(i,i); 
        }
        return ret_edf;
    }

    void compute_sigma_sq_hat(bool edf_flag=false) {

        sigma_sq_hat_ = 0.;	

        // set to fit vector zeros where there are missing values, so that the residual is computed correctly
        DVector<double> fit_adj = mu_;    
        for(int j=0; j<fit_adj.size(); ++j){
            if(Base::nan_mask()[j]){
                fit_adj(j) = 0.; 
            }
        }

        for(auto i=0; i < n_groups_; i++){ 

            DVector<double> mu_i = vector_indexing(fit_adj, loc_to_glob_map_[i]);
            DVector<double> res_i = vector_indexing(y(), loc_to_glob_map_[i]);  // note: y() has zeros in correspondence of missing values
            
            // res_i -= ( mu_i + Z_(i)*b_hat_[i] );   //  here the residual contains the random effect too 
            // // note: sum only over observed data (funziona se nell'init abbiamo settato in Z_ zero rows in correspondence of missing values)

            // M: set zeros in random part for missing values case (when Z_ is not zeroed)
            DVector<double> Zb_i = Z_(i)*b_hat_[i];
            for(int j=0; j<res_i.size(); ++j){
                if(Base::nan_mask()[loc_to_glob_map_[i][j]]){
                    Zb_i(j) = 0.; 
                } 
            }
            res_i -= ( mu_i + Zb_i );   //  here the residual contains the random effect too 
            // note: sum only over observed data

            sigma_sq_hat_ += res_i.dot(res_i);
        }

        // // Versione Pigani (per test 1-tris)
        // std::cout << "ATT: RUNNING PIGANI sigma2 computaiton!!" << std::endl;
        // double edf = compute_edf(); 
        // if(has_covariates()){
        //     edf += q();   // p()? Pigani non lo mette  
        // }
        // sigma_sq_hat_ /= (n_obs()-edf); 

    
        // Versione Melchionda 
        if(edf_flag){
            double edf = compute_edf(); 
            if(has_covariates()){
                edf += q();   // +m*p()?
            }
            sigma_sq_hat_ /= (n_obs()-edf); 

            std::cout << "edf = " << std::setprecision(16) << edf << std::endl;

        } else{
            sigma_sq_hat_ /= n_obs();  
        }
  
        
    }
    void build_LTL(){

        DMatrix<double> LTL_temp = DMatrix<double>::Zero(p(), p());
        
        for(auto i=0; i < n_groups_; ++i){
            LTL_temp += b_hat_[i]*(b_hat_[i]).transpose() / sigma_sq_hat_;
            LTL_temp += ZtildeTZtilde_(i).solve(DMatrix<double>::Identity(p(), p()));
        }
        
        LTL_.compute(LTL_temp); // ATT: rispetto alla formula della teoria, non manca un /n_groups_? no, perchè inserito dopo il calcolo di C 
    }
    void compute_C(){
        DMatrix<double> C_temp = LTL_.matrixL();
        C_ = C_temp.triangularView<Eigen::Lower>().solve( DMatrix<double>::Identity(p(), p()) );
    }
 
};

}   // namespace models
}   // namespace fdapde

#endif   // __MSRPDE_H__
