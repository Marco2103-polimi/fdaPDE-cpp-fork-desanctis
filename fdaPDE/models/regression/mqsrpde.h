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

#ifndef __MQSRPDE_H__
#define __MQSRPDE_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h" 
#include "../model_macros.h"
#include "../model_traits.h" 
#include "../sampling_design.h"
#include "distributions.h"
#include "fpirls.h"
#include "qsrpde.h"
using fdapde::models::QSRPDE;
#include "regression_base.h"


namespace fdapde{
namespace models{
	    
  
template <typename RegularizationType_>
    class MQSRPDE : public RegressionBase<MQSRPDE<RegularizationType_>, RegularizationType_> {  
    
    public:
        using RegularizationType = std::decay_t<RegularizationType_>;
        using This = MQSRPDE<RegularizationType>;
        using Base = RegressionBase<MQSRPDE<RegularizationType>, RegularizationType>;

    private:

        // model parameters 
        unsigned int h_;                      // number of quantile orders 
        const std::vector<double> alphas_;    // quantile orders
        bool do_process = false;
        bool force_entrance = false; 

        // algorithm's parameters 
        double gamma0_ = 5.;                   // crossing penalty   
        double eps_ = 1e-6;                    // crossing tolerance 
        double C_ = 1.5;                       // crossing penalty factor
        double tolerance_ = 1e-5;              // convergence tolerance 
        double tol_weights_ = 1e-6;            // weights tolerance
        std::size_t max_iter_ = 200;           // max number of inner iterations 
        std::size_t max_iter_global_ = 10;     // max number of outer iterations 

        unsigned int fpirls_iter_ = 200;       // number of max fpirls iterations for the single estimates

        std::size_t k_ = 0;                    // inner iteration index
        std::size_t iter_ = 0;                 // outer iteration index

        // linear system  
        SparseBlockMatrix<double,2,2> A_{};         // system matrix of non-parametric problem (2hN x 2hN matrix)
        fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
        DVector<double> b_{};                       // system rhs 

        // room for current solution 
        DVector<double> f_curr_{};     // current estimate of the spatial field f (1 x h*N vector)
        DVector<double> fn_curr_{};    // current estimate of the spatial field f_n (1 x h*n vector)
        DVector<double> g_curr_{};     // current PDE misfit (1 x h*N vector)
        DVector<double> beta_curr_{};  // current estimate of the coefficient vector (1 x h*q vector)

        // room for initial solution
        DVector<double> f_init_{}; 
        DVector<double> fn_init_{}; 
        DVector<double> g_init_{}; 
        DVector<double> beta_init_{}; 

        // room for algorithm's matrices  
        SpMatrix<double> Ih_;                 // identity h x h 
        SpMatrix<double> In_;                 // identity n x n
        SpMatrix<double> Iq_;                 // identity q x q 
        SpMatrix<double> Ihn_;                // identity h*n x h*n 
        SpMatrix<double> D_{};                
        SpMatrix<double> D_script_{}; 
        DVector<double> l_hn_{}; 
        SpMatrix<double> Psi_multiple_{}; 
        SpMatrix<double> R0_multiple_{};
        SpMatrix<double> R1_multiple_{}; 
        DVector<double> z_{};
        DVector<double> w_{};
        DiagMatrix<double> W_bar_{};
        SpMatrix<double> W_multiple_{}; 
        DMatrix<double> X_multiple_{};
        DMatrix<double> XtWX_multiple_{};
        Eigen::PartialPivLU<DMatrix<double>> invXtWX_multiple_{}; 
        DMatrix<double> Q_multiple_{}; 
        DMatrix<double> H_multiple_{}; 
        DMatrix<double> U_multiple_{};
        DMatrix<double> V_multiple_{};

    public:
        IMPORT_REGRESSION_SYMBOLS;
        using Base::invXtWX_;
        using Base::lambda_D;   // smoothing parameter in space
        using Base::n_basis;    // number of spatial basis
        using Base::P;          // discretized penalty matrix: P = \lambda_D*(R1^T*R0^{-1}*R1)
        using Base::W_;         // weight matrix
        using Base::XtWX_; 

        DVector<double> lambdas_D;       // smoothing parameters in space  

        // constructor
        MQSRPDE() = default;

        // MQSRPDE(const PDE_& pde, std::vector<double>& alphas = {0.1, 0.5, 0.9}) : Base(pde), alphas_(alphas) {
        //     // // Check if the provided quantile orders are an increasing sequence 
        //     // auto i = std::adjacent_find(alphas.begin(), alphas.end(), std::greater_equal<int>());
        //     // assert(i == alphas.end(), "Quantile orders must be an increasing sequence"); 
        //     h_ = alphas_.size();
        // }; 

        // space-only constructor
        template <
            typename U = RegularizationType,
            typename std::enable_if<std::is_same<U, SpaceOnly>::value, int>::type = 0>
        MQSRPDE(const Base::PDE& pde, Sampling s, std::vector<double>& alphas = {0.1, 0.5, 0.9}) : Base(pde, s), alphas_(alphas) {
            // auto i = std::adjacent_find(alphas.begin(), alphas.end(), std::greater_equal<int>());
            // if(i != alphas.end()){
            //     throw std::logic_error("Quantile orders are not in strictly ascending order");
            // }
            h_ = alphas_.size();
        };

        // getters
        const DVector<double>& f() const { return f_curr_; };            // estimate of spatial field
        const DVector<double>& g() const { return g_curr_; };            // PDE misfit
        const DVector<double>& beta() const { return beta_curr_; };      // estimate of regression coefficients
        const DMatrix<double>& H_multiple(); 
        const DMatrix<double>& Q_multiple(); 
        // const SparseBlockMatrix<double,2,2>& A_mult() const { return A_; };              // debug
        const SpMatrix<double>& Psi_mult() const { return Psi_multiple_; };                 // debug
        // const SpMatrix<double>& W_mult() const { return W_multiple_debug; };             // debug
        // const SpMatrix<double>& D_script() const { return D_script_; };                  // debug  
        // const DiagMatrix<double>& Delta_mult() const { return Delta_debug; };            // debug
        // const DiagMatrix<double>& Wbar_mult() const { return W_bar_debug; };             // debug 
        // const DMatrix<double>& Q_mult() const { return Q_multiple_; };                   // debug
        // const DMatrix<double>& H_mult_debug() const { return H_multiple_debug; };        // debug 
        // const DMatrix<double>& X_mult() const { return X_multiple_; };                   // debug 
        // const SpMatrix<double>&  Dscriptj() const { return Dscriptj_debug; };            // debug 
        // const DMatrix<double>& XtWX_multiple() const { return XtWX_multiple_debug; };    // debug

        // ModelBase implementation
        void init_model();
        void update_to_weights() { return; };   // update model object in case of changes in the weights matrix
        virtual void solve(); // finds a solution to the smoothing problem

        // Check if quantiles are crossing 
        const bool crossing_constraints() const;

        // Set the h lambdas of the QSRPDE models 
        void setLambdas_D(DMatrix<double> l){
            lambdas_D.resize(l.rows()); 
            for(auto i = 0; i < l.rows(); ++i)
                lambdas_D(i) = l(i,0);  
        }
            
        // Assemble matrices once and for all 
        void assemble_matrices(){

            // room for solution 
            f_curr_.resize(h_*n_basis());
            fn_curr_.resize(h_*n_obs());
            g_curr_.resize(h_*n_basis());
            f_init_.resize(h_*n_basis());
            fn_init_.resize(h_*n_obs());
            g_init_.resize(h_*n_basis());

            if(has_covariates()){
                beta_curr_.resize(h_*q());
                beta_init_.resize(h_*q());
            }

            // set identity matrices 
            Ih_.resize(h_, h_); 
            Ih_.setIdentity();
            In_.resize(n_obs(), n_obs()); 
            In_.setIdentity();
            Iq_.resize(q(), q()); 
            Iq_.setIdentity();
            Ihn_.resize(h_*n_obs(), h_*n_obs()); 
            Ihn_.setIdentity();

            // assemble FEM, mass and stiffness matrices
            Psi_multiple_ = Kronecker(Ih_, Psi()); 
            R0_multiple_ = Kronecker(SpMatrix<double>(DiagMatrix<double>(lambdas_D)), R0());
            R1_multiple_ = Kronecker(SpMatrix<double>(DiagMatrix<double>(lambdas_D)), R1());

            // parametric case 
            if(has_covariates()){ 
                //std::cout << "dim X() in MQSRPDE = " << X().rows() << ";" << X().cols() << std::endl; 
                X_multiple_ = Kronecker(DMatrix<double>(Ih_), X()); 
                //std::cout << "dim X_multiple_ in MQSRPDE = " << X_multiple_.rows() << ";" << X_multiple_.cols() << std::endl;
                XtWX_multiple_.resize(h_*q(), h_*q()); 
                H_multiple_.resize(n_obs()*h_, n_obs()*h_); 
                //std::cout << "dim H_multiple_ in MQSRPDE = " << H_multiple_.rows() << ";" << H_multiple_.cols() << std::endl;
                U_multiple_.resize(2*h_*n_basis(), h_*q());
                V_multiple_.resize(h_*q(), 2*h_*n_basis());
            }

            // assemble other vectors and matrices 
            l_hn_.resize(h_*n_obs()); 
            l_hn_ = DVector<double>::Zero(h_*n_obs());
            l_hn_.block(0,0, n_obs(), 1) = -DVector<double>::Ones(n_obs());
            l_hn_.block((h_-1)*n_obs(),0, n_obs(), 1) = DVector<double>::Ones(n_obs());

            SpMatrix<double> E_{};
            E_.resize(h_-1, h_); 
            std::vector<fdapde::Triplet<double>> tripletList;
            tripletList.reserve(2*(h_-1));
            for(std::size_t i = 0; i < h_-1; ++i){
                tripletList.emplace_back(i, i+1, 1.0);
                tripletList.emplace_back(i, i, -1.0);
            }
            E_.setFromTriplets(tripletList.begin(), tripletList.end());
            E_.makeCompressed();
            if(has_covariates()){
                D_ = Kronecker(E_, Iq_); 
            }       
            D_script_ = Kronecker(E_, In_);  

        }

        // pinball loss
        DVector<double> rho_alpha(const double&, const DVector<double>&) const; 
        // getter for the j-th block of the fitted vector. It returns the j-th fitted quantile 
        DVector<double> fitted(unsigned int j) const; 
        // getter for the fitted vector of all the quantiles 
        DVector<double> fitted() const; 
        // adjust the absolute residual vector with the tolerance tol_weights_
        void abs_res_adj(DVector<double>& res); 
        // returns the data loss of the model 
        double data_loss() const; 
        // return the crossing penalty value 
        double crossing_penalty() const;
        double crossing_penalty_f() const; 
        double crossing_penalty_param() const; 

        // setters
        void set_preprocess_option(bool preprocess){ do_process = preprocess;}; 
        void set_forcing_option(bool force){ force_entrance = force;}; 
        void set_max_iter(unsigned int max_it) {max_iter_ = max_it; };
        void set_gamma_init(unsigned int gamma) { gamma0_ = gamma;}; 
        void set_maxiter_fpirls_single(const unsigned int it){ fpirls_iter_ = it;}; 

        virtual ~MQSRPDE() = default;
    };

    // template <typename PDE_, typename RegularizationType_, typename SamplingDesign_, typename Solver_>
    //     struct model_traits<MQSRPDE<PDE_, RegularizationType_, SamplingDesign_, Solver_>> {
    //     typedef PDE_ PDE;
    //     typedef SpaceOnly regularization;
    //     typedef SamplingDesign_ sampling;
    //     typedef MonolithicSolver solver;
    //     enum { N = PDE::N, M = PDE::M, n_lambda = 1 };
    // };

    // // mqsrpde trait
    // template <typename Model>
    // struct is_mqsrpde { static constexpr bool value = is_instance_of<Model, MQSRPDE>::value; };


    // perform proper initialization and update of model. Computes quantites which can be reused
    // across many calls to solve() and are **not affected by a change in the data**.
    // It is implicitly called by ModelBase::init() as part of the initialization process.
    // NB: a change in the smoothing parameter must trigger a re-initialization of the model
    template <typename RegularizationType>
    void MQSRPDE<RegularizationType>::init_model() {

        // Assemble matrices
        assemble_matrices();  

        double J_init = 0.;  // degub

        // Definition of h QSRPDE models for initialization 
        for(std::size_t j = 0; j < h_; ++j){
            Sampling s = SamplingBase<This>::sampling(); 
            QSRPDE<SpaceOnly> model_j(pde(), s, alphas_[j]);

            // solver initialization
            model_j.data() = data();
            model_j.set_lambda_D(lambdas_D[j]); 
            //std::cout << "dim this->locs() in init_model = " << this->locs().rows() << std::endl; 
            model_j.set_spatial_locations(this->locs());
            model_j.set_fpirls_max_iter(fpirls_iter_); 
            // model_j.init_pde();
            // model_j.init_regularization();
            // model_j.init_sampling();    
            // model_j.init_nan();
            // model_j.init_model();
            model_j.init(); 
            model_j.solve();

            f_curr_.block(j*n_basis(), 0, n_basis(), 1) = model_j.f();
            fn_curr_.block(j*n_obs(), 0, n_obs(), 1) = Psi()*model_j.f();
            g_curr_.block(j*n_basis(), 0, n_basis(), 1) = model_j.g();
            if(has_covariates()){
                beta_curr_.block(j*q(), 0, q(), 1) = model_j.beta();
            }

            J_init += model_j.data_loss();   // degub 

        } 

        // init = curr 
        f_init_ = f_curr_; 
        fn_init_ = fn_curr_; 
        g_init_ = g_curr_; 
        if(has_covariates()){
            beta_init_ = beta_curr_;
        }

        // Processing 
        // rmk: media posta a distanza 2*eps e non eps per evitare instabilità numeriche nel caso in cui 
        // la soluzione postprocessata venga data in input all'algoritmo di MQSRPDE

        std::size_t ind_median; 
        if(do_process)
            ind_median = (find(alphas_.begin(), alphas_.end(), 0.5)) - alphas_.begin();
        //std::cout << "idx median : " << ind_median << std::endl;
        // DVector<double> fn_new = fn_curr_; // debug
        // DVector<double> fn_old = fn_curr_;  // debug 
        if(crossing_constraints() && do_process){  

            unsigned int count_iter = 0;
            while(crossing_constraints()){
                count_iter = count_iter +1;
                // if(count_iter % 1 == 0){
                //     std::cout << "################## Count_iter in processing : " << count_iter << std::endl; 
                //     std::cout << "################## crossing value " << std::setprecision(16) << crossing_penalty() << std::endl; 
                //     std::cout << "################## inf dist fn= " << std::setprecision(16) << (fn_new - fn_old).cwiseAbs().maxCoeff() << std::endl; 
                // }

                // Tentativo per semi-parametric case 
                // for(std::size_t j=ind_median; j>1; j--){
                //     auto fn_j = fn_curr_[(((j-1)*n_obs())+1):((j)*n_obs())];
                //     auto fn_jm1 = fn_curr_[(((j-2)*n_obs())+1):((j-1)*n_obs())];
                //     auto beta_j = beta_curr_[(((j-1)*q())+1):((j)*q())];
                //     auto beta_jm1 = beta_curr_[(((j-2)*q())+1):((j-1)*q())];
                //     for(std::size_t i=0; i<n_obs(); i++){  //  loop in spatial points 
                //         if(fitted(j)[i] < (fitted(j-1)[i] + eps_)){  //  if crossing (to be checked on mu!)
                //             double mean_fn = (fn_jm1[i] + fn_j[i])/2.; 
                //             fn_jm1[i] = mean_fn + eps_/4;
                //             fn_j[i] = mean_fn - eps_/4;
                //         }
                //     }
                //     for(std::size_t i=0; i<n_obs(); i++){  //  loop in spatial points 
                //         if(fitted(j)[i] < (fitted(j-1)[i] + eps_)){  //  if crossing (to be checked on mu!)
                //             double mean_beta = 0.; 
                //             for(std::size_t idx = 0; idx < q(); ++idx){
                //                 mean_beta = (beta_jm1[idx] + beta_j[idx])/2.;   // ?? vogliamo mediare le singole componenti del vettore beta??
                //                 beta_jm1[idx] = mean_beta + eps_/(4*X().coeff(i, idx));
                //                 beta_j[idx] = mean_beta - eps_/(4*X().coeff(i, idx));
                //                 {
                //                 // PROBLEMA: conta solo l'ultimo nodo di crossing per l'update di beta
                //                 }                             
                //             }
                //         }
                //     }
                //     fn_curr_[(((j-1)*n_obs())+1):((j)*n_obs())] = fn_j;
                //     fn_curr_[(((j-2)*n_obs())+1):((j-1)*n_obs())] = fn_jm1;
                //     beta_curr_[(((j-1)*q())+1):((j)*q())] = beta_j;
                //     beta_curr_[(((j-2)*q())+1):((j-1)*q())] = beta_jm1;
                // }
                // for(std::size_t j = ind_median; j < h_; j++){
                //     // ...
                // }

                for(std::size_t j=ind_median; j>=1; --j){
                    DVector<double> quantile_jm1;
                    quantile_jm1.resize(n_obs());  
                    quantile_jm1 = fitted(j-1);
                    DVector<double> quantile_j; 
                    quantile_j.resize(n_obs()); 
                    quantile_j = fitted(j);

                    DVector<double> quantile_jm1_nodes;
                    quantile_jm1_nodes.resize(n_basis());  
                    quantile_jm1_nodes = f_curr_.block((j-1)*n_basis(), 0, n_basis(), 1);
                    DVector<double> quantile_j_nodes; 
                    quantile_j_nodes.resize(n_basis()); 
                    quantile_j_nodes = f_curr_.block(j*n_basis(), 0, n_basis(), 1);

                    //  loop on spatial points 
                    for(std::size_t i=0; i<n_obs(); i++){  
                        if(quantile_j[i] < (quantile_jm1[i] + eps_)){  //  if crossing 
                            //idx_debug = i; 
                            double mean = (quantile_jm1[i] + quantile_j[i])/2;
                            quantile_jm1[i] = mean - eps_;
                            quantile_j[i] = mean + eps_;
                        }
                    }

                    //  loop on mesh nodes 
                    for(std::size_t i=0; i<n_basis(); i++){  
                        if(quantile_j_nodes[i] < (quantile_jm1_nodes[i] + eps_)){  //  if crossing 
                            double mean = (quantile_jm1_nodes[i] + quantile_j_nodes[i])/2;
                            quantile_jm1_nodes[i] = mean - eps_;
                            quantile_j_nodes[i] = mean + eps_;
                        }
                    }
                      
                    fn_curr_.block((j-1)*n_obs(),0, n_obs(),1) = quantile_jm1;
                    fn_curr_.block(j*n_obs(),0, n_obs(),1) = quantile_j;
                    f_curr_.block((j-1)*n_basis(),0, n_basis(),1) = quantile_jm1_nodes;
                    f_curr_.block(j*n_basis(),0, n_basis(),1) = quantile_j_nodes;

                }

                for(std::size_t j = ind_median; j < h_-1; ++j){  
                    DVector<double> quantile_jp1;
                    quantile_jp1.resize(n_obs());  
                    quantile_jp1 = fitted(j+1);
                    DVector<double> quantile_j;
                    quantile_j.resize(n_obs()); 
                    quantile_j = fitted(j);

                    DVector<double> quantile_jp1_nodes;
                    quantile_jp1_nodes.resize(n_basis());  
                    quantile_jp1_nodes = f_curr_.block((j+1)*n_basis(), 0, n_basis(), 1);
                    DVector<double> quantile_j_nodes;
                    quantile_j_nodes.resize(n_basis()); 
                    quantile_j_nodes = f_curr_.block(j*n_basis(), 0, n_basis(), 1);

                    //std::size_t idx_debug = 0; 

                     //  loop on spatial points 
                    for(std::size_t i=0; i<n_obs(); i++){ 
                        if(quantile_jp1[i] < (quantile_j[i] + eps_)){  //  if crossing 
                            double mean = (quantile_jp1[i] + quantile_j[i])/2;                         
                            quantile_j[i] = mean - eps_;
                            quantile_jp1[i] = mean + eps_;          
                        }
                    }

                    //  loop on mesh points 
                    for(std::size_t i=0; i<n_basis(); i++){ 
                        if(quantile_jp1_nodes[i] < (quantile_j_nodes[i] + eps_)){  //  if crossing 
                            double mean = (quantile_jp1_nodes[i] + quantile_j_nodes[i])/2;
                            quantile_j_nodes[i] = mean - eps_;
                            quantile_jp1_nodes[i] = mean + eps_;          
                        }
                    }
                      
                    fn_curr_.block((j+1)*n_obs(),0, n_obs(),1) = quantile_jp1; 
                    fn_curr_.block(j*n_obs(),0, n_obs(),1) = quantile_j; 
                    f_curr_.block((j+1)*n_basis(),0, n_basis(),1) = quantile_jp1_nodes; 
                    f_curr_.block(j*n_basis(),0, n_basis(),1) = quantile_j_nodes; 

                    
                }


                //std::cout << "------dim fn = " << fn_curr_.size() << std::endl;

                // fn_old = fn_new; 
                // fn_new = fn_curr_; 


            }

            // init <- curr 
            fn_init_ = fn_curr_;   
            f_init_ = f_curr_;     
            // fn_curr è l'unica quantità modificata (e l'unica che viene usata come inizializzazione)  
            
        }

        if(!crossing_constraints() && do_process)
            std::cout << "No crossing at the end of the processing" << std::endl ; 
        if(crossing_constraints() && do_process)
            std::cout << "---ATT: CROSSING at the end of the processing" << std::endl ; 

        // std::cout << "Range f_init_ : " << f_init_.minCoeff() << " , " << f_init_.maxCoeff() << std::endl ; 
        // std::cout << "Range g_init_ : " << g_init_.minCoeff() << " , " << g_init_.maxCoeff() << std::endl ; 
        // std::cout << "Model loss of the initializazion: " << data_loss() << " + penalty = " << g_curr_.dot(R0_multiple_*g_curr_) << std::endl;  
        // std::cout << "----- crossing global at init = " << crossing_penalty() << std::endl;
        return;
    }

    // finds a solution 
    template <typename RegularizationType>
        void MQSRPDE<RegularizationType>::solve() {

        // std::cout << "n_obs() in solve =" << n_obs() << std::endl; 
        // std::cout << "n_locs() in solve =" << n_locs() << std::endl; 
 
        // store room for W, Delta, z, t
        w_.resize(h_*n_obs()); 
        W_bar_.resize(h_*n_obs());    

        DiagMatrix<double> Delta_; 
        Delta_.resize((h_-1)*n_obs()); 

        z_.resize(h_*n_obs());

        DVector<double> t{};    
        t.resize(h_*n_obs()); 

        // initialize the crossing value 
        double crossing_penalty_init = crossing_penalty(); 

        // outer loop for the fulfilment of the crossing constraints 
        while(force_entrance || (crossing_constraints() && iter_ < max_iter_global_)){ 

            if(force_entrance)
                std::cout << "In the MQSRPDE algorithm with forced entrance" << std::endl;

            force_entrance = false; 

            std::cout << "----------------Gamma = " << gamma0_ << std::endl; 
            std::cout << "J_init = " << data_loss() + g_curr_.dot(R0_multiple_*g_curr_) + gamma0_*crossing_penalty() <<
                  "=" << data_loss() << "+" << g_curr_.dot(R0_multiple_*g_curr_) << "+" << gamma0_*crossing_penalty() << std::endl;
            
            // algorithm stops when an sufficiently small difference between two consecutive values of the J is recordered
            double J_old = tolerance_+1; double J_new = 0;
            k_ = 0;
            // inner loop for the convergence of J 
            while(k_ < max_iter_ && std::abs(J_new - J_old) > tolerance_){    

                // std::cout << "--------------------------  k_ = " << k_ << std::endl; 

                // assemble W, Delta, z 
                DVector<double> delta_((h_-1)*n_obs()); 

                for(int j = 0; j < h_; ++j){

                    // std::cout << "--------------------------  j = " << j << std::endl; 

                    DVector<double> abs_res_j;
                    DVector<double> delta_j; 
                    DVector<double> z_j;
                    
                    // compute absolute residuals (without tolerance correction)
                    abs_res_j = (y() - fitted(j)).cwiseAbs();

                    if(j < h_-1) {
                        delta_j = (2*(eps_*DVector<double>::Ones(n_obs()) - D_script_.block(j*n_obs(), 0, n_obs(), h_*n_obs())*fitted())).cwiseAbs().cwiseInverse(); 
                        //std::cout << "L inf norm abs delta j = " << (delta_j).cwiseAbs().maxCoeff() << std::endl;
                    }
                             
                    // compute pseudo-observations 
                    z_j = y() - (1 - 2*alphas_[j])*abs_res_j; 

                    // apply the weights correction for numerical stability 
                    abs_res_adj(abs_res_j);

                    // compute weights 
                    w_.block(j*n_obs(), 0, n_obs(), 1) = 2*n_obs()*abs_res_j;
  
                    // store the results in the global matrices 
                    if(j < h_-1) 
                        delta_.block(j*n_obs(), 0, n_obs(), 1) = delta_j; 


                    z_.block(j*n_obs(), 0, n_obs(), 1) = z_j;  
 
                }

                Delta_.diagonal() = delta_;
 
                W_bar_.diagonal() = w_.cwiseInverse(); 

                W_multiple_ = SpMatrix<double>(W_bar_) + gamma0_*D_script_.transpose()*Delta_*D_script_; 


                // assemble t 
                t = D_script_.transpose()*Delta_*eps_*DVector<double>::Ones((h_-1)*n_obs()) + 0.5*l_hn_; 


                // assemble nonparameteric system matrix (RMK: mass and stiffness matrices already contain lambda)
                A_ = SparseBlockMatrix<double,2,2>
                (-Psi_multiple_.transpose()*W_multiple_*Psi_multiple_,    R1_multiple_.transpose(),
                R1_multiple_,                                             R0_multiple_            );
 
                
                // cache non-parametric matrix factorization for reuse
                invA_.compute(A_);


                // prepare rhs of linear system (TODO: generalize to non zero rhs)
                b_.resize(A_.rows());
    
                // assemble b_g = (lambda_1*u, ..., lambda_h*u)' 
                DVector<double> b_g = u().replicate(h_, 1);   // u_1 = ... = u_h
  
                std::cout << "u norm inf: " << b_g.cwiseAbs().maxCoeff() << std::endl; 
                std::size_t count_lambda = 0; 

                for(std::size_t ind = 0; ind < h_*n_basis(); ind+=n_basis()){
                    // std::cout << "------------ ind = " << ind << std::endl ; 
                    // std::cout << "---- lambdas_D( " << count_lambda << " ) = " << lambdas_D(count_lambda) << std::endl ; 
                    // std::cout << "----- PRE b_g(ind) norm inf: " << b_g.block(ind,0, n_basis(),1).cwiseAbs().maxCoeff() << std::endl;  
                    
                    b_g.block(ind,0, n_basis(),1) = lambdas_D(count_lambda)*b_g.block(ind,0, n_basis(),1); 
                    // b_g(ind, 0) = lambdas_D(count_lambda)*b_g(ind, 0); 
                    // std::cout << "----- POST b_g(ind) norm inf: " << b_g.block(ind,0, n_basis(),1).cwiseAbs().maxCoeff() << std::endl;  
                    count_lambda++; 
                }

                // b_.block(h_*n_basis(),0, h_*n_basis(),1) = DVector<double>::Zero(h_*n_basis());  // b_g = 0 
                b_.block(h_*n_basis(),0, h_*n_basis(),1) = b_g; 

                // // check b_g
                // std::cout << "b_g size: " << b_g.size() << "=" << h_*n_basis() << std::endl; 
                // std::cout << "b_g norm inf: " << b_g.cwiseAbs().maxCoeff() << std::endl; 

                DVector<double> sol; // room for problem' solution      
                
                if(!has_covariates()){ // nonparametric case     

                    // update rhs of SR-PDE linear system
                    b_.block(0,0, h_*n_basis(),1) = -Psi_multiple_.transpose()*(W_bar_*z_ + gamma0_*t);

                    // solve linear system A_*x = b_
                    sol = invA_.solve(b_);

                    f_curr_ = sol.head(h_*n_basis());
                    fn_curr_ = Psi_multiple_*f_curr_; 

                } else{ // parametric case
    
 
                        XtWX_multiple_ = X_multiple_.transpose()*W_multiple_*X_multiple_;
  
                        invXtWX_multiple_ = XtWX_multiple_.partialPivLu(); 


                        // update rhs of SR-PDE linear system
                        // std::cout << "dim Psi_multiple_.transpose() = " << (Psi_multiple_.transpose()).rows() << ";" << (Psi_multiple_.transpose()).cols() << std::endl; 
                        // std::cout << "dim Ihn_ = " << Ihn_.rows() << ";" << Ihn_.cols() << std::endl;
                        // std::cout << "dim H_multiple().transpose() = " << (H_multiple().transpose()).rows() << ";" << (H_multiple().transpose()).cols() << std::endl;
                        // std::cout << "dim W_bar_ = " << W_bar_.rows() << ";" << W_bar_.cols() << std::endl;
                        // std::cout << "dim z_ = " << z_.rows() << ";" << z_.cols() << std::endl;
                        // std::cout << "dim t = " << t.rows() << ";" << t.cols() << std::endl;

                        b_.block(0,0, h_*n_basis(),1) = -Psi_multiple_.transpose()*(Ihn_ - H_multiple().transpose())*(W_bar_*z_ + gamma0_*t);  


                        // definition of matrices U and V  for application of woodbury formula
                        U_multiple_ = DMatrix<double>::Zero(2*h_*n_basis(), h_*q());
                        U_multiple_.block(0,0, h_*n_basis(), h_*q()) = Psi_multiple_.transpose()*W_multiple_*X_multiple_;
                        V_multiple_ = DMatrix<double>::Zero(h_*q(), 2*h_*n_basis());
                        V_multiple_.block(0,0, h_*q(), h_*n_basis()) = X_multiple_.transpose()*W_multiple_*Psi_multiple_;

                        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
                        sol = SMW<>().solve(invA_, U_multiple_, XtWX_multiple_, V_multiple_, b_); 
     
                        // store result of smoothing 
                        f_curr_    = sol.head(h_*n_basis());
                        fn_curr_ = Psi_multiple_*f_curr_; 
                        beta_curr_ = invXtWX_multiple_.solve(X_multiple_.transpose()*(W_bar_*z_ - W_multiple_*fn_curr_ + gamma0_*t));

 
                        
                        //std::cout << "Beta_curr: " << beta_curr_ << std::endl; 

                    }
                    // store PDE misfit
                    g_curr_ = sol.tail(h_*n_basis());

                    
                    //std::cout << "Range g_curr: " << g_curr_.minCoeff() << " , " << g_curr_.maxCoeff() << std::endl;
                    //std::cout << "Range f_curr: " << f_curr_.minCoeff() << " , " << f_curr_.maxCoeff() << std::endl; 
                    
                    // update J 
                    J_old = J_new; 
                    J_new = data_loss() + g_curr_.dot(R0_multiple_*g_curr_) + gamma0_*crossing_penalty();   // R0 multiple already contains lambdas!
                    

                    std::cout << "----- crossing global at iter k=" << k_ << " is " << crossing_penalty() << std::endl;
                    // std::cout << "----- J_old = " << J_old << std::endl;
                    std::cout << "----- J_new = " << J_new << " = " << data_loss() << " + " << g_curr_.dot(R0_multiple_*g_curr_) << std::endl;  
                    std::cout << "----- J_old - J_new = " << J_old - J_new << std::endl; 

                    // update number of inner iterations 
                    k_++;  
            }

            // update crossing value 
            double crossing_penalty_new = crossing_penalty(); 

            std::cout << "#################### cross new:  " << crossing_penalty_new << " , cross init: " << crossing_penalty_init << std::endl; 
            //std::cout << "#################### cross new-init: " << std::setprecision(10) << (crossing_penalty_new - crossing_penalty_init) << std::endl; 
            std::cout << "#################### number of inner iterations = " << k_ << std::endl;

            // update crossing penalty factor
            gamma0_ *= C_;  
            // update number of outer iterations
            iter_++;     
            
        }

    return;
    }   

    // Utilities 
    template <typename RegularizationType>
        DVector<double> MQSRPDE<RegularizationType>::fitted() const{

        DVector<double> fit = fn_curr_; 
        if(has_covariates())
            fit += X_multiple_*beta_curr_;
        return fit; 
    }

    template <typename RegularizationType>
        DVector<double> MQSRPDE<RegularizationType>::fitted(unsigned int j) const{
        // index j \in {0, ..., h-1}
        return fitted().block(j*n_obs(), 0, n_obs(), 1); 
    }

    template <typename RegularizationType>
        const bool MQSRPDE<RegularizationType>::crossing_constraints() const {
        // Return true if the current estimate of quantiles is crossing, false otherwise 
        return crossing_penalty() > 0.; 
    }

    template <typename RegularizationType>
        double MQSRPDE<RegularizationType>::data_loss() const{

        double loss = 0.; 
        for(auto j = 0; j < h_; ++j)
            loss += (rho_alpha(alphas_[j], y() - fitted(j))).sum(); 
        return loss/n_obs();

    }

    template <typename RegularizationType>
        double MQSRPDE<RegularizationType>::crossing_penalty() const{
  
        // compute value of the unpenalized unconstrained functional J: 
        double pen = 0.; 
        for(int j = 0; j < h_-1; ++j){
            pen += (eps_*DVector<double>::Ones(n_obs()) - (fitted(j+1) - fitted(j))).cwiseMax(0.).sum(); 
        }   
        return pen; 

    }

    template <typename RegularizationType>
        double MQSRPDE<RegularizationType>::crossing_penalty_f() const{
  
        // compute value of the unpenalized unconstrained functional J: 
        double pen = 0.; 
        for(int j = 0; j < h_-1; ++j){
            pen += (eps_*DVector<double>::Ones(n_obs()) - (fn_curr_.block((j+1)*n_obs(), 0, n_obs(), 1) - fn_curr_.block(j*n_obs(), 0, n_obs(), 1))).cwiseMax(0.).sum(); 
        }
        
        return pen; 
    }

    template <typename RegularizationType>
        double MQSRPDE<RegularizationType>::crossing_penalty_param() const{
  
        // compute value of the unpenalized unconstrained functional J: 
        double pen = 0.; 
        for(int j = 0; j < h_-1; ++j){
            pen += (eps_*DVector<double>::Ones(n_obs()) - (X()*(beta_curr_.block((j+1)*q(), 0, q(), 1) - beta_curr_.block(j*q(), 0, q(), 1)))).cwiseMax(0.).sum(); 

        }
    
        return pen; 
    }

    template <typename RegularizationType>
        const DMatrix<double>& MQSRPDE<RegularizationType>::H_multiple() {

        
        // compute H = X*(X^T*W*X)^{-1}*X^T*W
        H_multiple_ = X_multiple_*(invXtWX_multiple_.solve(X_multiple_.transpose())*W_multiple_);  

        return H_multiple_;
    }

    template <typename RegularizationType>
        const DMatrix<double>& MQSRPDE<RegularizationType>::Q_multiple() {
        // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
        Q_multiple_ = W_multiple_*(DMatrix<double>::Identity(n_obs()*h_, n_obs()*h_) - H_multiple());

        return Q_multiple_;
    }

    // returns the pinball loss at a specific x 
    template <typename RegularizationType>
    DVector<double> MQSRPDE<RegularizationType>::rho_alpha(const double& alpha, const DVector<double>& x) const{ 
        return 0.5*x.cwiseAbs() + (alpha - 0.5)*x; 
    }

    template <typename RegularizationType>
        void MQSRPDE<RegularizationType>::abs_res_adj(DVector<double>& res) {
            unsigned int count_debug = 1; 
            for(int i = 0; i < res.size(); ++i) {
                if(res(i) < tol_weights_) {
                    count_debug++; 
                    res(i) += tol_weights_;  
                }            
            }
    }


} // namespace models
} // namespace fdapde
    
#endif // __MQSRPDE_H__