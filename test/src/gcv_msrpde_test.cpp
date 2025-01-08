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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::diffusion;
using fdapde::core::PDE;
using fdapde::core::Triangulation;
using fdapde::core::bilaplacian;
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
using fdapde::core::Grid;

#include "../../fdaPDE/models/regression/msrpde.h"
#include "../../fdaPDE/models/regression/gcv.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::MSRPDE;
using fdapde::models::SpaceOnly;
using fdapde::models::ExactEDF;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;
using fdapde::models::SpaceTime;
using fdapde::models::SpaceTimeSeparable;
#include "../../fdaPDE/calibration/gcv.h"

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;



// helper functions
double RMSE_metric(DVector<double> v1, DVector<double> v2){
    double res = 0.; 
    if(v1.size() != v2.size())
        std::cout << std::endl << "----------ERROR IN RMSE COMPUTATION---------" << std::endl; 
    for(auto i = 0; i < v1.size(); ++i){
        res += (v1[i]-v2[i])*(v1[i]-v2[i]); 
    }
    return std::sqrt(1./(v1.size())*res); 
}

DMatrix<double> collapse_rows(DMatrix<double> m, DMatrix<double> unique_flags, unsigned int num_unique){
    DMatrix<double> m_ret; 
    m_ret.resize(num_unique, m.cols()); 
    if(unique_flags.rows() != m.rows()){
        std::cout << "problems in collapse_rows..." << std::endl; 
    }
    for(int j=0; j<m.cols(); ++j){
        unsigned int count = 0; 
        for(int i=0; i<m.rows(); ++i){
            if( unique_flags(i,0) > 1e-6){ // i.e. if unique_flags(i,0) == 1. ">tol" safer since unique_flags(i,0) is a double not bool 
                m_ret(count,j) = m(i,j); 
                count++; 
            }     
        }
    }
    return m_ret; 
}


// // test 1 
// //    domain:       c-shaped
// //    sampling:     locations = nodes
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msrpde_test1, laplacian_semiparametric_samplingatnodes_gridexact) {

//     // path test  
//     std::string test_number = "1-bis";   // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-only/Test_" + test_number;

//     const unsigned int n_sim = 50; 
//     const unsigned int sim_start = 1; 
//     const unsigned int max_fpirls_iter = 200; 
    
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  


//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE  
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // Read 
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
//     DMatrix<double> locs; 
//     if(test_number == "1-bis" || test_number == "1-tris"){
//         locs = read_csv<double>(R_path + "/locs.csv");
//         std::cout << "dim locs = " << locs.rows() << ";" << locs.cols() << std::endl;
//         std::cout << "max(locs) = " << locs.maxCoeff() << std::endl;
//     }

//     std::vector<double> lambdas; 
//     for(double x = -4.0; x <= +2.0; x += 2./3) lambdas.push_back(std::pow(10, x));
    
//     DMatrix<double> lambdas_mat;
//     lambdas_mat.resize(lambdas.size(), 1); 
//     for(auto i = 0; i < lambdas_mat.rows(); ++i){
//         // std::cout << "inserting lambdas[i] = " << std::setprecision(16) << lambdas[i] << std::endl;
//         lambdas_mat(i,0) = lambdas[i]; 
//     }
//     double best_lambda; 


//     // Simulations  
//     for(auto sim = sim_start; sim <= n_sim; ++sim){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");

//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);

//         std::string solutions_path_gcv = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 


//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 
//         enum Sampling sampling_int; 
//         if(test_number == "1-bis" || test_number == "1-tris"){
//             sampling_int = Sampling::pointwise; 
//         } else{
//             sampling_int = Sampling::mesh_nodes; 
//         }
//         MSRPDE<SpaceOnly> model_gcv(problem, sampling_int); 
        
//         // set model 
//         model_gcv.set_data(df);
//         model_gcv.set_ids_groups(ids_groups); 
//         if(test_number == "1-bis" || test_number == "1-tris"){
//             model_gcv.set_spatial_locations(locs);
//         }
//         model_gcv.set_fpirls_max_iter(max_fpirls_iter);

//         // define GCV function and grid of \lambda_D values
//         auto GCV = model_gcv.gcv<ExactEDF>();
//         // optimize GCV
//         Grid<fdapde::Dynamic> opt;
//         opt.optimize(GCV, lambdas_mat);
        
//         best_lambda = opt.optimum()(0,0);

//         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//         // Save lambda sequence 
//         std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq.csv");
//         for(std::size_t i = 0; i < lambdas.size(); ++i) 
//             fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//         fileLambdaS.close();

//         // Save lambda GCVopt for all alphas
//         std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt.csv");
//         if(fileLambdaoptS.is_open()){
//             fileLambdaoptS << std::setprecision(16) << best_lambda;
//             fileLambdaoptS.close();
//         }

//         // Save GCV 
//         std::ofstream fileGCV_scores(solutions_path_gcv + "/score.csv");
//         for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//             fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//         fileGCV_scores.close();


//         std::ofstream fileGCV_edf(solutions_path_gcv + "/edf.csv");
//         for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//             fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//         fileGCV_edf.close();



//     }
// }




// // test 2 
// //    domain:       c-shaped
// //    sampling:     locations = nodes
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msrpde_test2, laplacian_semiparametric_samplingatnodes_gridexact) {

//     // path test  
//     std::string test_number = "2";   // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-only/Test_" + test_number;

//     const unsigned int n_sim = 50; 
//     const unsigned int sim_start = 1; 
    
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  


//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE  
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // Read 
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");

//     std::vector<double> lambdas; 
//     for(double x = -3.0; x <= +2.0; x += 0.55555555) lambdas.push_back(std::pow(10, x));

    
//     DMatrix<double> lambdas_mat;
//     lambdas_mat.resize(lambdas.size(), 1); 
//     for(auto i = 0; i < lambdas_mat.rows(); ++i){
//         // std::cout << "inserting lambdas[i] = " << std::setprecision(16) << lambdas[i] << std::endl;
//         lambdas_mat(i,0) = lambdas[i]; 
//     }
//     double best_lambda; 


//     // Simulations  
//     for(auto sim = sim_start; sim <= n_sim; ++sim){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");

//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);

//         std::string solutions_path_gcv = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 


//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 

//         MSRPDE<SpaceOnly> model_gcv(problem, Sampling::mesh_nodes); 
        
//         // set model 
//         model_gcv.set_data(df);
//         model_gcv.set_ids_groups(ids_groups); 

//         // define GCV function and grid of \lambda_D values
//         auto GCV = model_gcv.gcv<ExactEDF>();
//         // optimize GCV
//         Grid<fdapde::Dynamic> opt;
//         opt.optimize(GCV, lambdas_mat);
        
//         best_lambda = opt.optimum()(0,0);

//         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//         // Save lambda sequence 
//         std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq.csv");
//         for(std::size_t i = 0; i < lambdas.size(); ++i) 
//             fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//         fileLambdaS.close();

//         // Save lambda GCVopt for all alphas
//         std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt.csv");
//         if(fileLambdaoptS.is_open()){
//             fileLambdaoptS << std::setprecision(16) << best_lambda;
//             fileLambdaoptS.close();
//         }

//         // Save GCV 
//         std::ofstream fileGCV_scores(solutions_path_gcv + "/score.csv");
//         for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//             fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//         fileGCV_scores.close();


//         std::ofstream fileGCV_edf(solutions_path_gcv + "/edf.csv");
//         for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//             fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//         fileGCV_edf.close();



//     }
// }


// // test 3 (space-time) 
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    space penalization: laplacian 
// //    time penalization: separable 
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msrpde_test3, laplacian_semiparametric_samplingatnodes_gridexact) {

//     // path test  
//     std::string test_number = "3";   // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-time/Test_" + test_number;

//     const unsigned int n_sim = 20; 
//     const unsigned int sim_start = 1; 
    
//     // define domain
//     const double t0 = 0.0;
//     const double tf = 1.0;
//     const unsigned int M = 11;  // number of time mesh nodes 
//     Triangulation<1, 1> time_mesh(t0, tf, M-1);  // interval [t0, tf] with M-1 knots
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_reduced_censoring_476");  


//     const unsigned int max_fpirls_iter = 15; 

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

//     // define regularizing PDE  in space
//     auto Ld = -laplacian<FEM>();   
//     PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    
//     // define regularizing PDE in time
//     auto Lt = -bilaplacian<SPLINE>();
//     PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);

//     // Read 
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
//     std::cout << "dim X = " << X.rows() << ";" << X.cols() << std::endl;
//     std::cout << "max(X) = " << X.maxCoeff() << std::endl; 

//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");

//     // Read locations
//     DMatrix<double> time_locs = read_csv<double>(R_path + "/time_locs.csv");  
//     std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
//     std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

//     DMatrix<double> space_locs = read_csv<double>(R_path + "/space_locs.csv");  
//     std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
//     std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

//     std::vector<double> lambdas_d; std::vector<double> lambdas_t; std::vector<DVector<double>> lambdas_d_t;
//     for(double xs = -3.5; xs <= -2.49; xs += 0.111111111111111111)
//         lambdas_d.push_back(std::pow(10,xs));

//     for(double xt = -6.0; xt <= -6.0; xt += 1.0)
//         lambdas_t.push_back(std::pow(10,xt));

//     for(auto i = 0; i < lambdas_d.size(); ++i)
//         for(auto j = 0; j < lambdas_t.size(); ++j) 
//             lambdas_d_t.push_back(SVector<2>(lambdas_d[i], lambdas_t[j]));

//     DMatrix<double> lambdas_mat(lambdas_d.size()*lambdas_t.size(), 2);
//     for(int i = 0; i < lambdas_d.size(); ++i) { 
//         for (int j = 0; j < lambdas_t.size(); ++j) {
//             lambdas_mat(i * lambdas_t.size() + j, 0) = lambdas_d[i];
//             lambdas_mat(i * lambdas_t.size() + j, 1) = lambdas_t[j];
//         }
//     }

//     // print lambdas_d in scientific notation
//     for(int i=0; i < lambdas_d.size(); ++i){
//         std::cout << "lambdas_d[" << i << "] = " << std::scientific << lambdas_d[i] << std::endl;
//     }
//     // print lambdas_t in scientific notation
//     for(int i=0; i < lambdas_t.size(); ++i){
//         std::cout << "lambdas_t[" << i << "] = " << std::scientific << lambdas_t[i] << std::endl;
//     }


//     SVector<2> best_lambda;  


//     // Simulations  
//     for(auto sim = sim_start; sim <= n_sim; ++sim){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//         std::cout << "dim y = " << y.rows() << "," << y.cols() << std::endl;

//         BlockFrame<double, int> df;
//         df.stack(OBSERVATIONS_BLK, y);           // ATT: stack for space-time data!
//         df.insert(DESIGN_MATRIX_BLK, X);         // ATT: insert for space-time covariates!
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);   // ATT: insert for space-time covariates!
                
//         std::string solutions_path_gcv = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 
//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 

//         MSRPDE<SpaceTimeSeparable> model_gcv(space_penalty, time_penalty, Sampling::pointwise);    
//         model_gcv.set_spatial_locations(space_locs);
//         model_gcv.set_temporal_locations(time_locs);
        
//         // set model 
//         model_gcv.set_data(df);
//         model_gcv.set_ids_groups(ids_groups); 

//         model_gcv.set_fpirls_max_iter(max_fpirls_iter); 

//         // define GCV function and grid of \lambda_D values
//         auto GCV = model_gcv.gcv<ExactEDF>();
//         // optimize GCV
//         Grid<fdapde::Dynamic> opt;
//         opt.optimize(GCV, lambdas_mat);
        
//         best_lambda = opt.optimum();

//         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//         // Save lambda sequence 
//         std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq_S.csv");
//         for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
//             fileLambdaS << std::setprecision(16) << lambdas_d[i] << "\n"; 
//         fileLambdaS.close();

//         std::ofstream fileLambda_T_Seq(solutions_path_gcv + "/lambdas_T_seq.csv");
//         for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
//             fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
//         fileLambda_T_Seq.close();


//         // Save lambda GCVopt for all alphas
//         std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambda_s_opt.csv");
//         if(fileLambdaoptS.is_open()){
//           fileLambdaoptS << std::setprecision(16) << best_lambda[0];
//           fileLambdaoptS.close();
//         }
//         std::ofstream fileLambdaoptT(solutions_path_gcv + "/lambda_t_opt.csv");
//         if(fileLambdaoptT.is_open()){
//           fileLambdaoptT << std::setprecision(16) << best_lambda[1];
//           fileLambdaoptT.close();
//         }

//         // Save GCV 
//         std::ofstream fileGCV_scores(solutions_path_gcv + "/score.csv");
//         std::cout << "dim GCV.gcvs() = " << GCV.gcvs().size() << std::endl;
//         for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//             fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//         fileGCV_scores.close();


//         std::ofstream fileGCV_edf(solutions_path_gcv + "/edf.csv");
//         for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//             fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//         fileGCV_edf.close();



//     }
// }


// test 4 (space-time with missing) 
//    domain:       unit square
//    sampling:     locations != nodes
//    space penalization: laplacian 
//    time penalization: separable 
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
TEST(gcv_msrpde_test4, laplacian_semiparametric_samplingatnodes_gridexact) {

    // path test  
    std::string test_number = "4";   // ATT controlla calcolo sigma in msrpde.h !!
    std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-time/Test_" + test_number;
    std::string fit_type = "fit_Zfull";   // "fit": for Z with rows to zero where missing data are present; "_Zfull": leave Z untouched
    bool zero_rows_Z_flag = true; 
    if(fit_type == "fit_Zfull"){
        zero_rows_Z_flag = false; 
    }

    const unsigned int n_sim = 20; 
    const unsigned int sim_start = 1; 
    const bool normalized_loss_flag = true;  // true to normalize the loss
    
    // define domain
    const double t0 = 0.0;
    const double tf = 1.0;
    const unsigned int M = 11;  // number of time mesh nodes 
    Triangulation<1, 1> time_mesh(t0, tf, M-1);  // interval [t0, tf] with M-1 knots
    MeshLoader<Triangulation<2, 2>> domain("unit_square_reduced_censoring_476"); 

    // rhs 
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

    // define regularizing PDE  in space
    auto Ld = -laplacian<FEM>();   
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);

    // Read  
    DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
    DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");

    // Read locations
    DMatrix<double> time_locs = read_csv<double>(R_path + "/time_locs.csv");  
    std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
    std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

    DMatrix<double> space_locs = read_csv<double>(R_path + "/space_locs.csv");  
    std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
    std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

    std::vector<double> lambdas_d; std::vector<double> lambdas_t; std::vector<DVector<double>> lambdas_d_t;
    for(double xs = -5.3; xs <= -4.0; xs += 0.1052631578947368)  // ATT: cambiato per Zfull 
        lambdas_d.push_back(std::pow(10,xs));

    for(double xt = -6.0; xt <= -6.0; xt += 1.0)
        lambdas_t.push_back(std::pow(10,xt));

    for(auto i = 0; i < lambdas_d.size(); ++i)
        for(auto j = 0; j < lambdas_t.size(); ++j) 
        lambdas_d_t.push_back(SVector<2>(lambdas_d[i], lambdas_t[j]));

    DMatrix<double> lambdas_mat(lambdas_d.size()*lambdas_t.size(), 2);
    for(int i = 0; i < lambdas_d.size(); ++i) { 
        for (int j = 0; j < lambdas_t.size(); ++j) {
        lambdas_mat(i * lambdas_t.size() + j, 0) = lambdas_d[i];
        lambdas_mat(i * lambdas_t.size() + j, 1) = lambdas_t[j];
        }
    }


    SVector<2> best_lambda;  


    // Simulations  
    for(auto sim = sim_start; sim <= n_sim; ++sim){
        std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

        // load data from .csv files
        DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
        std::cout << "dim y = " << y.rows() << "," << y.cols() << std::endl;

        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);           // ATT: stack for space-time data!
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);   // ATT: insert for space-time covariates!
                
        std::string solutions_path_gcv = R_path + "/simulations/sim_" + std::to_string(sim) + "/" + fit_type; 
        std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/" + fit_type; 

        MSRPDE<SpaceTimeSeparable> model_gcv(space_penalty, time_penalty, Sampling::pointwise);    
        model_gcv.set_spatial_locations(space_locs);
        model_gcv.set_temporal_locations(time_locs);
        model_gcv.set_normalize_loss(normalized_loss_flag);
        model_gcv.set_miss_rows_Z_to_zero(zero_rows_Z_flag);
        
        // set model 
        model_gcv.set_data(df);
        model_gcv.set_ids_groups(ids_groups); 

        model_gcv.set_fpirls_max_iter(15); 

        // define GCV function and grid of \lambda_D values
        auto GCV = model_gcv.gcv<ExactEDF>();
        // optimize GCV
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas_mat);
        
        best_lambda = opt.optimum();

        std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

        // Save lambda sequence 
        std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq_S.csv");
        for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
            fileLambdaS << std::setprecision(16) << lambdas_d[i] << "\n"; 
        fileLambdaS.close();

        std::ofstream fileLambda_T_Seq(solutions_path_gcv + "/lambdas_T_seq.csv");
        for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
            fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
        fileLambda_T_Seq.close();


        // Save lambda GCVopt for all alphas
        std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambda_s_opt.csv");
        if(fileLambdaoptS.is_open()){
          fileLambdaoptS << std::setprecision(16) << best_lambda[0];
          fileLambdaoptS.close();
        }
        std::ofstream fileLambdaoptT(solutions_path_gcv + "/lambda_t_opt.csv");
        if(fileLambdaoptT.is_open()){
          fileLambdaoptT << std::setprecision(16) << best_lambda[1];
          fileLambdaoptT.close();
        }

        // Save GCV 
        std::ofstream fileGCV_scores(solutions_path_gcv + "/score.csv");
        std::cout << "dim GCV.gcvs() = " << GCV.gcvs().size() << std::endl;
        for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
            fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
        fileGCV_scores.close();


        std::ofstream fileGCV_edf(solutions_path_gcv + "/edf.csv");
        for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
            fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
        fileGCV_edf.close();



    }
}


// // test 5 (osservazioni ripetute) 
// //    domain:       c-shaped
// //    sampling:     locations = nodes (ma obs.rip -> sampling:pointwise)
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msrpde_test5, laplacian_semiparametric_samplingatlocs_gridexact) {

//     // path test  
//     std::string test_number = "5";  
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-only/Test_" + test_number;

//     const unsigned int n_sim = 20; 
//     const unsigned int sim_start = 11; 

//     const unsigned int max_iter_fpirls = 15; 
//     const bool normalized_loss_flag = true;  // true to normalize the loss 
//     std::string norm_loss_str = ""; 
//     if(normalized_loss_flag){
//         norm_loss_str = "_norm"; 
//     }

//     const bool has_covariates = true; 
//     DVector<double> beta_true; 
//     beta_true.resize(2); 
//     beta_true << -2.0, -1.0; 
    
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  


//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE  
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     std::vector<double> lambdas; 

//     if(!normalized_loss_flag)
//         for(double x = -3.5; x <= +2.0; x += 0.5) lambdas.push_back(std::pow(10, x));
//     if(normalized_loss_flag)
//         for(double x = -3.5; x <= +2.0; x += 0.5) lambdas.push_back(std::pow(10, x));
    
//     DMatrix<double> lambdas_mat;
//     lambdas_mat.resize(lambdas.size(), 1); 
//     for(auto i = 0; i < lambdas_mat.rows(); ++i){
//         lambdas_mat(i,0) = lambdas[i]; 
//     }
//     double best_lambda; 


//     std::vector<std::string> data_types =  {"data", "data_rip_10", "data_rip_50"};

//     // Simulations  
//     for(auto sim = sim_start; sim <= n_sim; ++sim){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         for(std::string data_type : data_types){

//             std::string data_path = R_path + "/" + data_type; 

//             std::string solutions_path_gcv = data_path + "/simulations/sim_" + std::to_string(sim) + "/GCV"; 
//             std::string solution_path_rmse = data_path + "/simulations/sim_" + std::to_string(sim) + "/RMSE";

//             // load data from .csv files
//             DMatrix<double> y = read_csv<double>(data_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");

//             // Read 
//             DMatrix<double> X;
//             if(has_covariates) 
//                 X = read_csv<double>(data_path + "/X.csv");

//             DMatrix<double> Z = read_csv<double>(data_path + "/Z.csv");  
//             DVector<unsigned int> ids_groups = read_csv<unsigned int>(data_path + "/ids_groups.csv");
//             DMatrix<double> locs = read_csv<double>(data_path + "/locs.csv");

//             BlockFrame<double, int> df;
//             df.insert(OBSERVATIONS_BLK, y);
//             df.insert(DESIGN_MATRIX_BLK, X);
//             df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);


//             std::cout << "------------------GCV selection-----------------" << std::endl;


//             MSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise); 
            
//             // set model 
//             model_gcv.set_data(df);
//             model_gcv.set_ids_groups(ids_groups); 
//             model_gcv.set_spatial_locations(locs);
//             model_gcv.set_fpirls_max_iter(max_iter_fpirls);
//             model_gcv.set_normalize_loss(normalized_loss_flag);

//             // define GCV function and grid of \lambda_D values
//             auto GCV = model_gcv.gcv<ExactEDF>();
//             // optimize GCV
//             Grid<fdapde::Dynamic> opt;
//             opt.optimize(GCV, lambdas_mat);
            
//             best_lambda = opt.optimum()(0,0);

//             std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//             // Save lambda sequence 
//             std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq" + norm_loss_str + ".csv");
//             for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                 fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//             fileLambdaS.close();

//             // Save lambda GCVopt for all alphas
//             std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt" + norm_loss_str + ".csv");
//             if(fileLambdaoptS.is_open()){
//                 fileLambdaoptS << std::setprecision(16) << best_lambda;
//                 fileLambdaoptS.close();
//             }

//             // Save GCV 
//             std::ofstream fileGCV_scores(solutions_path_gcv + "/score" + norm_loss_str + ".csv");
//             for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                 fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//             fileGCV_scores.close();




//             std::cout << "------------------RMSE selection-----------------" << std::endl;

//             DMatrix<double> sol_true_unique = read_csv<double>(R_path + "/true/mu_true_sim_" + std::to_string(sim) + ".csv");      // without repetitions (fn + X\beta + Zb)
//             // nota: sol_true_unique dipende anche da sim perchè c'è la random part 
//             DMatrix<double> X_collapsed = read_csv<double>(R_path + "/data/X.csv");        // without repetitions 
//             DMatrix<double> unique_flags = read_csv<double>(data_path + "/unique_flags.csv"); 

//             DMatrix<double> locs_unique = read_csv<double>(R_path + "/data/locs.csv"); 
//             unsigned int num_unique = locs_unique.rows(); 
//             std::cout << "num_unique for " << data_type << ", is " << num_unique << std::endl;


//             std::vector<double> rmse_score; 
//             rmse_score.resize(lambdas.size()); 
//             unsigned int count_l = 0; 
//             for(auto lambda : lambdas){

//                 MSRPDE<SpaceOnly> model_rmse(problem, Sampling::pointwise); 
            
//                 // set model 
//                 model_rmse.set_data(df);
//                 model_rmse.set_ids_groups(ids_groups); 
//                 model_rmse.set_spatial_locations(locs);
//                 model_rmse.set_fpirls_max_iter(max_iter_fpirls);
//                 model_rmse.set_lambda_D(lambda);   
//                 model_rmse.set_normalize_loss(normalized_loss_flag);       
                
//                 model_rmse.set_data(df);

//                 model_rmse.init();
//                 model_rmse.solve();
                
//                 // std::cout << "RMSE selection: end solve" << std::endl;
//                 // std::cout << "size of model_rmse.random_part() = " << model_rmse.random_part().size() << std::endl;
//                 // std::cout << "collapse_rows(model_rmse.Psi()*model_rmse.f(), unique_flags, num_unique) = " << collapse_rows(model_rmse.Psi()*model_rmse.f(), unique_flags, num_unique).size() << std::endl;
//                 // std::cout << "size of X_collapsed*model_rmse.beta() = " << (X_collapsed*model_rmse.beta()).size() << std::endl;
                
//                 DVector<double> sol = collapse_rows(model_rmse.Psi()*model_rmse.f() + model_rmse.random_part(), unique_flags, num_unique); 
//                 if(has_covariates)
//                     sol += X_collapsed*model_rmse.beta();   

//                 std::cout << "size of sol_true_unique = " << sol_true_unique.size() << std::endl;
//                 std::cout << "calling RMSE_metric" << std::endl;
//                 rmse_score[count_l] = RMSE_metric(sol, sol_true_unique); 

//                 count_l = count_l+1; 
//             }

//             std::cout << "calling distance" << std::endl;
//             auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
            
//             // Save lambda sequence 
//             std::ofstream fileLambdaS_rmse(solution_path_rmse + "/lambdas_seq" + norm_loss_str + ".csv");
//             for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                 fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
//             fileLambdaS_rmse.close();

//             // Save lambda RMSEopt for all alphas
//             std::ofstream fileLambdaoptS_rmse(solution_path_rmse + "/lambdas_opt" + norm_loss_str + ".csv");
//             if(fileLambdaoptS_rmse.is_open()){
//                 fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
//                 fileLambdaoptS_rmse.close();
//             }

//             // Save score 
//             std::ofstream fileRMSE_scores(solution_path_rmse + "/score" + norm_loss_str + ".csv");
//             for(std::size_t i = 0; i < rmse_score.size(); ++i) 
//                 fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
//             fileRMSE_scores.close();
        
//         }    

    


//     }
// }