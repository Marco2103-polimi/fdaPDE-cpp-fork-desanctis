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
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::SPLINE;
using fdapde::core::bilaplacian;
using fdapde::core::fem_order;
using fdapde::core::spline_order;

using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "../../fdaPDE/models/regression/msrpde.h"
#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SpaceOnly;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;
using fdapde::models::SRPDE;
using fdapde::models::STRPDE;
using fdapde::models::MSRPDE;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;



// // test 1 
// //    domain:       c-shaped
// //    sampling:     locations = nodes
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// TEST(msrpde_test1, laplacian_semiparametric_samplingatnodes) {
    
//     // path test  
//     std::string test_number = "1-bis";  // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-only/Test_" + test_number;

//     const unsigned int n_sim = 50; 
//     const unsigned int sim_start = 1; 
//     const bool debug = false; 
//     const unsigned int max_fpirls_iter = 200; 
    
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  
//     // std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE  
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // Read covariates
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
//     std::cout << "dim X = " << X.rows() << ";" << X.cols() << std::endl;
//     std::cout << "max(X) = " << X.maxCoeff() << std::endl;   

//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
//     std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
//     std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
//     std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 

//     DMatrix<double> locs; 
//     if(test_number == "1-bis" || test_number == "1-tris"){
//         locs = read_csv<double>(R_path + "/locs.csv");
//         std::cout << "dim locs = " << locs.rows() << ";" << locs.cols() << std::endl;
//         std::cout << "max(locs) = " << locs.maxCoeff() << std::endl;
//     }

//     double lambda; 
    
//     for(auto sim = sim_start; sim <= n_sim; ++sim){

//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//         std::cout << "dim y = " << y.size() << std::endl;
//         std::cout << "max(y) = " << y.maxCoeff() << std::endl; 

//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);

                    
//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 

//         enum Sampling sampling_int; 
//         if(test_number == "1-bis" || test_number == "1-tris"){
//             sampling_int = Sampling::pointwise;   // pointwise
//         } else{
//             sampling_int = Sampling::mesh_nodes;   // mesh nodes
//         }

//         MSRPDE<SpaceOnly> model(problem, sampling_int); 

//         // set model 
//         model.set_data(df);
//         model.set_ids_groups(ids_groups); 
//         if(test_number == "1-bis" || test_number == "1-tris"){
//             model.set_spatial_locations(locs);
//         }
//         model.set_fpirls_max_iter(max_fpirls_iter);

//         // read lambda 
//         std::ifstream fileLambda(solution_path + "/lambdas_opt.csv");
//         if(fileLambda.is_open()){
//             fileLambda >> lambda; 
//             std::cout << "lambda=" << lambda << std::endl; 
//             fileLambda.close();
//         }
//         // std::cout << "ATT: forcing lambda..." << std::endl; 
//         // lambda = 10.; 
        
//         model.set_lambda_D(lambda);

//         // std::cout << "ATT: forcing max iter fpirls..." << std::endl; 
//         // model.set_fpirls_max_iter(2); 
        

//         // solve smoothing problem
//         //std::cout << "model init in test" << std::endl;
//         model.init();
//         //std::cout << "model solve in test" << std::endl;
//         model.solve();
//         //std::cout << "model end solve in test" << std::endl;

//         // Save solution
//         DMatrix<double> computedF = model.f();
//         // std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
//         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filef(solution_path + "/f.csv");
//         if(filef.is_open()){
//             filef << computedF.format(CSVFormatf);
//             filef.close();
//         }

//         DMatrix<double> computedFn = model.Psi()*model.f();
//         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filefn(solution_path + "/fn.csv");
//         if(filefn.is_open()){
//             filefn << computedFn.format(CSVFormatfn);
//             filefn.close();
//         }

//         DMatrix<double> computedBeta = model.beta();
//         const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filebeta(solution_path + "/beta.csv");
//         if(filebeta.is_open()){
//             filebeta << computedBeta.format(CSVFormatbeta);
//             filebeta.close();
//         }

//         std::vector<DVector<double>> temp_bhat = model.b_hat(); 
//         DMatrix<double> computed_b;
//         computed_b.resize(temp_bhat.size(), model.p());   // m x p
//         for(int i=0; i<temp_bhat.size(); ++i){
//             computed_b.row(i) = temp_bhat[i]; 
//         }
//         const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream fileb(solution_path + "/b_random.csv");
//         if(fileb.is_open()){
//             fileb << computed_b.format(CSVFormatb);
//             fileb.close();
//         }

//         double computedsigmahat = std::sqrt(model.sigma_sq_hat());
//         std::ofstream filesigmahat(solution_path + "/sigma_hat.csv");
//         if(filesigmahat.is_open()){
//             filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
//             filesigmahat.close();
//         }

//         DMatrix<double> computedsigma_b_hat = model.Sigma_b();
//         const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filesigma_b_hat(solution_path + "/Sigma_b_hat.csv");
//         if(filesigma_b_hat.is_open()){
//             filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
//             filesigma_b_hat.close();
//         }

//         if(debug){

//             unsigned int computediter = model.n_inter_fpirls();
//             std::ofstream fileiter(solution_path + "/n_iter.csv");
//             if(fileiter.is_open()){
//                 fileiter << computediter << "\n";
//                 fileiter.close();
//             }

//             double computedminJ = model.min_J();
//             std::ofstream fileminJ(solution_path + "/minJ.csv");
//             if(fileminJ.is_open()){
//                 fileminJ << std::setprecision(16) << computedminJ << "\n";
//                 fileminJ.close();
//             }

//             DVector<DMatrix<double>> computedZ = model.Z_debug();
//             for(int j=0; j<computedZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZ_j(solution_path + "/Z_" + std::to_string(j+1) + ".csv");
//                 if(fileZ_j.is_open()){
//                     fileZ_j << computedZ(j).format(CSVFormatZ_j);
//                     fileZ_j.close();
//                 }
//             }

//             DVector<DMatrix<double>> computedZTZ = model.ZTZ();
//             for(int j=0; j<computedZTZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZTZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZTZ_j(solution_path + "/ZTZ_" + std::to_string(j+1) + ".csv");
//                 if(fileZTZ_j.is_open()){
//                     fileZTZ_j << computedZTZ(j).format(CSVFormatZTZ_j);
//                     fileZTZ_j.close();
//                 }
//             }

//             DMatrix<double> computedW = model.pW_init();
//             const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileW(solution_path + "/W0.csv");
//             if(fileW.is_open()){
//                 fileW << computedW.format(CSVFormatW);
//                 fileW.close();
//             }

//             DMatrix<double> computedDelta0 = model.Delta0_debug();
//             const static Eigen::IOFormat CSVFormatDelta0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileDelta0(solution_path + "/Delta0.csv");
//             if(fileDelta0.is_open()){
//                 fileDelta0 << computedDelta0.format(CSVFormatDelta0);
//                 fileDelta0.close();
//             }

            
//             DMatrix<double> computePsi = model.Psi_debug();
//             const static Eigen::IOFormat CSVFormatPsi(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filePsi(solution_path + "/Psi.csv");
//             if(filePsi.is_open()){
//                 filePsi << computePsi.format(CSVFormatPsi);
//                 filePsi.close();
//             }

//             DMatrix<double> computeR0 = model.R0_debug();
//             const static Eigen::IOFormat CSVFormatR0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR0(solution_path + "/R0.csv");
//             if(fileR0.is_open()){
//                 fileR0 << computeR0.format(CSVFormatR0);
//                 fileR0.close();
//             }

//             DMatrix<double> computeR1 = model.R1_debug();
//             const static Eigen::IOFormat CSVFormatR1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR1(solution_path + "/R1.csv");
//             if(fileR1.is_open()){
//                 fileR1 << computeR1.format(CSVFormatR1);
//                 fileR1.close();
//             }
//         }


//     }


// }



// // test 2
// //    domain:       c-shaped
// //    sampling:     locations = nodes
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// TEST(msrpde_test2, laplacian_semiparametric_samplingatnodes) {
    
//     // path test  
//     std::string test_number = "2";  // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-only/Test_" + test_number;

//     const unsigned int n_sim = 50; 
//     const unsigned int sim_start = 1; 
//     const bool debug = false; 
    
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  
//     // std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE  
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // Read covariates
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
//     std::cout << "dim X = " << X.rows() << ";" << X.cols() << std::endl;
//     std::cout << "max(X) = " << X.maxCoeff() << std::endl;   

//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
//     std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
//     std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
//     std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 

//     double lambda; 
    
//     for(auto sim = sim_start; sim <= n_sim; ++sim){

//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//         std::cout << "dim y = " << y.size() << std::endl;
//         std::cout << "max(y) = " << y.maxCoeff() << std::endl; 

//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);

                    
//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit";

//         MSRPDE<SpaceOnly> model(problem, Sampling::mesh_nodes); 

//         // set model 
//         model.set_data(df);
//         model.set_ids_groups(ids_groups); 

//         // read lambda 
//         std::ifstream fileLambda(solution_path + "/lambdas_opt.csv");
//         if(fileLambda.is_open()){
//             fileLambda >> lambda; 
//             std::cout << "lambda=" << lambda << std::endl; 
//             fileLambda.close();
//         }
//         // std::cout << "ATT: forcing lambda..." << std::endl; 
//         // lambda = 10.; 
        
//         model.set_lambda_D(lambda);

//         // std::cout << "ATT: forcing max iter fpirls..." << std::endl; 
//         // model.set_fpirls_max_iter(2); 
        

//         // solve smoothing problem
//         //std::cout << "model init in test" << std::endl;
//         model.init();
//         //std::cout << "model solve in test" << std::endl;
//         model.solve();
//         //std::cout << "model end solve in test" << std::endl;

//         // Save solution
//         DMatrix<double> computedF = model.f();
//         // std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
//         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filef(solution_path + "/f.csv");
//         if(filef.is_open()){
//             filef << computedF.format(CSVFormatf);
//             filef.close();
//         }

//         DMatrix<double> computedFn = model.Psi()*model.f();
//         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filefn(solution_path + "/fn.csv");
//         if(filefn.is_open()){
//             filefn << computedFn.format(CSVFormatfn);
//             filefn.close();
//         }

//         DMatrix<double> computedBeta = model.beta();
//         const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filebeta(solution_path + "/beta.csv");
//         if(filebeta.is_open()){
//             filebeta << computedBeta.format(CSVFormatbeta);
//             filebeta.close();
//         }

//         std::vector<DVector<double>> temp_bhat = model.b_hat(); 
//         DMatrix<double> computed_b;
//         computed_b.resize(temp_bhat.size(), model.p());   // m x p
//         for(int i=0; i<temp_bhat.size(); ++i){
//             computed_b.row(i) = temp_bhat[i]; 
//         }
//         const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream fileb(solution_path + "/b_random.csv");
//         if(fileb.is_open()){
//             fileb << computed_b.format(CSVFormatb);
//             fileb.close();
//         }

//         double computedsigmahat = std::sqrt(model.sigma_sq_hat());
//         std::ofstream filesigmahat(solution_path + "/sigma_hat.csv");
//         if(filesigmahat.is_open()){
//             filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
//             filesigmahat.close();
//         }

//         DMatrix<double> computedsigma_b_hat = model.Sigma_b();
//         const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filesigma_b_hat(solution_path + "/Sigma_b_hat.csv");
//         if(filesigma_b_hat.is_open()){
//             filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
//             filesigma_b_hat.close();
//         }

//         if(debug){

//             unsigned int computediter = model.n_inter_fpirls();
//             std::ofstream fileiter(solution_path + "/n_iter.csv");
//             if(fileiter.is_open()){
//                 fileiter << computediter << "\n";
//                 fileiter.close();
//             }

//             double computedminJ = model.min_J();
//             std::ofstream fileminJ(solution_path + "/minJ.csv");
//             if(fileminJ.is_open()){
//                 fileminJ << std::setprecision(16) << computedminJ << "\n";
//                 fileminJ.close();
//             }

//             DVector<DMatrix<double>> computedZ = model.Z_debug();
//             for(int j=0; j<computedZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZ_j(solution_path + "/Z_" + std::to_string(j+1) + ".csv");
//                 if(fileZ_j.is_open()){
//                     fileZ_j << computedZ(j).format(CSVFormatZ_j);
//                     fileZ_j.close();
//                 }
//             }

//             DVector<DMatrix<double>> computedZTZ = model.ZTZ();
//             for(int j=0; j<computedZTZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZTZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZTZ_j(solution_path + "/ZTZ_" + std::to_string(j+1) + ".csv");
//                 if(fileZTZ_j.is_open()){
//                     fileZTZ_j << computedZTZ(j).format(CSVFormatZTZ_j);
//                     fileZTZ_j.close();
//                 }
//             }

//             DMatrix<double> computedW = model.pW_init();
//             const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileW(solution_path + "/W0.csv");
//             if(fileW.is_open()){
//                 fileW << computedW.format(CSVFormatW);
//                 fileW.close();
//             }

//             DMatrix<double> computedDelta0 = model.Delta0_debug();
//             const static Eigen::IOFormat CSVFormatDelta0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileDelta0(solution_path + "/Delta0.csv");
//             if(fileDelta0.is_open()){
//                 fileDelta0 << computedDelta0.format(CSVFormatDelta0);
//                 fileDelta0.close();
//             }

            
//             DMatrix<double> computePsi = model.Psi_debug();
//             const static Eigen::IOFormat CSVFormatPsi(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filePsi(solution_path + "/Psi.csv");
//             if(filePsi.is_open()){
//                 filePsi << computePsi.format(CSVFormatPsi);
//                 filePsi.close();
//             }

//             DMatrix<double> computeR0 = model.R0_debug();
//             const static Eigen::IOFormat CSVFormatR0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR0(solution_path + "/R0.csv");
//             if(fileR0.is_open()){
//                 fileR0 << computeR0.format(CSVFormatR0);
//                 fileR0.close();
//             }

//             DMatrix<double> computeR1 = model.R1_debug();
//             const static Eigen::IOFormat CSVFormatR1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR1(solution_path + "/R1.csv");
//             if(fileR1.is_open()){
//                 fileR1 << computeR1.format(CSVFormatR1);
//                 fileR1.close();
//             }
//         }


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
// TEST(msrpde_test3, laplacian_semiparametric_samplingatnodes) {
    
//     // path test  
//     std::string test_number = "3";  // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-time/Test_" + test_number;

//     const unsigned int n_sim = 1; 
//     const unsigned int sim_start = 1; 
//     const bool debug = false; 
    
//     // define domain
//     const double t0 = 0.0;
//     const double tf = 1.0;
//     const unsigned int M = 11;   // number of time mesh nodes    
//     Triangulation<1, 1> time_mesh(t0, tf, M-1);
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_reduced_censoring_476");    
//     // std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

//     const unsigned int max_fpirls_iter = 15; 

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

//     // define regularizing PDE in space
//     auto Ld = -laplacian<FEM>();   
//     PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    
//     // define regularizing PDE in time
//     auto Lt = -bilaplacian<SPLINE>();
//     PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);

//     // Read covariates
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");

//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
//     std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
//     std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
//     std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 

//     // Read locations
//     DMatrix<double> time_locs = read_csv<double>(R_path + "/time_locs.csv");  
//     std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
//     std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

//     DMatrix<double> space_locs = read_csv<double>(R_path + "/space_locs.csv");  
//     std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
//     std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

//     double lambda_space; double lambda_time; 
    
//     for(auto sim = sim_start; sim <= n_sim; ++sim){

//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//         std::cout << "dim y = " << y.rows() << "," << y.cols() << std::endl;

//         // check number of missing values
//         int count_na = 0;
//         double max_y = -1000.; // for debug
//         for (int i = 0; i < y.rows(); ++i) {
//             for (int j = 0; j < y.cols(); ++j) {
//                 if(std::isnan(y(i,j))) {
//                     ++count_na;
//                 } else{
//                     if(y(i,j) > max_y)
//                         max_y = y(i,j);
//                 }
//             }
//         }
//         std::cout << "num missing values = " << count_na << std::endl;
//         std::cout << "max y = " << max_y << std::endl;

//         BlockFrame<double, int> df;
//         df.stack(OBSERVATIONS_BLK, y);           // ATT: stack for space-time data!
//         df.insert(DESIGN_MATRIX_BLK, X);         // ATT: insert for space-time covariates!
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);   // ATT: insert for space-time covariates!

                    
//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit";

//         MSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise);    
//         model.set_spatial_locations(space_locs);
//         model.set_temporal_locations(time_locs);

//         // set model 
//         model.set_data(df);
//         model.set_ids_groups(ids_groups); 

//         // // read lambdas
//         double lambda_D;  
//         double lambda_T;  

//         std::ifstream fileLambdaS_gcv(solution_path + "/lambda_s_opt.csv");
//         if(fileLambdaS_gcv.is_open()){
//             fileLambdaS_gcv >> lambda_D; 
//             fileLambdaS_gcv.close();
//         }
//         std::ifstream fileLambdaT(solution_path + "/lambda_t_opt.csv");
//         if(fileLambdaT.is_open()){
//             fileLambdaT >> lambda_T; 
//             fileLambdaT.close();
//         }

//         // std::cout << "ATT: forcing lambdas..." << std::endl; 
//         // lambda_D = 1e-5; 
//         // lambda_T = 1e-5; 
        
//         model.set_lambda_D(lambda_D);
//         model.set_lambda_T(lambda_T);

//         model.set_fpirls_max_iter(max_fpirls_iter); 
        

//         // solve smoothing problem
//         //std::cout << "model init in test" << std::endl;
//         model.init();
//         //std::cout << "model solve in test" << std::endl;
//         model.solve();
//         //std::cout << "model end solve in test" << std::endl;

//         // Save solution
//         DMatrix<double> computedF = model.f();
//         // std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
//         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filef(solution_path + "/f.csv");
//         if(filef.is_open()){
//             filef << computedF.format(CSVFormatf);
//             filef.close();
//         }

//         DMatrix<double> computedFn = model.Psi()*model.f();
//         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filefn(solution_path + "/fn.csv");
//         if(filefn.is_open()){
//             filefn << computedFn.format(CSVFormatfn);
//             filefn.close();
//         }

//         DMatrix<double> computedBeta = model.beta();
//         const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filebeta(solution_path + "/beta.csv");
//         if(filebeta.is_open()){
//             filebeta << computedBeta.format(CSVFormatbeta);
//             filebeta.close();
//         }        

//         std::vector<DVector<double>> temp_bhat = model.b_hat(); 
//         DMatrix<double> computed_b;
//         computed_b.resize(temp_bhat.size(), model.p());   // m x p
//         for(int i=0; i<temp_bhat.size(); ++i){
//             computed_b.row(i) = temp_bhat[i]; 
//         }
//         const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream fileb(solution_path + "/b_random.csv");
//         if(fileb.is_open()){
//             fileb << computed_b.format(CSVFormatb);
//             fileb.close();
//         }

//         double computedsigmahat = std::sqrt(model.sigma_sq_hat());
//         std::ofstream filesigmahat(solution_path + "/sigma_hat.csv");
//         if(filesigmahat.is_open()){
//             filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
//             filesigmahat.close();
//         }

//         DMatrix<double> computedsigma_b_hat = model.Sigma_b();
//         const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filesigma_b_hat(solution_path + "/Sigma_b_hat.csv");
//         if(filesigma_b_hat.is_open()){
//             filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
//             filesigma_b_hat.close();
//         }

//         if(debug){

//             unsigned int computediter = model.n_inter_fpirls();
//             std::ofstream fileiter(solution_path + "/n_iter.csv");
//             if(fileiter.is_open()){
//                 fileiter << computediter << "\n";
//                 fileiter.close();
//             }

//             double computedminJ = model.min_J();
//             std::ofstream fileminJ(solution_path + "/minJ.csv");
//             if(fileminJ.is_open()){
//                 fileminJ << std::setprecision(16) << computedminJ << "\n";
//                 fileminJ.close();
//             }

//             DVector<DMatrix<double>> computedZ = model.Z_debug();
//             for(int j=0; j<computedZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZ_j(solution_path + "/Z_" + std::to_string(j+1) + ".csv");
//                 if(fileZ_j.is_open()){
//                     fileZ_j << computedZ(j).format(CSVFormatZ_j);
//                     fileZ_j.close();
//                 }
//             }

//             DVector<DMatrix<double>> computedZTZ = model.ZTZ();
//             for(int j=0; j<computedZTZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZTZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZTZ_j(solution_path + "/ZTZ_" + std::to_string(j+1) + ".csv");
//                 if(fileZTZ_j.is_open()){
//                     fileZTZ_j << computedZTZ(j).format(CSVFormatZTZ_j);
//                     fileZTZ_j.close();
//                 }
//             }

//             DMatrix<double> computedW = model.pW_init();
//             const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileW(solution_path + "/W0.csv");
//             if(fileW.is_open()){
//                 fileW << computedW.format(CSVFormatW);
//                 fileW.close();
//             }

//             DMatrix<double> computedDelta0 = model.Delta0_debug();
//             const static Eigen::IOFormat CSVFormatDelta0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileDelta0(solution_path + "/Delta0.csv");
//             if(fileDelta0.is_open()){
//                 fileDelta0 << computedDelta0.format(CSVFormatDelta0);
//                 fileDelta0.close();
//             }

            
//             DMatrix<double> computePsi = model.Psi_debug();
//             const static Eigen::IOFormat CSVFormatPsi(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filePsi(solution_path + "/Psi.csv");
//             if(filePsi.is_open()){
//                 filePsi << computePsi.format(CSVFormatPsi);
//                 filePsi.close();
//             }

//             DMatrix<double> computeR0 = model.R0_debug();
//             const static Eigen::IOFormat CSVFormatR0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR0(solution_path + "/R0.csv");
//             if(fileR0.is_open()){
//                 fileR0 << computeR0.format(CSVFormatR0);
//                 fileR0.close();
//             }

//             DMatrix<double> computeR1 = model.R1_debug();
//             const static Eigen::IOFormat CSVFormatR1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR1(solution_path + "/R1.csv");
//             if(fileR1.is_open()){
//                 fileR1 << computeR1.format(CSVFormatR1);
//                 fileR1.close();
//             }

//             DMatrix<double> computePenT = model.PT_debug();
//             const static Eigen::IOFormat CSVFormatPenT(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filePenT(solution_path + "/penT.csv");
//             if(filePenT.is_open()){
//                 filePenT << computePenT.format(CSVFormatPenT);
//                 filePenT.close();
//             }


//             DMatrix<double> computeR0_space = model.R0_space_debug();
//             const static Eigen::IOFormat CSVFormatR0_space(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR0_space(solution_path + "/space_mass.csv");
//             if(fileR0_space.is_open()){
//                 fileR0_space << computeR0_space.format(CSVFormatR0_space);
//                 fileR0_space.close();
//             }


//             DMatrix<double> computeR1_space = model.R1_space_debug();
//             const static Eigen::IOFormat CSVFormatR1_space(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR1_space(solution_path + "/space_stiff.csv");
//             if(fileR1_space.is_open()){
//                 fileR1_space << computeR1_space.format(CSVFormatR1_space);
//                 fileR1_space.close();
//             }


//             DMatrix<double> computeP0 = model.P0_debug();
//             const static Eigen::IOFormat CSVFormatP0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileP0(solution_path + "/P0.csv");
//             if(fileP0.is_open()){
//                 fileP0 << computeP0.format(CSVFormatP0);
//                 fileP0.close();
//             }

//             DMatrix<double> computeP1 = model.P1_debug();
//             const static Eigen::IOFormat CSVFormatP1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileP1(solution_path + "/P1.csv");
//             if(fileP1.is_open()){
//                 fileP1 << computeP1.format(CSVFormatP1);
//                 fileP1.close();
//             }
            


//         }


//     }


// }


// // test 4 (space-time with missing)
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    space penalization: laplacian 
// //    time penalization: separable 
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// TEST(msrpde_test4, laplacian_semiparametric_samplingatnodes) {
    
//     // path test  
//     std::string test_number = "4";  // ATT controlla calcolo sigma in msrpde.h !!
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-time/Test_" + test_number;
//     std::string fit_type = "fit_Zfull";   // "fit": for Z with rows to zero where missing data are present; "_Zfull": leave Z untouched
//     bool zero_rows_Z_flag = true; 
//     if(fit_type == "fit_Zfull"){
//         zero_rows_Z_flag = false; 
//     }

//     const unsigned int n_sim = 20; 
//     const unsigned int sim_start = 1; 
//     const bool debug = true; 
//     const bool normalized_loss_flag = true;  // true to normalize the loss
    
//     // define domain
//     const double t0 = 0.0;
//     const double tf = 1.0;
//     const unsigned int M = 11;   // number of time mesh nodes    
//     Triangulation<1, 1> time_mesh(t0, tf, M-1);
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_reduced_censoring_476");     
//     // std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

//     // define regularizing PDE in space
//     auto Ld = -laplacian<FEM>();   
//     PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    
//     // define regularizing PDE in time
//     auto Lt = -bilaplacian<SPLINE>();
//     PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);

//     // Read covariates
//     DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
//     std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
//     std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

//     DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
//     std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
//     std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 

//     // Read locations
//     DMatrix<double> time_locs = read_csv<double>(R_path + "/time_locs.csv");  
//     std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
//     std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

//     DMatrix<double> space_locs = read_csv<double>(R_path + "/space_locs.csv");  
//     std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
//     std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

//     double lambda_space; double lambda_time; 
    
//     for(auto sim = sim_start; sim <= n_sim; ++sim){

//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

//         // load data from .csv files
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//         std::cout << "dim y = " << y.rows() << "," << y.cols() << std::endl;

//         // check number of missing values
//         int count_na = 0;
//         double max_y = -1000.; // for debug
//         for (int i = 0; i < y.rows(); ++i) {
//             for (int j = 0; j < y.cols(); ++j) {
//                 if(std::isnan(y(i,j))) {
//                     ++count_na;
//                 } else{
//                     if(y(i,j) > max_y)
//                         max_y = y(i,j);
//                 }
//             }
//         }
//         std::cout << "num missing values = " << count_na << std::endl;
//         std::cout << "max y = " << max_y << std::endl;

//         BlockFrame<double, int> df;
//         df.stack(OBSERVATIONS_BLK, y);           // ATT: stack for space-time data!
//         df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);   // ATT: insert for space-time covariates!

                    
//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/" + fit_type;

//         MSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise);    
//         model.set_spatial_locations(space_locs);
//         model.set_temporal_locations(time_locs);
//         model.set_normalize_loss(normalized_loss_flag);
//         model.set_miss_rows_Z_to_zero(zero_rows_Z_flag);

//         // set model 
//         model.set_data(df);
//         model.set_ids_groups(ids_groups); 

//         // // read lambdas
//         double lambda_D;  
//         double lambda_T;  

//         std::ifstream fileLambdaS_gcv(solution_path + "/lambda_s_opt.csv");
//         if(fileLambdaS_gcv.is_open()){
//             fileLambdaS_gcv >> lambda_D; 
//             fileLambdaS_gcv.close();
//         }
//         std::ifstream fileLambdaT(solution_path + "/lambda_t_opt.csv");
//         if(fileLambdaT.is_open()){
//             fileLambdaT >> lambda_T; 
//             fileLambdaT.close();
//         }

//         // std::cout << "ATT: forcing lambdas..." << std::endl; 
//         // lambda_D = 1e-5; 
//         // lambda_T = 1e-5; 
        
//         model.set_lambda_D(lambda_D);
//         model.set_lambda_T(lambda_T);

//         std::cout << "ATT: forcing max iter fpirls..." << std::endl; 
//         model.set_fpirls_max_iter(15); 
        

//         // solve smoothing problem
//         //std::cout << "model init in test" << std::endl;
//         model.init();
//         //std::cout << "model solve in test" << std::endl;
//         model.solve();
//         //std::cout << "model end solve in test" << std::endl;

//         // Save solution
//         DMatrix<double> computedF = model.f();
//         // std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
//         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filef(solution_path + "/f.csv");
//         if(filef.is_open()){
//             filef << computedF.format(CSVFormatf);
//             filef.close();
//         }

//         DMatrix<double> computedFn = model.Psi()*model.f();
//         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filefn(solution_path + "/fn.csv");
//         if(filefn.is_open()){
//             filefn << computedFn.format(CSVFormatfn);
//             filefn.close();
//         }

//         DMatrix<double> computedBeta = model.beta();
//         const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filebeta(solution_path + "/beta.csv");
//         if(filebeta.is_open()){
//             filebeta << computedBeta.format(CSVFormatbeta);
//             filebeta.close();
//         }        

//         std::vector<DVector<double>> temp_bhat = model.b_hat(); 
//         DMatrix<double> computed_b;
//         computed_b.resize(temp_bhat.size(), model.p());   // m x p
//         for(int i=0; i<temp_bhat.size(); ++i){
//             computed_b.row(i) = temp_bhat[i]; 
//         }
//         const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream fileb(solution_path + "/b_random.csv");
//         if(fileb.is_open()){
//             fileb << computed_b.format(CSVFormatb);
//             fileb.close();
//         }

//         double computedsigmahat = std::sqrt(model.sigma_sq_hat());
//         std::ofstream filesigmahat(solution_path + "/sigma_hat.csv");
//         if(filesigmahat.is_open()){
//             filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
//             filesigmahat.close();
//         }

//         DMatrix<double> computedsigma_b_hat = model.Sigma_b();
//         const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filesigma_b_hat(solution_path + "/Sigma_b_hat.csv");
//         if(filesigma_b_hat.is_open()){
//             filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
//             filesigma_b_hat.close();
//         }

//         if(debug){

//             unsigned int computediter = model.n_inter_fpirls();
//             std::ofstream fileiter(solution_path + "/n_iter.csv");
//             if(fileiter.is_open()){
//                 fileiter << computediter << "\n";
//                 fileiter.close();
//             }

//             double computedminJ = model.min_J();
//             std::ofstream fileminJ(solution_path + "/minJ.csv");
//             if(fileminJ.is_open()){
//                 fileminJ << std::setprecision(16) << computedminJ << "\n";
//                 fileminJ.close();
//             }

//             DVector<DMatrix<double>> computedZ = model.Z_debug();
//             for(int j=0; j<computedZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZ_j(solution_path + "/Z_" + std::to_string(j+1) + ".csv");
//                 if(fileZ_j.is_open()){
//                     fileZ_j << computedZ(j).format(CSVFormatZ_j);
//                     fileZ_j.close();
//                 }
//             }

//             DVector<DMatrix<double>> computedZTZ = model.ZTZ();
//             for(int j=0; j<computedZTZ.size(); ++j){
//                 const static Eigen::IOFormat CSVFormatZTZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream fileZTZ_j(solution_path + "/ZTZ_" + std::to_string(j+1) + ".csv");
//                 if(fileZTZ_j.is_open()){
//                     fileZTZ_j << computedZTZ(j).format(CSVFormatZTZ_j);
//                     fileZTZ_j.close();
//                 }
//             }

//             DMatrix<double> computedW = model.pW_init();
//             const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileW(solution_path + "/W0.csv");
//             if(fileW.is_open()){
//                 fileW << computedW.format(CSVFormatW);
//                 fileW.close();
//             }

//             DMatrix<double> computedDelta0 = model.Delta0_debug();
//             const static Eigen::IOFormat CSVFormatDelta0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileDelta0(solution_path + "/Delta0.csv");
//             if(fileDelta0.is_open()){
//                 fileDelta0 << computedDelta0.format(CSVFormatDelta0);
//                 fileDelta0.close();
//             }

            
//             DMatrix<double> computePsi = model.Psi_debug();
//             const static Eigen::IOFormat CSVFormatPsi(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filePsi(solution_path + "/Psi.csv");
//             if(filePsi.is_open()){
//                 filePsi << computePsi.format(CSVFormatPsi);
//                 filePsi.close();
//             }

//             DMatrix<double> computeR0 = model.R0_debug();
//             const static Eigen::IOFormat CSVFormatR0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR0(solution_path + "/R0.csv");
//             if(fileR0.is_open()){
//                 fileR0 << computeR0.format(CSVFormatR0);
//                 fileR0.close();
//             }

//             DMatrix<double> computeR1 = model.R1_debug();
//             const static Eigen::IOFormat CSVFormatR1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR1(solution_path + "/R1.csv");
//             if(fileR1.is_open()){
//                 fileR1 << computeR1.format(CSVFormatR1);
//                 fileR1.close();
//             }

//             DMatrix<double> computePenT = model.PT_debug();
//             const static Eigen::IOFormat CSVFormatPenT(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filePenT(solution_path + "/penT.csv");
//             if(filePenT.is_open()){
//                 filePenT << computePenT.format(CSVFormatPenT);
//                 filePenT.close();
//             }


//             DMatrix<double> computeR0_space = model.R0_space_debug();
//             const static Eigen::IOFormat CSVFormatR0_space(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR0_space(solution_path + "/space_mass.csv");
//             if(fileR0_space.is_open()){
//                 fileR0_space << computeR0_space.format(CSVFormatR0_space);
//                 fileR0_space.close();
//             }


//             DMatrix<double> computeR1_space = model.R1_space_debug();
//             const static Eigen::IOFormat CSVFormatR1_space(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileR1_space(solution_path + "/space_stiff.csv");
//             if(fileR1_space.is_open()){
//                 fileR1_space << computeR1_space.format(CSVFormatR1_space);
//                 fileR1_space.close();
//             }


//             DMatrix<double> computeP0 = model.P0_debug();
//             const static Eigen::IOFormat CSVFormatP0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileP0(solution_path + "/P0.csv");
//             if(fileP0.is_open()){
//                 fileP0 << computeP0.format(CSVFormatP0);
//                 fileP0.close();
//             }

//             DMatrix<double> computeP1 = model.P1_debug();
//             const static Eigen::IOFormat CSVFormatP1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileP1(solution_path + "/P1.csv");
//             if(fileP1.is_open()){
//                 fileP1 << computeP1.format(CSVFormatP1);
//                 fileP1.close();
//             }
            


//         }


//     }


// }



// // test 5 (osservazioni ripetute) 
// //    domain:       c-shaped
// //    sampling:     locations = nodes (ma obs.rip -> sampling:pointwise)
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(msrpde_test5, laplacian_semiparametric_samplingatlocs_gridexact) {

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


//     std::vector<std::string> data_types = {"data", "data_rip_10", "data_rip_50"}; 

//     double lambda_gcv; double lambda_rmse;

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


//             std::cout << "------------------GCV run-----------------" << std::endl;


//             MSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise); 
            
//             // set model 
//             model_gcv.set_data(df);
//             model_gcv.set_ids_groups(ids_groups); 
//             model_gcv.set_spatial_locations(locs);
//             model_gcv.set_fpirls_max_iter(max_iter_fpirls);
//             model_gcv.set_normalize_loss(normalized_loss_flag);


//             // read lambda 
//             std::ifstream fileLambda(solutions_path_gcv + "/lambdas_opt" + norm_loss_str + ".csv");
//             if(fileLambda.is_open()){
//                 fileLambda >> lambda_gcv; 
//                 std::cout << "lambda=" << lambda_gcv << std::endl; 
//                 fileLambda.close();
//             }
            
//             model_gcv.set_lambda_D(lambda_gcv);

//             // solve smoothing problem
//             model_gcv.init();
//             model_gcv.solve();


//             // Save solution
//             DMatrix<double> computedF = model_gcv.f();
//             const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filef(solutions_path_gcv + "/f" + norm_loss_str + ".csv");
//             if(filef.is_open()){
//                 filef << computedF.format(CSVFormatf);
//                 filef.close();
//             }

//             DMatrix<double> computedFn = model_gcv.Psi()*model_gcv.f();
//             const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filefn(solutions_path_gcv + "/fn" + norm_loss_str + ".csv");
//             if(filefn.is_open()){
//                 filefn << computedFn.format(CSVFormatfn);
//                 filefn.close();
//             }

//             if(has_covariates){
//                 DMatrix<double> computedBeta = model_gcv.beta();
//                 const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filebeta(solutions_path_gcv + "/beta" + norm_loss_str + ".csv");
//                 if(filebeta.is_open()){
//                     filebeta << computedBeta.format(CSVFormatbeta);
//                     filebeta.close();
//                 }
//             }


//             std::vector<DVector<double>> temp_bhat = model_gcv.b_hat(); 
//             DMatrix<double> computed_b;
//             computed_b.resize(temp_bhat.size(), model_gcv.p());   // m x p
//             for(int i=0; i<temp_bhat.size(); ++i){
//                 computed_b.row(i) = temp_bhat[i]; 
//             }
//             const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileb(solutions_path_gcv + "/b_random" + norm_loss_str + ".csv");
//             if(fileb.is_open()){
//                 fileb << computed_b.format(CSVFormatb);
//                 fileb.close();
//             }

//             double computedsigmahat = std::sqrt(model_gcv.sigma_sq_hat());
//             std::ofstream filesigmahat(solutions_path_gcv + "/sigma_hat" + norm_loss_str + ".csv");
//             if(filesigmahat.is_open()){
//                 filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
//                 filesigmahat.close();
//             }

//             DMatrix<double> computedsigma_b_hat = model_gcv.Sigma_b();
//             const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filesigma_b_hat(solutions_path_gcv + "/Sigma_b_hat" + norm_loss_str + ".csv");
//             if(filesigma_b_hat.is_open()){
//                 filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
//                 filesigma_b_hat.close();
//             }            




//             std::cout << "------------------RMSE run-----------------" << std::endl;

  
//             MSRPDE<SpaceOnly> model_rmse(problem, Sampling::pointwise); 
            
//             // set model 
//             model_rmse.set_data(df);
//             model_rmse.set_ids_groups(ids_groups); 
//             model_rmse.set_spatial_locations(locs);
//             model_rmse.set_fpirls_max_iter(max_iter_fpirls);
//             model_rmse.set_normalize_loss(normalized_loss_flag);


//             // read lambda 
//             std::ifstream fileLambda_rmse(solution_path_rmse + "/lambdas_opt" + norm_loss_str + ".csv");
//             if(fileLambda_rmse.is_open()){
//                 fileLambda_rmse >> lambda_rmse; 
//                 std::cout << "lambda=" << lambda_rmse << std::endl; 
//                 fileLambda_rmse.close();
//             }
            
//             model_rmse.set_lambda_D(lambda_rmse);

//             // solve smoothing problem
//             model_rmse.init();
//             model_rmse.solve();


//             // Save solution
//             DMatrix<double> computedF_rmse = model_rmse.f();
//             const static Eigen::IOFormat CSVFormatf_rmse(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filef_rmse(solution_path_rmse + "/f" + norm_loss_str + ".csv");
//             if(filef_rmse.is_open()){
//                 filef_rmse << computedF_rmse.format(CSVFormatf_rmse);
//                 filef_rmse.close();
//             }

//             DMatrix<double> computedFn_rmse = model_rmse.Psi()*model_rmse.f();
//             const static Eigen::IOFormat CSVFormatfn_rmse(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filefn_rmse(solution_path_rmse + "/fn" + norm_loss_str + ".csv");
//             if(filefn_rmse.is_open()){
//                 filefn_rmse << computedFn_rmse.format(CSVFormatfn_rmse);
//                 filefn_rmse.close();
//             }

//             if(has_covariates){
//                 DMatrix<double> computedBeta_rmse = model_rmse.beta();
//                 const static Eigen::IOFormat CSVFormatbeta_rmse(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filebeta_rmse(solution_path_rmse + "/beta" + norm_loss_str + ".csv");
//                 if(filebeta_rmse.is_open()){
//                     filebeta_rmse << computedBeta_rmse.format(CSVFormatbeta_rmse);
//                     filebeta_rmse.close();
//                 }
//             }


//             std::vector<DVector<double>> temp_bhat_rmse = model_rmse.b_hat(); 
//             DMatrix<double> computed_b_rmse;
//             computed_b_rmse.resize(temp_bhat_rmse.size(), model_rmse.p());   // m x p
//             for(int i=0; i<temp_bhat_rmse.size(); ++i){
//                 computed_b_rmse.row(i) = temp_bhat_rmse[i]; 
//             }
//             const static Eigen::IOFormat CSVFormatb_rmse(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream fileb_rmse(solution_path_rmse + "/b_random" + norm_loss_str + ".csv");
//             if(fileb_rmse.is_open()){
//                 fileb_rmse << computed_b_rmse.format(CSVFormatb_rmse);
//                 fileb_rmse.close();
//             }

//             double computedsigmahat_rmse = std::sqrt(model_rmse.sigma_sq_hat());
//             std::ofstream filesigmahat_rmse(solution_path_rmse + "/sigma_hat" + norm_loss_str + ".csv");
//             if(filesigmahat_rmse.is_open()){
//                 filesigmahat_rmse << std::setprecision(16) << computedsigmahat_rmse << "\n"; 
//                 filesigmahat_rmse.close();
//             }

//             DMatrix<double> computedsigma_b_hat_rmse = model_rmse.Sigma_b();
//             const static Eigen::IOFormat CSVFormatsigma_b_hat_rmse(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filesigma_b_hat_rmse(solution_path_rmse + "/Sigma_b_hat" + norm_loss_str + ".csv");
//             if(filesigma_b_hat_rmse.is_open()){
//                 filesigma_b_hat_rmse << std::setprecision(16) << computedsigma_b_hat_rmse.format(CSVFormatsigma_b_hat_rmse);
//                 filesigma_b_hat_rmse.close();
//             }            



        
//         }    

    


//     }
// }


// test 6 (space-time)
//    domain:       unit square
//    sampling:     locations != nodes
//    space penalization: anisotropic diffusion (constant in time) 
//    time penalization: separable 
//    covariates:   yes
//    BC:           no
//    order FE:     1
TEST(msrpde_test6, laplacian_semiparametric_samplingatnodes) {
    
    // path test  
    std::string test_number = "6";  
    std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/space-time/Test_" + test_number;

    const unsigned int n_sim = 1; 
    const unsigned int sim_start = 1; 
    
    // define domain
    const double t0 = 0.0;
    const double tf = 1.0;
    const unsigned int M = 11;   // number of time mesh nodes    
    Triangulation<1, 1> time_mesh(t0, tf, M-1);
    MeshLoader<Triangulation<2, 2>> domain("unit_square_reduced_censoring_476");    
    // std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

    const unsigned int max_fpirls_iter = 15; 

    // rhs 
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

    // define regularizing PDE in space
    
    // TODO: leggere anisotropia !! 


    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);


    // Read covariates
    DMatrix<double> X = read_csv<double>(R_path + "/X.csv");

    DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
    std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
    std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

    DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
    std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
    std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 

    // Read locations
    DMatrix<double> time_locs = read_csv<double>(R_path + "/time_locs.csv");  
    std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
    std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

    DMatrix<double> space_locs = read_csv<double>(R_path + "/space_locs.csv");  
    std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
    std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

    double lambda_space; double lambda_time; 
    
    for(auto sim = sim_start; sim <= n_sim; ++sim){

        std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

        // load data from .csv files
        DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
        std::cout << "dim y = " << y.rows() << "," << y.cols() << std::endl;

        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);           // ATT: stack for space-time data!
        df.insert(DESIGN_MATRIX_BLK, X);         // ATT: insert for space-time covariates!
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);   // ATT: insert for space-time covariates!

                    
        std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit";

        MSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise);    
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);

        // set model 
        model.set_data(df);
        model.set_ids_groups(ids_groups); 

        // // read lambdas
        double lambda_D;  
        double lambda_T;  

        std::ifstream fileLambdaS_gcv(solution_path + "/lambda_s_opt.csv");
        if(fileLambdaS_gcv.is_open()){
            fileLambdaS_gcv >> lambda_D; 
            fileLambdaS_gcv.close();
        }
        std::ifstream fileLambdaT(solution_path + "/lambda_t_opt.csv");
        if(fileLambdaT.is_open()){
            fileLambdaT >> lambda_T; 
            fileLambdaT.close();
        }

        // std::cout << "ATT: forcing lambdas..." << std::endl; 
        // lambda_D = 1e-5; 
        // lambda_T = 1e-5; 
        
        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);

        model.set_fpirls_max_iter(max_fpirls_iter); 
        

        // solve smoothing problem
        model.init();
        model.solve();

        // Save solution
        DMatrix<double> computedF = model.f();
        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filef(solution_path + "/f.csv");
        if(filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
        }

        DMatrix<double> computedFn = model.Psi()*model.f();
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(solution_path + "/fn.csv");
        if(filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }

        DMatrix<double> computedBeta = model.beta();
        const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filebeta(solution_path + "/beta.csv");
        if(filebeta.is_open()){
            filebeta << computedBeta.format(CSVFormatbeta);
            filebeta.close();
        }        

        std::vector<DVector<double>> temp_bhat = model.b_hat(); 
        DMatrix<double> computed_b;
        computed_b.resize(temp_bhat.size(), model.p());   // m x p
        for(int i=0; i<temp_bhat.size(); ++i){
            computed_b.row(i) = temp_bhat[i]; 
        }
        const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream fileb(solution_path + "/b_random.csv");
        if(fileb.is_open()){
            fileb << computed_b.format(CSVFormatb);
            fileb.close();
        }

        double computedsigmahat = std::sqrt(model.sigma_sq_hat());
        std::ofstream filesigmahat(solution_path + "/sigma_hat.csv");
        if(filesigmahat.is_open()){
            filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
            filesigmahat.close();
        }

        DMatrix<double> computedsigma_b_hat = model.Sigma_b();
        const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filesigma_b_hat(solution_path + "/Sigma_b_hat.csv");
        if(filesigma_b_hat.is_open()){
            filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
            filesigma_b_hat.close();
        }

    }


}