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
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/Test_" + test_number;

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
//     DMatrix<double> locs; 
//     if(test_number == "1-tris"){
//         locs = read_csv<double>(R_path + "/locs.csv");
//         std::cout << "dim locs = " << locs.rows() << ";" << locs.cols() << std::endl;
//         std::cout << "max(locs) = " << locs.maxCoeff() << std::endl;
//     }

//     std::vector<double> lambdas; 
//     if(test_number == "1" || test_number == "1-tris"){
//         for(double x = -4.0; x <= +2.0; x += 2./3) lambdas.push_back(std::pow(10, x));
//     }
//     if(test_number == "1-bis"){
//         for(double x = -8.0; x <= +2.0; x += 0.5) lambdas.push_back(std::pow(10, x));
//     }
    
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
//         if(test_number == "1-tris"){
//             sampling_int = Sampling::pointwise; 
//         } else{
//             sampling_int = Sampling::mesh_nodes; 
//         }
//         MSRPDE<SpaceOnly> model_gcv(problem, sampling_int); 
        
//         // set model 
//         model_gcv.set_data(df);
//         model_gcv.set_ids_groups(ids_groups); 
//         if(test_number == "1-tris"){
//             model_gcv.set_spatial_locations(locs);
//         }

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




// test 2 
//    domain:       c-shaped
//    sampling:     locations = nodes
//    penalization: laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
TEST(gcv_msrpde_test2, laplacian_semiparametric_samplingatnodes_gridexact) {

    // path test  
    std::string test_number = "2";   // ATT controlla calcolo sigma in msrpde.h !!
    std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/Test_" + test_number;

    const unsigned int n_sim = 50; 
    const unsigned int sim_start = 1; 
    
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  


    // rhs 
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

    // define regularizing PDE  
    auto L = -laplacian<FEM>();   
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // Read 
    DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
    DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
    DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");

    std::vector<double> lambdas; 
    for(double x = -3.0; x <= +2.0; x += 0.55555555) lambdas.push_back(std::pow(10, x));

    
    DMatrix<double> lambdas_mat;
    lambdas_mat.resize(lambdas.size(), 1); 
    for(auto i = 0; i < lambdas_mat.rows(); ++i){
        // std::cout << "inserting lambdas[i] = " << std::setprecision(16) << lambdas[i] << std::endl;
        lambdas_mat(i,0) = lambdas[i]; 
    }
    double best_lambda; 


    // Simulations  
    for(auto sim = sim_start; sim <= n_sim; ++sim){
        std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

        // load data from .csv files
        DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");

        BlockFrame<double, int> df;
        df.insert(OBSERVATIONS_BLK, y);
        df.insert(DESIGN_MATRIX_BLK, X);
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);

        std::string solutions_path_gcv = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 


        std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 

        MSRPDE<SpaceOnly> model_gcv(problem, Sampling::mesh_nodes); 
        
        // set model 
        model_gcv.set_data(df);
        model_gcv.set_ids_groups(ids_groups); 

        // define GCV function and grid of \lambda_D values
        auto GCV = model_gcv.gcv<ExactEDF>();
        // optimize GCV
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas_mat);
        
        best_lambda = opt.optimum()(0,0);

        std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

        // Save lambda sequence 
        std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq.csv");
        for(std::size_t i = 0; i < lambdas.size(); ++i) 
            fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
        fileLambdaS.close();

        // Save lambda GCVopt for all alphas
        std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt.csv");
        if(fileLambdaoptS.is_open()){
            fileLambdaoptS << std::setprecision(16) << best_lambda;
            fileLambdaoptS.close();
        }

        // Save GCV 
        std::ofstream fileGCV_scores(solutions_path_gcv + "/score.csv");
        for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
            fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
        fileGCV_scores.close();


        std::ofstream fileGCV_edf(solutions_path_gcv + "/edf.csv");
        for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
            fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
        fileGCV_edf.close();



    }
}