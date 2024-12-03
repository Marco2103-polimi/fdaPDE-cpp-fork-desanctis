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
using fdapde::core::fem_order;

using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "../../fdaPDE/models/regression/msrpde.h"
#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SpaceOnly;
using fdapde::models::SRPDE;
using fdapde::models::MSRPDE;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;



// test 1 
//    domain:       c-shaped
//    sampling:     locations = nodes
//    penalization: laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
TEST(msrpde_test1, laplacian_semiparametric_samplingatnodes) {
    
    // path test  
    std::string test_number = "1-bis"; 
    std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/Test_" + test_number;

    const unsigned int n_sim = 50; 
    const unsigned int sim_start = 1; 
    const bool debug = false; 
    
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  
    std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

    // rhs 
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

    // define regularizing PDE  
    auto L = -laplacian<FEM>();   
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // Read covariates
    DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
    std::cout << "dim X = " << X.rows() << ";" << X.cols() << std::endl;
    std::cout << "max(X) = " << X.maxCoeff() << std::endl;   

    DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
    std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
    std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

    DVector<unsigned int> ids_groups = read_csv<unsigned int>(R_path + "/ids_groups.csv");
    std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
    std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 

    DMatrix<double> locs; 
    if(test_number == "1-bis"){
        locs = read_csv<double>(R_path + "/locs.csv");
        std::cout << "dim locs = " << locs.rows() << ";" << locs.cols() << std::endl;
        std::cout << "max(locs) = " << locs.maxCoeff() << std::endl;
    }

    double lambda; 
    
    for(auto sim = sim_start; sim <= n_sim; ++sim){

        std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

        // load data from .csv files
        DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
        std::cout << "dim y = " << y.size() << std::endl;
        std::cout << "max(y) = " << y.maxCoeff() << std::endl; 

        BlockFrame<double, int> df;
        df.insert(OBSERVATIONS_BLK, y);
        df.insert(DESIGN_MATRIX_BLK, X);
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);

                    
        std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 
        std::cout << "Sampling::mesh_nodes = " << Sampling::mesh_nodes << std::endl;
        std::cout << "Sampling::pointwise = " << Sampling::pointwise << std::endl;

        enum Sampling sampling_int; 
        if(test_number == "1-bis"){
            sampling_int = Sampling::pointwise;   // pointwise
        } else{
            sampling_int = Sampling::mesh_nodes;   // mesh nodes
        }

        MSRPDE<SpaceOnly> model(problem, sampling_int); 

        // set model 
        model.set_data(df);
        model.set_ids_groups(ids_groups); 
        if(test_number == "1-bis"){
            model.set_spatial_locations(locs);
        }

        // read lambda 
        std::ifstream fileLambda(solution_path + "/lambdas_opt.csv");
        if(fileLambda.is_open()){
            fileLambda >> lambda; 
            std::cout << "lambda=" << lambda << std::endl; 
            fileLambda.close();
        }
        // std::cout << "ATT: forcing lambda..." << std::endl; 
        // lambda = 10.; 
        
        model.set_lambda_D(lambda);

        // std::cout << "ATT: forcing max iter fpirls..." << std::endl; 
        // model.set_fpirls_max_iter(2); 
        

        // solve smoothing problem
        //std::cout << "model init in test" << std::endl;
        model.init();
        //std::cout << "model solve in test" << std::endl;
        model.solve();
        //std::cout << "model end solve in test" << std::endl;

        // Save solution
        DMatrix<double> computedF = model.f();
        // std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
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
            filesigmahat << computedsigmahat << "\n"; 
            filesigmahat.close();
        }

        DMatrix<double> computedsigma_b_hat = model.Sigma_b();
        const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filesigma_b_hat(solution_path + "/Sigma_b_hat.csv");
        if(filesigma_b_hat.is_open()){
            filesigma_b_hat << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
            filesigma_b_hat.close();
        }

        if(debug){

            unsigned int computediter = model.n_inter_fpirls();
            std::ofstream fileiter(solution_path + "/n_iter.csv");
            if(fileiter.is_open()){
                fileiter << computediter << "\n";
                fileiter.close();
            }

            double computedminJ = model.min_J();
            std::ofstream fileminJ(solution_path + "/minJ.csv");
            if(fileminJ.is_open()){
                fileminJ << computedminJ << "\n";
                fileminJ.close();
            }

            DVector<DMatrix<double>> computedZ = model.Z_debug();
            for(int j=0; j<computedZ.size(); ++j){
                const static Eigen::IOFormat CSVFormatZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream fileZ_j(solution_path + "/Z_" + std::to_string(j+1) + ".csv");
                if(fileZ_j.is_open()){
                    fileZ_j << computedZ(j).format(CSVFormatZ_j);
                    fileZ_j.close();
                }
            }

            DVector<DMatrix<double>> computedZTZ = model.ZTZ();
            for(int j=0; j<computedZTZ.size(); ++j){
                const static Eigen::IOFormat CSVFormatZTZ_j(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream fileZTZ_j(solution_path + "/ZTZ_" + std::to_string(j+1) + ".csv");
                if(fileZTZ_j.is_open()){
                    fileZTZ_j << computedZTZ(j).format(CSVFormatZTZ_j);
                    fileZTZ_j.close();
                }
            }

            DMatrix<double> computedW = model.pW_init();
            const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream fileW(solution_path + "/W0.csv");
            if(fileW.is_open()){
                fileW << computedW.format(CSVFormatW);
                fileW.close();
            }

            DMatrix<double> computedDelta0 = model.Delta0_debug();
            const static Eigen::IOFormat CSVFormatDelta0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream fileDelta0(solution_path + "/Delta0.csv");
            if(fileDelta0.is_open()){
                fileDelta0 << computedDelta0.format(CSVFormatDelta0);
                fileDelta0.close();
            }

            
            DMatrix<double> computePsi = model.Psi_debug();
            const static Eigen::IOFormat CSVFormatPsi(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filePsi(solution_path + "/Psi.csv");
            if(filePsi.is_open()){
                filePsi << computePsi.format(CSVFormatPsi);
                filePsi.close();
            }

            DMatrix<double> computeR0 = model.R0_debug();
            const static Eigen::IOFormat CSVFormatR0(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream fileR0(solution_path + "/R0.csv");
            if(fileR0.is_open()){
                fileR0 << computeR0.format(CSVFormatR0);
                fileR0.close();
            }

            DMatrix<double> computeR1 = model.R1_debug();
            const static Eigen::IOFormat CSVFormatR1(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream fileR1(solution_path + "/R1.csv");
            if(fileR1.is_open()){
                fileR1 << computeR1.format(CSVFormatR1);
                fileR1.close();
            }
        }


    }


}



// // test srpde x confronto melchionda
// //    domain:       c-shaped
// //    sampling:     locations = nodes
// //    penalization: laplacian
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// TEST(msrpde_test_srpde, laplacian_semiparametric_samplingatnodes) {
    
//     // path test  
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/SRPDE/Tests/Test_1";

//     const unsigned int n_sim = 1; 
//     const unsigned int sim_start = 1; 
    
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE  
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // Read covariates
//     DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
//     std::cout << "dim X = " << X.rows() << ";" << X.cols() << std::endl;
//     std::cout << "max(X) = " << X.maxCoeff() << std::endl;   

//     double lambda; 
    
//     for(auto sim = sim_start; sim <= n_sim; ++sim){

//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

//         // load data from .csv files
//         std::cout << "read y" << std::endl;
//         DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//         std::cout << "dim y = " << y.size() << std::endl;
//         std::cout << "max(y) = " << y.maxCoeff() << std::endl; 

//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
  
//         std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/fit"; 

//         SRPDE model(problem, Sampling::mesh_nodes); 

//         // set model 
//         model.set_data(df);

//         // // read lambda 
//         // std::ifstream fileLambda(solution_path + "/lambdas_opt.csv");
//         // if(fileLambda.is_open()){
//         //     fileLambda >> lambda; 
//         //     std::cout << "lambda=" << lambda << std::endl; 
//         //     fileLambda.close();
//         // }
//         std::cout << "ATT: forcing lambda..." << std::endl; 
//         lambda = 10.; 
//         model.set_lambda_D(lambda);


//         // solve smoothing problem
//         //std::cout << "model init in test" << std::endl;
//         model.init();
//         //std::cout << "model solve in test" << std::endl;
//         model.solve();
//         //std::cout << "model end solve in test" << std::endl;

//         // Save solution
//         DMatrix<double> computedF = model.f();
//         std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
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


//     }


// }
