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
    std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MSRPDE/Tests/Test_1";

    const unsigned int n_sim = 1; 
    const unsigned int sim_start = 1; 
    
    // define domain
    std::cout << "read mesh" << std::endl;
    MeshLoader<Triangulation<2, 2>> domain("c_shaped_242");  
    std::cout << "num domain.mesh.n_cells() = " << domain.mesh.n_cells() << std::endl; 

    // rhs 
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

    // define regularizing PDE  
    auto L = -laplacian<FEM>();   
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // Read covariates
    std::cout << "read X" << std::endl;
    DMatrix<double> X = read_csv<double>(R_path + "/X.csv");
    std::cout << "dim X = " << X.rows() << ";" << X.cols() << std::endl;
    std::cout << "max(X) = " << X.maxCoeff() << std::endl;   
    std::cout << "read Z" << std::endl;
    DMatrix<double> Z = read_csv<double>(R_path + "/Z.csv");  
    std::cout << "dim Z = " << Z.rows() << ";" << Z.cols() << std::endl;
    std::cout << "max(Z) = " << Z.maxCoeff() << std::endl; 

    double lambda = 1e-6; 
    
    for(auto sim = sim_start; sim <= n_sim; ++sim){

            std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl;

            // load data from .csv files
            std::cout << "read y" << std::endl;
            DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
            std::cout << "dim y = " << y.size() << std::endl;
            std::cout << "max(y) = " << y.maxCoeff() << std::endl; 

            BlockFrame<double, int> df;
            std::cout << "Setting y in df.." << std::endl; 
            df.insert(OBSERVATIONS_BLK, y);
            std::cout << "Setting X in df.." << std::endl;
            df.insert(DESIGN_MATRIX_BLK, X);
            std::cout << "Setting Z in df.." << std::endl;
            df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);
            std::cout << "End Setting Z in df!" << std::endl;
                        
            unsigned int idx = 0; 
            std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim); 

            std::cout << "define model" << std::endl;
            MSRPDE<SpaceOnly> model(problem, Sampling::mesh_nodes); 
            
            // set model data
            std::cout << "set data" << std::endl;
            model.set_data(df);

            std::cout << "set lambda" << std::endl;
            model.set_lambda_D(lambda);

            // solve smoothing problem
            std::cout << "model init in test" << std::endl;
            model.init();
            std::cout << "model solve in test" << std::endl;
            model.solve();
            std::cout << "model end solve in test" << std::endl;

            // Save solution
            std::cout << "save f" << std::endl;
            DMatrix<double> computedF = model.f();
            const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filef(solution_path + "/f.csv");
            if(filef.is_open()){
                filef << computedF.format(CSVFormatf);
                filef.close();
            }

            std::cout << "save fn" << std::endl;
            DMatrix<double> computedFn = model.Psi()*model.f();
            const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filefn(solution_path + "/fn.csv");
            if(filefn.is_open()){
                filefn << computedFn.format(CSVFormatfn);
                filefn.close();
            }

            std::cout << "save beta" << std::endl;
            DMatrix<double> computedBeta = model.beta();
            const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filebeta(solution_path + "/beta.csv");
            if(filebeta.is_open()){
                filebeta << computedBeta.format(CSVFormatbeta);
                filebeta.close();
            }

            std::cout << "save b_random" << std::endl;
            std::vector<DVector<double>> temp_bhat = model.b_hat(); 
            DMatrix<double> computed_b;
            computed_b.resize(model.p(), temp_bhat.size()); 
            for(int i=0; i<temp_bhat.size(); ++i){
                computed_b.col(i) = temp_bhat[i]; 
            }
            const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream fileb(solution_path + "/b_random.csv");
            if(fileb.is_open()){
                fileb << computed_b.format(CSVFormatb);
                fileb.close();
            }

            idx++;
            

    }


}
