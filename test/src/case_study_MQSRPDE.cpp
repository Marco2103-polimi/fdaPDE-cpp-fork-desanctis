#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::laplacian;
using fdapde::core::bilaplacian;
using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::Grid; 
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
using fdapde::core::PDE;
using fdapde::core::Triangulation;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/sampling_design.h"
#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/qsrpde.h"
#include "../../fdaPDE/models/regression/mqsrpde.h"
using fdapde::models::SRPDE;
using fdapde::models::QSRPDE;
using fdapde::models::MQSRPDE;

#include "../../fdaPDE/models/regression/gcv.h"
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;



// gcv 
TEST(case_study_mqsrpde_gcv, NO2_restricted) {

    const std::string month = "gennaio";       // gennaio dicembre 
    const std::string day_chosen = "11"; 

    const std::string eps_string = "1e-1.5";   // "1e-0.25" "0"  "1e+0" "1e+0.5" "1e+1" "1e+2"

    std::string pde_type = "tr";  // ""  "tr" 
    const std::string u_string = "1e-1"; 

    const bool return_smoothing = false;    // if true, metti exact gcv!! 
    std::string gcv_type = "stochastic";   // "exact" "stochastic"  ---> MODIFICA ANCHE GIU'!

    const unsigned int num_fpirls_iter = 200;   // per ora GCV lancia sempre stesso numero per ogni alpha
    const std::string fpirls_iter_strategy = "1"; 

    std::size_t seed = 438172;
    unsigned int MC_run = 100; 
    const std::string model_type = "param";  // "nonparam" "param"
    const std::string  cascading_model_type = "parametric";  // "parametric" "nonparametric" 
    if(cascading_model_type == "parametric"){
        pde_type = pde_type + "_par"; 
    }

    const std::string cov_strategy = "7"; 
    const std::string covariate_type = "cov_" + cov_strategy; 
    std::string covariate_type_for_data; 
    if(cov_strategy == "1" || cov_strategy == "2"){
        covariate_type_for_data = "dens.new_log.elev.original"; 
    }
    if(cov_strategy == "3"){
        covariate_type_for_data = "log.dens_log.elev.original"; 
    }
    if(cov_strategy == "4"){
        covariate_type_for_data = "log.dens.ch_log.elev.original"; 
    }
    if(cov_strategy == "5"){
        covariate_type_for_data = "N.dens.new_N.elev.original"; 
    }
    if(cov_strategy == "6"){
        covariate_type_for_data = "log.dens_N.elev.original"; 
    }
    if(cov_strategy == "7"){
        covariate_type_for_data = "sqrt.dens_log.elev.original"; 
    }
    if(cov_strategy == "8"){
        covariate_type_for_data = "sqrt.dens_sqrt.elev.original"; 
    }

    const std::string num_months  = "one_month"; 
    const std::string mesh_type = "canotto_fine";  // "square" "esagoni" "convex_hull" "CH_fine" "canotto" "canotto_fine"
    const std::string mesh_type_param_casc = mesh_type; 

    std::string est_type = "quantile";    // mean quantile
    std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 
                                  0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 
                                  0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99
                                  };


    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
    std::string path_data = path + "/data/MQSRPDE/NO2/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

    if(est_type == "mean"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    if(est_type == "quantile"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string;
        } else{
            solutions_path = path + "/results/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type + "/eps_" + eps_string;
        }

        solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    std::cout << "solution path: " << solutions_path << std::endl; 
    
    // lambdas sequence 
    std::vector<double> lambdas; 
    DMatrix<double> lambdas_mat;

    // lambdas sequence for fine grid of quantiles 
    std::vector<double> lambdas_1_5;
    std::vector<double> lambdas_10_25;
    std::vector<double> lambdas_30_70;
    std::vector<double> lambdas_75_90;
    std::vector<double> lambdas_95_99;

    if(est_type == "mean"){
        if(!return_smoothing){
            for(double xs = -6.0; xs <= -0.5; xs += 0.05)
                lambdas.push_back(std::pow(10,xs));   

        } else{
            double lambda_S;  
            std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
            if(fileLambdaS_opt.is_open()){
                fileLambdaS_opt >> lambda_S; 
                fileLambdaS_opt.close();
            }
            lambdas.push_back(lambda_S); 
        }

        // define lambda sequence as matrix 
        lambdas_mat.resize(lambdas.size(), 1); 
        for(auto i = 0; i < lambdas_mat.rows(); ++i){
            lambdas_mat(i,0) = lambdas[i]; 
        }
        std::cout << "dim lambdas mat" << lambdas_mat.rows() << " " << lambdas_mat.cols() << std::endl;
    }


    if(return_smoothing && lambdas.size() > 1){
        std::cout << "ERROR: you want S, but you are providing more lambdas" << std::endl; 
    } 

    if(est_type == "quantile"){
        for(double x = -6.8; x <= 0.0; x += 0.1) lambdas_1_5.push_back(std::pow(10, x)); 
        for(double x = -6.0; x <= 0.0; x += 0.1) lambdas_10_25.push_back(std::pow(10, x));
        for(double x = -5.5; x <= 1.0; x += 0.1) lambdas_30_70.push_back(std::pow(10, x)); 
        for(double x = -6.5; x <= -1.0; x += 0.1) lambdas_75_90.push_back(std::pow(10, x)); 
        for(double x = -6.5; x <= -1.5; x += 0.1) lambdas_95_99.push_back(std::pow(10, x));
    }

    // define spatial domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X;  

    y = read_csv<double>(path_data + "/y_rescale" + ".csv"); 
    space_locs = read_csv<double>(path_data + "/locs" + ".csv");       
    if(model_type == "param"){
        X = read_csv<double>(path_data + "/X_" + covariate_type_for_data + ".csv");
    }
      
    // check dimensions
    std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
    std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;


    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(model_type == "param")
        df.insert(DESIGN_MATRIX_BLK, X);
   
    // ATT: parameter cascading legge sempre il fit nonparametrico 
    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 


    DMatrix<double, Eigen::RowMajor> b_data; 
    DMatrix<double> u; 

    b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + "_" + covariate_type_for_data + ".csv");
    u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + "_" + covariate_type_for_data + ".csv");    
      
    std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
    DiscretizedVectorField<2, 2> b(b_data);
    
    //DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    
    std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl ; 
    

    //auto L = -laplacian<FEM>();
    auto L = -laplacian<FEM>() + advection<FEM>(b);
    
    
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);


    // // Save quadrature nodes 
    // DMatrix<double> quad_nodes = problem.quadrature_nodes();  
    // std::cout << "rows quad nodes = " << quad_nodes.rows() << std::endl; 
    // std::cout << "cols quad nodes = " << quad_nodes.cols() << std::endl; 
    // std::ofstream file_quad(path_data + "/" + mesh_type + "/quadrature_nodes.csv");
    // for(int i = 0; i < quad_nodes.rows(); ++i) {
    //     file_quad << quad_nodes(i, 0) << "," << quad_nodes(i, 1) << "\n";
    // }
    // file_quad.close();

    std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 

    if(est_type == "mean"){

        SRPDE model(problem, Sampling::pointwise);
        
        // set model's data
        model.set_spatial_locations(space_locs);
        
        model.set_data(df);
        model.init();

        // define GCV function and grid of \lambda_D values

        // stochastic
        auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
        if(return_smoothing){
            std::cout << "ATTENTION: YOU WANT S, BUT STOCHASTIC GCV IS ACTIVATED"; 
        }

        // // exact
        // auto GCV = model.gcv<ExactEDF>();
        // if(!return_smoothing){
        //     std::cout << "ATTENTION: YOU WANT TO RUN GCV, BUT EXACT GCV IS ACTIVATED"; 
        // }

           
        // optimize GCV
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas_mat);
        SVector<1> best_lambda = opt.optimum();

        if(!return_smoothing){
            // Save lambda sequence 
            std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq.csv");
            for(std::size_t i = 0; i < lambdas.size(); ++i) 
                fileLambda_S_Seq << std::setprecision(16) << lambdas[i] << "\n"; 
            fileLambda_S_Seq.close();

            // Save Lambda opt
            std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt.csv");
            if(fileLambdaoptS.is_open()){
                fileLambdaoptS << std::setprecision(16) << best_lambda[0];
                fileLambdaoptS.close();
            }
            // Save GCV scores
            std::ofstream fileGCV_scores(solutions_path + "/gcv_scores.csv");
            for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
            fileGCV_scores.close();

            // Save edfs
            std::ofstream fileEDF(solutions_path + "/edfs.csv");
            for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
            fileEDF.close();

        }
        
        if(return_smoothing){
            // Save S
            DMatrix<double> computedS = GCV.S_get_gcv();
            Eigen::saveMarket(computedS, solutions_path + "/S.mtx");
        }

    }

    if(est_type == "quantile"){

        std::string solutions_path_single = solutions_path + "/fp_" + fpirls_iter_strategy; 
        
        for(auto alpha : alphas){  

            unsigned int alpha_int = alpha*100; 
            std::string alpha_string = std::to_string(alpha_int); 

            std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 
                    
                QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
                model_gcv.set_spatial_locations(space_locs);
                model_gcv.set_fpirls_max_iter(num_fpirls_iter); 

                std::vector<double> lambdas;
                
                if(alpha < 0.06){
                    lambdas = lambdas_1_5; 
                }
                if((0.09 < alpha) && (alpha < 0.26)){
                    lambdas = lambdas_10_25; 
                }
                if((0.29 < alpha) && (alpha < 0.71)){
                    lambdas = lambdas_30_70; 
                }
                if((0.74 < alpha) && (alpha < 0.91)){
                    lambdas = lambdas_75_90; 
                }
                if(alpha > 0.94){
                    lambdas = lambdas_95_99; 
                }
                // refactor lambda as a matrix 
                lambdas_mat.resize(lambdas.size(), 1); 
                for(auto i = 0; i < lambdas_mat.rows(); ++i){
                    lambdas_mat(i,0) = lambdas[i]; 
                }
                std::cout << "dim lambdas mat" << lambdas_mat.rows() << " " << lambdas_mat.cols() << std::endl;

                // set model's data
                if(eps_string == "1e-0.5"){
                    model_gcv.set_eps_power(-0.5); 
                }
                if(eps_string == "1e-1"){
                    model_gcv.set_eps_power(-1.0); 
                }
                if(eps_string == "1e-1.5"){
                    model_gcv.set_eps_power(-1.5); 
                }
                if(eps_string == "1e-2"){
                    model_gcv.set_eps_power(-2.0); 
                }
                if(eps_string == "1e-3"){
                    model_gcv.set_eps_power(-3.0); 
                }
                
                model_gcv.set_data(df);
                model_gcv.init();


                // define GCV function and grid of \lambda_D values
                // stochastic
                auto GCV = model_gcv.gcv<StochasticEDF>(MC_run, seed);
                // optimize GCV
                Grid<fdapde::Dynamic> opt;
                opt.optimize(GCV, lambdas_mat);
                
                double best_lambda = opt.optimum()(0,0);
        
                std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

                // Save lambda sequence 
                std::ofstream fileLambdaS(solutions_path_single + "/lambdas_seq_alpha_" + alpha_string + ".csv");
                for(std::size_t i = 0; i < lambdas.size(); ++i) 
                    fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
                fileLambdaS.close();

                // Save lambda GCVopt for all alphas
                std::ofstream fileLambdaoptS(solutions_path_single + "/lambdas_opt_alpha_" + alpha_string + ".csv");
                if(fileLambdaoptS.is_open()){
                    fileLambdaoptS << std::setprecision(16) << best_lambda;
                    fileLambdaoptS.close();
                }

                // Save GCV 
                std::ofstream fileGCV_scores(solutions_path_single + "/score_alpha_" + alpha_string + ".csv");
                for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                    fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
                fileGCV_scores.close();


                // Save edfs
                std::ofstream fileEDF(solutions_path_single + "/edfs_alpha_" + alpha_string + ".csv");
                for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                    fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
                fileEDF.close();

            }

        }


}


// run 
TEST(case_study_mqsrpde_run, NO2_restricted) {

    const std::string month = "gennaio";       // gennaio dicembre 
    const std::string day_chosen = "11"; 

    const std::string eps_string = "1e-1.5";   // "0" "1e+0" "1e+1"

    std::string pde_type = "tr";  // "" "tr" "tr2"
    const std::string u_string = "1e-2"; 

    std::string est_type = "quantile";    // mean quantile
    bool single_est = false;
    bool mult_est = true;

    bool save_fpirls_iter = false; 
    std::vector<unsigned int> num_fpirls_iter_per_alpha = {200, 200, 200, 200, 200, 
                                                           200, 200, 200, 200, 200, 200, 200, 200, 
                                                           200, 
                                                           200, 200, 200, 200, 200, 200, 200, 200, 200, 
                                                           200, 200, 200, 200}; 

   // std::vector<unsigned int> num_fpirls_iter_per_alpha = {10, 10}; 

    const std::string fpirls_iter_strategy = "1"; 

    double gamma0 = 5.;   // initial crossing penalty parameter 
    std::string gamma0_str = "5"; 

    const std::string model_type = "param";  // "nonparam" "param"
    const std::string  cascading_model_type = "nonparametric";  // ok parametric2 anche per tr3
    if(cascading_model_type == "parametric" || cascading_model_type == "parametric2"){
        pde_type = pde_type + "_par"; 
    }
    
    
    const std::string cov_strategy = "8"; 
    const std::string covariate_type = "cov_" + cov_strategy; 
    std::string covariate_type_for_data; 
    if(cov_strategy == "1" || cov_strategy == "2"){
        covariate_type_for_data = "dens.new_log.elev.original"; 
    }
    if(cov_strategy == "3"){
        covariate_type_for_data = "log.dens_log.elev.original"; 
    }
    if(cov_strategy == "4"){
        covariate_type_for_data = "log.dens.ch_log.elev.original"; 
    }
    if(cov_strategy == "5"){
        covariate_type_for_data = "N.dens.new_N.elev.original"; 
    }
    if(cov_strategy == "6"){
        covariate_type_for_data = "log.dens_N.elev.original"; 
    }
    if(cov_strategy == "7"){
        covariate_type_for_data = "sqrt.dens_log.elev.original"; 
    }
    if(cov_strategy == "8"){
        covariate_type_for_data = "sqrt.dens_sqrt.elev.original"; 
    }


    const std::string num_months  = "one_month";    
    
    const std::string mesh_type = "canotto_fine";  // "square" "esagoni" "convex_hull" "CH_fine" "canotto_fine"
    const std::string mesh_type_param_casc = mesh_type;

    std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 
                                  0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 
                                  0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99
                                  };

    // std::vector<double> alphas = {0.5, 0.55};
    bool debug = false; 

                                
    const unsigned int max_it_convergence_loop = 10;       // numero massimo iterazioni MM 

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
    std::string path_data = path + "/data/MQSRPDE/NO2/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

     if(est_type == "mean"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    if(est_type == "quantile"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string;
        } else{
            solutions_path = path + "/results/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type + "/eps_" + eps_string;
        }

        solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    std::cout << "path: " << solutions_path << std::endl; 


    // define spatial domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X; 


    y = read_csv<double>(path_data + "/y_rescale" + ".csv"); 
    space_locs = read_csv<double>(path_data + "/locs" + ".csv");
    if(model_type == "param"){
        X = read_csv<double>(path_data + "/X_" + covariate_type_for_data + ".csv");
    }
      
    
    // check dimensions
    std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
    std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;
    std::cout << "sum X " << X.sum() << std::endl;

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(model_type == "param")
        df.insert(DESIGN_MATRIX_BLK, X);
   

    // ATT: parameter cascading è SEMPRE quelli su modelli nonparametrici, anche quando ci sono covariate, 
    //      tanto già c'è l'approssimazione che vale per la media e non per i quantili
    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
    

    DMatrix<double, Eigen::RowMajor> b_data; 
    DMatrix<double> u; 

    b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + "_" + covariate_type_for_data + ".csv");
    u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + "_" + covariate_type_for_data + ".csv");    
  
    
    // DMatrix<double> u = DMatrix<double>::Ones(domain.mesh.n_cells() * 3, 1); // *0.001;
    std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl; 
    //DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_value + ".csv");
    std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl;


    DiscretizedVectorField<2, 2> b(b_data);
    // auto L = -intensity_value*laplacian<FEM>() + advection<FEM>(b);

    auto L = -laplacian<FEM>() + advection<FEM>(b);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);


    std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 
    if(est_type == "mean"){

        SRPDE model(problem, Sampling::pointwise);
    
        // set model's data
        model.set_spatial_locations(space_locs);

        // Read optima lambdas 
        double lambda_S; 
        std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
        if(fileLambdaS_opt.is_open()){
            fileLambdaS_opt >> lambda_S; 
            fileLambdaS_opt.close();
        }

        std::cout << "lambda S" << lambda_S << std::endl;

        model.set_lambda_D(lambda_S);
        
        model.set_data(df);

        model.init();
        model.solve();

        // Save C++ solution 
        DMatrix<double> computedF = model.f();
        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filef(solutions_path + "/f.csv");
        if (filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
        }

        DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(solutions_path + "/fn.csv");
        if (filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }

        if(model_type == "param"){
            DMatrix<double> computedBeta = model.beta();
            const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filebeta(solutions_path + "/beta.csv");
            if (filebeta.is_open()){
                filebeta << computedBeta.format(CSVFormatBeta);
                filebeta.close();
            }
        }

    }

    if(est_type == "quantile"){

        if(single_est){
            std::cout << "-----------------------SINGLE running---------------" << std::endl;

            std::size_t idx = 0;

            std::string solutions_path_single = solutions_path + "/fp_" + fpirls_iter_strategy; 

            for(double alpha : alphas){
                unsigned int alpha_int = alphas[idx]*100;  

                QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
                model.set_spatial_locations(space_locs);
                model.set_fpirls_max_iter(num_fpirls_iter_per_alpha[idx]); 
               
                double lambda; 
                std::ifstream fileLambda(solutions_path_single + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
                if(fileLambda.is_open()){
                    fileLambda >> lambda; 
                    fileLambda.close();
                }
                model.set_lambda_D(lambda);

                // set model data
                model.set_data(df);

                // solve smoothing problem
                model.init();
                model.solve();

                // Save solution
                DMatrix<double> computedF = model.f();
                const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filef(solutions_path_single + "/f_" + std::to_string(alpha_int) + ".csv");
                if(filef.is_open()){
                    filef << computedF.format(CSVFormatf);
                    filef.close();
                }

                DMatrix<double> computedG = model.g();
                const static Eigen::IOFormat CSVFormatg(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream fileg(solutions_path_single + "/g_" + std::to_string(alpha_int) + ".csv");
                if(fileg.is_open()){
                    fileg << computedG.format(CSVFormatg);
                    fileg.close();
                }

                DMatrix<double> computedFn = model.Psi()*model.f();
                const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filefn(solutions_path_single + "/fn_" + std::to_string(alpha_int) + ".csv");
                if(filefn.is_open()){
                    filefn << computedFn.format(CSVFormatfn);
                    filefn.close();
                }

                if(save_fpirls_iter){
                    const unsigned int fpirls_iter = model.n_iter_qsrpde(); 
                    std::ofstream filefpirls(solutions_path_single  + "/fpirls_it_a" + std::to_string(alpha_int) + ".csv");
                    if(filefpirls.is_open()){
                        filefpirls << fpirls_iter;
                        filefpirls.close();
                    }     
                }

                if(debug){
                    std::ofstream fileLoss(solutions_path_single + "/loss_" + std::to_string(alpha_int) + ".csv");
                    if (fileLoss.is_open()) {
                        for (size_t i = 0; i < model.loss_each_iter().size(); ++i) {
                            fileLoss << model.loss_each_iter()[i] << "\n";  // Write the element
                        }
                        fileLoss.close();
                    } 

                    std::ofstream fileMisfit(solutions_path_single + "/mis_" + std::to_string(alpha_int) + ".csv");
                    if (fileMisfit.is_open()) {
                        for (size_t i = 0; i < model.pdemisfit_each_iter().size(); ++i) {
                            fileMisfit << model.pdemisfit_each_iter()[i] << "\n";   // Write the element
                        }
                        fileMisfit.close();
                    } 

   
                }
       

                if(model_type == "param"){
                    DMatrix<double> computedBeta = model.beta(); 
                    const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream filebeta(solutions_path_single + "/beta_" + std::to_string(alpha_int) + ".csv");
                    if(filebeta.is_open()){
                        filebeta << computedBeta.format(CSVFormatBeta);
                        filebeta.close();
                    }
                }


                // Save Psi, R0 and R1 per l'inferenza per un solo alpha 
                if(idx == 0){        
                    SpMatrix<double> Psi_mat = model.Psi(fdapde::models::not_nan());
                    Eigen::saveMarket(Psi_mat, solutions_path_single + "/Psi" + ".mtx");

                    SpMatrix<double> R0_mat = model.R0(); 
                    Eigen::saveMarket(R0_mat, solutions_path_single + "/R0" + ".mtx");

                    SpMatrix<double> R1_mat = model.R1(); 
                    Eigen::saveMarket(R1_mat, solutions_path_single + "/R1" + ".mtx");
 
                }
                

                idx++;
            }

        }
        
        if(mult_est){

            std::cout << "-----------------------MULTIPLE running---------------" << std::endl;

            std::string solutions_path_mult = solutions_path + "/fp_" + fpirls_iter_strategy + "/g_" + gamma0_str; 
            
            MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
            model.set_spatial_locations(space_locs);
            model.set_preprocess_option(false); 
            model.set_forcing_option(false);
            model.set_max_iter(max_it_convergence_loop); 
            model.set_gamma_init(gamma0); 

            // use optimal lambda to avoid possible numerical issues
            DMatrix<double> lambdas;
            DVector<double> lambdas_temp; 
            lambdas_temp.resize(alphas.size());
            for(std::size_t idx = 0; idx < alphas.size(); ++idx){
                unsigned int alpha_int = alphas[idx]*100;  
                std::ifstream fileLambdas(solutions_path + "/fp_" + fpirls_iter_strategy + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
                if(fileLambdas.is_open()){
                    fileLambdas >> lambdas_temp(idx); 
                    fileLambdas.close();
                }
            }
            lambdas = lambdas_temp;                
            model.setLambdas_D(lambdas);

            // set model data
            model.set_data(df);

            // solve smoothing problem
            model.init();
            model.solve();

            // Save solution
            DMatrix<double> computedF = model.f();
            const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filef(solutions_path_mult +  "/f_all.csv");
            if(filef.is_open()){
                filef << computedF.format(CSVFormatf);
                filef.close();
            }

            DMatrix<double> computedFn = model.Psi_mult()*model.f();
            const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filefn(solutions_path_mult + "/fn_all.csv");
            if(filefn.is_open()){
                filefn << computedFn.format(CSVFormatfn);
                filefn.close();
            }

            
            if(model_type == "param"){
                DMatrix<double> computedBeta = model.beta();
                const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filebeta(solutions_path_mult + "/beta_all.csv");
                if(filebeta.is_open()){
                    filebeta << computedBeta.format(CSVFormatbeta);
                    filebeta.close();
                }
            }
        }
        

    }        
}



//////////////////////////////// SRPDE per densità abitativa ///////////////////////////////////////////////////////////


// TEST(srpde_dens_pop, gcv_srpde_dens_pop_ARPA) {

//     const std::string mesh_type = "canotto_fine";  // "square" "esagoni" "convex_hull" "CH_fine"
//     const std::string log_str = "_log"; 


//     // Marco 
//     std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
//     std::string path_data = path + "/covariates_nodes/density/SRPDE/data";  
//     std::string solutions_path = path + "/covariates_nodes/density/SRPDE/" + mesh_type; 
  
//     std::cout << "solution path: " << solutions_path << std::endl; 
    
//     // lambdas sequence 
//     std::vector<double> lambdas; 
//     DMatrix<double> lambdas_mat;

//     for(double xs = -8.0; xs <= -1.0; xs += 0.1)
//         lambdas.push_back(std::pow(10,xs));   

//     // define lambda sequence as matrix 
//     lambdas_mat.resize(lambdas.size(), 1); 
//     for(auto i = 0; i < lambdas_mat.rows(); ++i){
//         lambdas_mat(i,0) = lambdas[i]; 
//     }


//     // define spatial domain and regularizing PDE
//     MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs;

//     y = read_csv<double>(path_data + "/y" + log_str + ".csv"); 
//     space_locs = read_csv<double>(path_data + "/locs" + ".csv");       
        
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
   
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     auto L = -laplacian<FEM>();
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 

//     SRPDE model_gcv(problem, Sampling::pointwise);
    
//     // set model's data
//     model_gcv.set_spatial_locations(space_locs);
    
//     model_gcv.set_data(df);
//     model_gcv.init();

//     // exact
//     auto GCV = model_gcv.gcv<ExactEDF>();
        
//     // optimize GCV
//     Grid<fdapde::Dynamic> opt;
//     opt.optimize(GCV, lambdas_mat);
//     SVector<1> best_lambda = opt.optimum();


//     // Save lambda sequence 
//     std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq" + log_str + ".csv");
//     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//         fileLambda_S_Seq << std::setprecision(16) << lambdas[i] << "\n"; 
//     fileLambda_S_Seq.close();

//     // Save Lambda opt
//     std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt" + log_str + ".csv");
//     if(fileLambdaoptS.is_open()){
//         fileLambdaoptS << std::setprecision(16) << best_lambda[0];
//         fileLambdaoptS.close();
//     }
//     // Save GCV scores
//     std::ofstream fileGCV_scores(solutions_path + "/gcv_scores" + log_str + ".csv");
//     for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//     fileGCV_scores.close();     


//     std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 

//     SRPDE model(problem, Sampling::pointwise);

//     // set model's data
//     model.set_spatial_locations(space_locs);

//     // Read optima lambdas 
//     double lambda_S; 
//     std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt" + log_str + ".csv");
//     if(fileLambdaS_opt.is_open()){
//         fileLambdaS_opt >> lambda_S; 
//         fileLambdaS_opt.close();
//     }

//     std::cout << "lambda S" << lambda_S << std::endl;

//     model.set_lambda_D(lambda_S);
    
//     model.set_data(df);

//     model.init();
//     model.solve();

//     // Save C++ solution 
//     DMatrix<double> computedF = model.f();
//     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filef(solutions_path + "/f" + log_str + ".csv");
//     if (filef.is_open()){
//         filef << computedF.format(CSVFormatf);
//         filef.close();
//     }

//     DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
//     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filefn(solutions_path + "/fn" + log_str + ".csv");
//     if (filefn.is_open()){
//         filefn << computedFn.format(CSVFormatfn);
//         filefn.close();
//     }


// }

















///////////////////////////////////// OSSERVAZIONI RIPETUTE /////////////////////////////////////////////////////////////////////////////////

// // gcv 
// TEST(case_study_mqsrpde_gcv_obs_rip, NO2) {

//     const std::string month = "gennaio";       // gennaio dicembre 

//     // several days or day fixed?
//     const std::string days_begin_end = "";   // "" "7_11"
//     const std::string day_begin = "7"; 
//     const std::string day_end = "11"; 

//     const std::string day_chosen = "11"; 
//     const std::string K = "24";    // maximum number of repeated observations per each location 

//     const std::string eps_string = "1e+0";   

//     const std::string pde_type = "b";  // ""  "b"     ---> ATT modifica giù
//     const std::string u_string = "1"; 
//     const std::string rescale = "_rescale";    // "" "_rescale"

//     const bool return_smoothing = false;    // if true, metti exact gcv!! 
//     std::string gcv_type = "exact";   // "exact" "stochastic"  ---> MODIFICA ANCHE GIU'!
//     // messo exact perchè stochastic_edf non ha ancora IV strategia 
//     std::size_t seed = 438172;
//     unsigned int MC_run = 100; 

//     const std::string mesh_type = "canotto_fine";   

//     const std::string model_type = "param";  // "nonparam" "param"
    
//     const std::string cov_strategy = "8"; 
//     const std::string covariate_type = "cov_" + cov_strategy; 
//     std::string covariate_type_for_data; 
//     if(cov_strategy == "8"){
//         covariate_type_for_data = "sqrt.dens_sqrt.elev.original"; 
//     }

//     std::string est_type = "quantile";    // mean quantile
//     std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 
//                                   0.30, 0.35, 0.40, 0.45,
//                                   0.5,
//                                   0.55, 0.60, 0.65, 0.70, 
//                                   0.75, 0.80, 0.85, 
//                                   0.90,
//                                   0.95, 0.96, 0.97, 0.98, 0.99
//                                   };

                                  
//     std::vector<std::string> gcv_summaries = {"_II_appr"};   // "_I_appr",  "_II_appr", "_III_appr", "_IV_appr


//     // Data path  
//     std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
//     std::string path_data; 
//     if(days_begin_end == ""){
//         path_data = path + "/data/MQSRPDE/NO2/" + month + "/day_" + day_chosen;  
//     } else{
//         path_data = path + "/data/MQSRPDE/NO2/" + month + "/days_" + day_begin + "_" + day_end;  
//     }
//     std::string solutions_path; 
  
//     // // Ilenia 
//     // std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
//     // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen; 
//     // std::string solutions_path;

//     // Solution path 
//     if(days_begin_end == ""){
//         if(est_type == "mean"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/SRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type;
//             } else{
//                 solutions_path = path + "/results/SRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
//             }

//             if(pde_type == "b") 
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//         if(est_type == "quantile"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/MQSRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string;
//             } else{
//                 solutions_path = path + "/results/MQSRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string + "/" + covariate_type;
//             }

//             if(pde_type == "b") 
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//     } else{
//         if(est_type == "mean"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/SRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type;
//             } else{
//                 solutions_path = path + "/results/SRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type; + "/" + covariate_type;
//             }

//             if(pde_type == "b") 
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//         if(est_type == "quantile"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/MQSRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type + "/eps_" + eps_string;
//             } else{
//                 solutions_path = path + "/results/MQSRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type + "/eps_" + eps_string + "/" + covariate_type;
//             }

//             if(pde_type == "b")
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//     }


//     std::cout << "solution path: " << solutions_path << std::endl; 
    
//     // lambdas sequence 
//     std::vector<double> lambdas; 

//     // lambdas sequence for fine grid of quantiles 
//     std::vector<double> lambdas_1;
//     std::vector<double> lambdas_2;
//     std::vector<double> lambdas_3;
//     std::vector<double> lambdas_4;
//     std::vector<double> lambdas_5;
//     std::vector<double> lambdas_10;
//     std::vector<double> lambdas_15;
//     std::vector<double> lambdas_20;
//     std::vector<double> lambdas_25;
//     std::vector<double> lambdas_30;
//     std::vector<double> lambdas_35;
//     std::vector<double> lambdas_40;
//     std::vector<double> lambdas_45;
//     std::vector<double> lambdas_50;
//     std::vector<double> lambdas_55;
//     std::vector<double> lambdas_60;
//     std::vector<double> lambdas_65;
//     std::vector<double> lambdas_70;
//     std::vector<double> lambdas_75;
//     std::vector<double> lambdas_80;
//     std::vector<double> lambdas_85;
//     std::vector<double> lambdas_90;
//     std::vector<double> lambdas_95;
//     std::vector<double> lambdas_96;
//     std::vector<double> lambdas_97;
//     std::vector<double> lambdas_98;
//     std::vector<double> lambdas_99;

//     std::vector<std::vector<double>> lambdas_alphas;  

//     // define lambda sequence as matrix 
//     DMatrix<double> lambdas_mat;
//     if(est_type == "mean"){
//         if(!return_smoothing){
//             for(double xs = -11.0; xs <= -3.0; xs += 0.05)
//                 lambdas.push_back(std::pow(10,xs));   
//         } else{
//             double lambda_S;  
//             std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
//             if(fileLambdaS_opt.is_open()){
//                 fileLambdaS_opt >> lambda_S; 
//                 fileLambdaS_opt.close();
//             }
//             lambdas.push_back(lambda_S); 
//         }

//         lambdas_mat.resize(lambdas.size(), 1); 
//         for(auto i = 0; i < lambdas_mat.rows(); ++i){
//             lambdas_mat(i,0) = lambdas[i]; 
//         }
//         std::cout << "dim lambdas mat " << lambdas_mat.rows() << " " << lambdas_mat.cols() << std::endl;

//         if(return_smoothing && lambdas.size() > 1){
//             std::cout << "ERROR: you want S, but you are providing more lambdas" << std::endl; 
//         } 
//     }

//     if(est_type == "quantile"){
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_1.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_2.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_3.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_4.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_5.push_back(std::pow(10, x)); 

//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_10.push_back(std::pow(10, x)); 
//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_15.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_20.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_25.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_30.push_back(std::pow(10, x)); 

//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_35.push_back(std::pow(10, x)); 
//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_40.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_45.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_50.push_back(std::pow(10, x)); 
//         for(double x = -6.5; x <= -6.0; x += 0.1) lambdas_55.push_back(std::pow(10, x)); 

//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_60.push_back(std::pow(10, x)); 
//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_65.push_back(std::pow(10, x)); 
//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_70.push_back(std::pow(10, x)); 
//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_75.push_back(std::pow(10, x)); 
//         for(double x = -5.5; x <= -5.0; x += 0.1) lambdas_80.push_back(std::pow(10, x)); 

//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_85.push_back(std::pow(10, x)); 
//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_90.push_back(std::pow(10, x)); 
//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_95.push_back(std::pow(10, x)); 
//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_96.push_back(std::pow(10, x)); 
//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_97.push_back(std::pow(10, x)); 
//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_98.push_back(std::pow(10, x)); 
//         for(double x = -7.5; x <= -6.0; x += 0.1) lambdas_99.push_back(std::pow(10, x)); 

//         lambdas_alphas.push_back(lambdas_1); 
//         lambdas_alphas.push_back(lambdas_2); 
//         lambdas_alphas.push_back(lambdas_3); 
//         lambdas_alphas.push_back(lambdas_4); 
//         lambdas_alphas.push_back(lambdas_5); 
//         lambdas_alphas.push_back(lambdas_10); 
//         lambdas_alphas.push_back(lambdas_15); 
//         lambdas_alphas.push_back(lambdas_20); 
//         lambdas_alphas.push_back(lambdas_25); 
//         lambdas_alphas.push_back(lambdas_30); 
//         lambdas_alphas.push_back(lambdas_35); 
//         lambdas_alphas.push_back(lambdas_40); 
//         lambdas_alphas.push_back(lambdas_45); 
//         lambdas_alphas.push_back(lambdas_50); 
//         lambdas_alphas.push_back(lambdas_55); 
//         lambdas_alphas.push_back(lambdas_60); 
//         lambdas_alphas.push_back(lambdas_65); 
//         lambdas_alphas.push_back(lambdas_70); 
//         lambdas_alphas.push_back(lambdas_75); 
//         lambdas_alphas.push_back(lambdas_80); 
//         lambdas_alphas.push_back(lambdas_85); 
//         lambdas_alphas.push_back(lambdas_90); 
//         lambdas_alphas.push_back(lambdas_95); 
//         lambdas_alphas.push_back(lambdas_96); 
//         lambdas_alphas.push_back(lambdas_97); 
//         lambdas_alphas.push_back(lambdas_98); 
//         lambdas_alphas.push_back(lambdas_99); 


//     }



//     // define spatial domain and regularizing PDE
//     MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X;  

//     if(days_begin_end == ""){
//         y = read_csv<double>(path_data + "/y" + rescale + "_rip" + K + ".csv"); 
//         space_locs = read_csv<double>(path_data + "/locs_ripetute_rip" + K + ".csv");   
//     } else{
//         y = read_csv<double>(path_data + "/y" + rescale + "_" + ".csv"); 
//         space_locs = read_csv<double>(path_data + "/locs_ripetute.csv"); 
//     }
    
//     if(model_type == "param"){
//         X = read_csv<double>(path_data + "/X_" + covariate_type_for_data + "_rip" + K + ".csv");
//     }
        
//     // check dimensions
//     std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
//     std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
//     std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     if(model_type == "param")
//         df.insert(DESIGN_MATRIX_BLK, X);
   

//     // // Laplacian 
//     // if(pde_type == "b")
//     //     std::cout << "ATT You want to run a model with transport field but you are using a PDE with laplacian only"; 
//     // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     // auto L = -laplacian<FEM>();

//     // diffusion + transport 
//     if(pde_type == "")
//         std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport";  
//     DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/" + mesh_type + "/b_" + u_string + "_opt_parametric_" + covariate_type_for_data + "_rip" + K + ".csv");
//     DMatrix<double> u = read_csv<double>(path_data + "/" + mesh_type  + "/u_" + u_string + "_opt_parametric_" + covariate_type_for_data + "_rip" + K + ".csv");
//     DiscretizedVectorField<2, 2> b(b_data);

//     std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
//     std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl ; 

//     auto L = -laplacian<FEM>() + advection<FEM>(b);


//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 

//     for(std::string gcv_summary : gcv_summaries){

//         std::string strategy_gcv; 
//         if(gcv_summary == "_I_appr")
//             strategy_gcv = "first";   
//         if(gcv_summary == "_II_appr")
//             strategy_gcv = "second"; 
//         if(gcv_summary == "_III_appr")
//             strategy_gcv = "third"; 
//         if(gcv_summary == "_IV_appr")
//             strategy_gcv = "fourth";   

//         if(est_type == "mean"){

//             SRPDE model(problem, Sampling::pointwise);
            
//             // set model's data
//             model.set_spatial_locations(space_locs);

//             model.gcv_oss_rip_strategy_set(strategy_gcv); 
            
//             model.set_data(df);

//             model.init();

//             // define GCV function and grid of \lambda_D values

//             // // stochastic
//             // auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
//             // if(return_smoothing){
//             //     std::cout << "ATTENTION: YOU WANT S, BUT STOCHASTIC GCV IS ACTIVATED" << std::endl;
//             // }

//             // exact
//             std::cout << "running EXACT GCV" << std::endl; 
//             auto GCV = model.gcv<ExactEDF>();

            
//             // optimize GCV
//             Grid<fdapde::Dynamic> opt;
//             opt.optimize(GCV, lambdas_mat);
//             SVector<1> best_lambda = opt.optimum();

//             if(!return_smoothing){
//                 // Save lambda sequence 
//                 std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq" + gcv_summary + ".csv");
//                 for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                     fileLambda_S_Seq << std::setprecision(16) << lambdas[i] << "\n"; 
//                 fileLambda_S_Seq.close();

//                 // Save Lambda opt
//                 std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt" + gcv_summary + ".csv");
//                 if(fileLambdaoptS.is_open()){
//                     fileLambdaoptS << std::setprecision(16) << best_lambda[0];
//                     fileLambdaoptS.close();
//                 }
//                 // Save GCV scores
//                 std::ofstream fileGCV_scores(solutions_path + "/gcv_scores" + gcv_summary + ".csv");
//                 for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                     fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//                 fileGCV_scores.close();

//                 // Save edfs
//                 std::ofstream fileEDF(solutions_path + "/edfs" + gcv_summary + ".csv");
//                 for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//                     fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//                 fileEDF.close();

//             }
            
//             if(return_smoothing){
//                 // Save S
//                 DMatrix<double> computedS = GCV.S_get_gcv();
//                 Eigen::saveMarket(computedS, solutions_path + "/S" + gcv_summary + ".mtx");
//             }

//         }

//         if(est_type == "quantile"){
            
//             unsigned int alpha_index = 0; 
//             for(auto alpha : alphas){

//                 unsigned int alpha_int = alpha*100; 
//                 std::string alpha_string = std::to_string(alpha_int); 

//                 std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 
                        
//                     QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                     model_gcv.set_spatial_locations(space_locs);
//                     model_gcv.gcv_oss_rip_strategy_set(strategy_gcv); 

//                     std::vector<double> lambdas = lambdas_alphas[alpha_index];
                
//                     // refactor lambda as a matrix 
//                     lambdas_mat.resize(lambdas.size(), 1); 
//                     for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                         lambdas_mat(i,0) = lambdas[i]; 
//                     }

//                     // set model's data
//                     if(eps_string == "1e+0"){
//                         model_gcv.set_eps_power(0.); 
//                     }
//                     if(eps_string == "1e-0.5"){
//                         model_gcv.set_eps_power(-0.5); 
//                     }
//                     if(eps_string == "1e-1"){
//                         model_gcv.set_eps_power(-1.0); 
//                     }
//                     if(eps_string == "1e-1.5"){
//                         model_gcv.set_eps_power(-1.5); 
//                     }
//                     if(eps_string == "1e-2"){
//                         model_gcv.set_eps_power(-2.0); 
//                     }
//                     if(eps_string == "1e-3"){
//                         model_gcv.set_eps_power(-3.0); 
//                     }
                    
//                     model_gcv.set_data(df);
//                     model_gcv.init();

//                     // define GCV function and grid of \lambda_D values

//                     // // stochastic
//                     // std::cout << "running STOCHASTIC GCV" << std::endl;
//                     // auto GCV = model_gcv.gcv<StochasticEDF>(MC_run, seed);

//                     // exact
//                     std::cout << "running EXACT GCV" << std::endl; 
//                     auto GCV = model_gcv.gcv<ExactEDF>();

//                     // optimize GCV
//                     Grid<fdapde::Dynamic> opt;
//                     opt.optimize(GCV, lambdas_mat);
                    
//                     double best_lambda = opt.optimum()(0,0);
            
//                     std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                     // Save lambda sequence 
//                     std::ofstream fileLambdaS(solutions_path + "/lambdas_seq_" + std::to_string(alpha_int) + gcv_summary +  ".csv");
//                     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                         fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//                     fileLambdaS.close();

//                     // Save lambda GCVopt for all alphas
//                     std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                     if(fileLambdaoptS.is_open()){
//                         fileLambdaoptS << std::setprecision(16) << best_lambda;
//                         fileLambdaoptS.close();
//                     }

//                     // Save GCV 
//                     std::ofstream fileGCV_scores(solutions_path + "/score_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                     for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                     fileGCV_scores.close();

//                     alpha_index = alpha_index + 1; 
//                 }

//             }

//     }


// }

// // run 
// TEST(case_study_mqsrpde_run_obs_rip, NO2) {

//     const std::string month = "gennaio";       // gennaio dicembre 

//     // several days or day fixed?
//     const std::string days_begin_end = "";   // "" "7_11"
//     const std::string day_begin = "7"; 
//     const std::string day_end = "11"; 

//     const std::string day_chosen = "11"; 
//     const std::string K = "24";    // maximum number of repeated observations per each location 

//     const std::string eps_string = "1e+0";  

//     const std::string pde_type = "b";  // ""  "b"  ---> ATT modifica giù
//     const std::string u_string = "1"; 
//     const std::string rescale = "_rescale";    // "" "_rescale"

//     const bool return_smoothing = false;    // if true, metti exact gcv!! 
    
//     const std::string model_type = "param";  // "nonparam" "param"

//     const std::string mesh_type = "canotto_fine";  

//     const std::string cov_strategy = "8"; 
//     const std::string covariate_type = "cov_" + cov_strategy; 
//     std::string covariate_type_for_data; 
//     if(cov_strategy == "8"){
//         covariate_type_for_data = "sqrt.dens_sqrt.elev.original"; 
//     }
//     std::string est_type = "quantile";    // mean quantile
//     std::string gamma0_str = "5"; 

//     std::vector<std::string> gcv_summaries = {"_II_appr"};   // "_I_appr",  "_II_appr", "_III_appr", "_IV_appr" "_fix"

//     std::vector<double> lambdas_forced; 
//     std::vector<double> powers = {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0}; // {-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0}; 
//     std::vector<std::string> powers_str = {"-6.0", "-5.0", "-4.0", "-3.0", "-2.0", "-1.0"};
//     DMatrix<double> lambdas_forced_mult; 
//     lambdas_forced_mult.resize(powers_str.size(), 1); 

//     std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 
//                                   0.30, 0.35, 0.40, 0.45,
//                                   0.5, 
//                                   0.55, 0.60, 0.65, 0.70, 
//                                   0.75, 0.80, 0.85, 
//                                   0.90,
//                                   0.95, 0.96, 0.97, 0.98, 0.99
//                                   };

    
//     bool single_est = false;
//     bool mult_est = true;
//     const unsigned int max_it_convergence_loop = 60; 

//     // Marco 
//     std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
//     std::string path_data; 
//     if(days_begin_end == ""){
//         path_data = path + "/data/MQSRPDE/NO2/" + month + "/day_" + day_chosen;  
//     } else{
//         path_data = path + "/data/MQSRPDE/NO2/" + month + "/days_" + day_begin + "_" + day_end;  
//     }
//     std::string solutions_path; 
  
//     if(days_begin_end == ""){
//         if(est_type == "mean"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/SRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type;
//             } else{
//                 solutions_path = path + "/results/SRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
//             }

//             if(pde_type == "b") 
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//         if(est_type == "quantile"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/MQSRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string;
//             } else{
//                 solutions_path = path + "/results/MQSRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string + "/" + covariate_type;
//             }

//             if(pde_type == "b") 
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//     } else{
//         if(est_type == "mean"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/SRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type;
//             } else{
//                 solutions_path = path + "/results/SRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type; + "/" + covariate_type;
//             }

//             if(pde_type == "b") 
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//         if(est_type == "quantile"){
//             if(model_type == "nonparam"){
//                 solutions_path = path + "/results/MQSRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type + "/eps_" + eps_string;
//             } else{
//                 solutions_path = path + "/results/MQSRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type + "/eps_" + eps_string + "/" + covariate_type;
//             }

//             if(pde_type == "b")
//                 solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//         }

//     }


//     std::cout << "solution path: " << solutions_path << std::endl; 

//     // define spatial domain and regularizing PDE   
//     MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X;  

//     if(days_begin_end == ""){
//         y = read_csv<double>(path_data + "/y" + rescale + "_rip" + K + ".csv"); 
//         space_locs = read_csv<double>(path_data + "/locs_ripetute_rip" + K + ".csv");   
//     } else{
//         y = read_csv<double>(path_data + "/y" + rescale + "_" + ".csv"); 
//         space_locs = read_csv<double>(path_data + "/locs_ripetute.csv"); 
//     }


//     if(model_type == "param"){
//         X = read_csv<double>(path_data + "/X_" + covariate_type_for_data + "_rip" + K + ".csv");
//     }


//     // check dimensions
//     std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
//     std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
//     std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     if(model_type == "param")
//         df.insert(DESIGN_MATRIX_BLK, X);
    

//     // // Laplacian 
//     // if(pde_type == "b")
//     //     std::cout << "ATT You want to run a model with transport field but you are using a PDE with laplacian only"; 
//     // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     // auto L = -laplacian<FEM>();

//     // diffusion + transport 
//     if(pde_type == "")
//         std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
//     DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/" + mesh_type + "/b_" + u_string + "_opt_parametric_" + covariate_type_for_data + "_rip" + K + ".csv");
//     DMatrix<double> u = read_csv<double>(path_data + "/" + mesh_type  + "/u_" + u_string + "_opt_parametric_" + covariate_type_for_data + "_rip" + K + ".csv");

//     DiscretizedVectorField<2, 2> b(b_data);
//     auto L = -laplacian<FEM>() + advection<FEM>(b);

//     std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
//     std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl ; 


//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);


//     std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 


//     for(std::string gcv_summary : gcv_summaries){

//         std::cout << "gcv_summary=" << gcv_summary << std::endl; 

//         std::string strategy_gcv; 
//         if(gcv_summary == "_I_appr")
//             strategy_gcv = "first";   
//         if(gcv_summary == "_II_appr")
//             strategy_gcv = "second"; 
//         if(gcv_summary == "_III_appr")
//             strategy_gcv = "third"; 
//         if(gcv_summary == "_IV_appr")
//             strategy_gcv = "fourth"; 

//         if(gcv_summary == "_fix"){
//             unsigned int count_pow = 0; 
//             for(double xs : powers){
//                 std::cout << "forcing lambda value to" << std::pow(10,xs) << std::endl; 
//                 lambdas_forced.push_back(std::pow(10,xs));  
//                 lambdas_forced_mult(count_pow, 0) = std::pow(10,xs);
//                 count_pow ++;  
//             }
//             std::cout << "dim lambdas_forced = " << lambdas_forced.size() << std::endl; 
//             std::cout << "dim lambdas_forced_mult = " << lambdas_forced_mult.rows() << std::endl; 
    
//         }  

//         if(est_type == "mean"){

//             SRPDE model(problem, Sampling::pointwise);
        
//             // set model's data
//             model.set_spatial_locations(space_locs);

//             // Read optima lambdas 
//             double lambda_S; 

//             if(gcv_summary != "_fix"){
//                 std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt" + gcv_summary + ".csv");
//                 if(fileLambdaS_opt.is_open()){
//                     fileLambdaS_opt >> lambda_S; 
//                     fileLambdaS_opt.close();
//                 }


//                 std::cout << "lambda S" << lambda_S << std::endl;

//                 model.set_lambda_D(lambda_S);
                
//                 model.set_data(df);

//                 model.init();
//                 model.solve();

//                 // Save C++ solution 
//                 DMatrix<double> computedF = model.f();
//                 const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filef(solutions_path + "/f" + gcv_summary + ".csv");
//                 if (filef.is_open()){
//                     filef << computedF.format(CSVFormatf);
//                     filef.close();
//                 }

//                 DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
//                 const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filefn(solutions_path + "/fn" + gcv_summary + ".csv");
//                 if (filefn.is_open()){
//                     filefn << computedFn.format(CSVFormatfn);
//                     filefn.close();
//                 }

//                 if(model_type == "param"){
//                     DMatrix<double> computedBeta = model.beta();
//                     const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filebeta(solutions_path + "/beta" + gcv_summary + ".csv");
//                     if (filebeta.is_open()){
//                         filebeta << computedBeta.format(CSVFormatBeta);
//                         filebeta.close();
//                     }
//                 }

//             } else{

//                 for(int iter = 0; iter < powers.size(); ++iter){
//                     double lambda_forced = lambdas_forced[iter]; 
//                     std::cout << "forcing lambda value to " << lambda_forced << std::endl; 
//                     lambda_S = lambda_forced; 

//                     std::cout << "lambda S" << lambda_S << std::endl;

//                     model.set_lambda_D(lambda_S);
                    
//                     model.set_data(df);

//                     model.init();
//                     model.solve();

//                     // Save C++ solution 
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solutions_path + "/f" + "_fix_" + std::to_string(powers[iter]).substr(0, 4) + ".csv");
//                     if (filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solutions_path + "/fn" + "_fix_" + std::to_string(powers[iter]).substr(0, 4) + ".csv");
//                     if (filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                     if(model_type == "param"){
//                         DMatrix<double> computedBeta = model.beta();
//                         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         std::ofstream filebeta(solutions_path + "/beta" + "_fix_" + std::to_string(powers[iter]).substr(0, 4) + ".csv");
//                         if (filebeta.is_open()){
//                             filebeta << computedBeta.format(CSVFormatBeta);
//                             filebeta.close();
//                         }
//                     }
//                 }


//             }


//             // // Save Psi, Psi_reduced, PsiTD_reduced e X_reduced for degub

//             // SpMatrix<double> Psi_mat = model.Psi(fdapde::models::not_nan());
//             // Eigen::saveMarket(Psi_mat, solutions_path + "/Psi.mtx");

//             // SpMatrix<double> Psi_red_mat = model.Psi_reduced(fdapde::models::not_nan());
//             // Eigen::saveMarket(Psi_red_mat, solutions_path + "/Psi_reduced.mtx");

//             // SpMatrix<double> PsiTD_red_mat = model.PsiTD_reduced(fdapde::models::not_nan());
//             // Eigen::saveMarket(PsiTD_red_mat, solutions_path + "/PsiTD_reduced.mtx");

//             // DMatrix<double> X_mat = model.X_reduced();
//             // Eigen::saveMarket(X_mat, solutions_path + "/X_reduced.mtx");


//         }

//         if(est_type == "quantile"){

//             if(single_est){
//                 std::cout << "-----------------------SINGLE running---------------" << std::endl;

//                 if(gcv_summary != "_fix"){

//                     std::size_t idx = 0;
//                     for(double alpha : alphas){
//                         QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
//                         model.set_spatial_locations(space_locs);
//                         unsigned int alpha_int = alphas[idx]*100;  

//                         std::cout << "alpha=" << alpha_int << "%" << std::endl; 

//                         double lambda; 
//                         std::ifstream fileLambda(solutions_path + "/lambda_s_opt_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                         if(fileLambda.is_open()){
//                             fileLambda >> lambda; 
//                             fileLambda.close();
//                         }

//                         std::cout << "lambda=" << lambda << std::endl; 
//                         model.set_lambda_D(lambda);

//                         // set model data
//                         model.set_data(df);

//                         // solve smoothing problem
//                         model.init();
//                         model.solve();

//                         // Save solution
//                         DMatrix<double> computedF = model.f();
//                         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         std::ofstream filef(solutions_path + "/f_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                         if(filef.is_open()){
//                             filef << computedF.format(CSVFormatf);
//                             filef.close();
//                         }

//                         // DMatrix<double> computedG = model.g();
//                         // const static Eigen::IOFormat CSVFormatg(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         // std::ofstream fileg(solutions_path + "/g_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                         // if(fileg.is_open()){
//                         //     fileg << computedG.format(CSVFormatg);
//                         //     fileg.close();
//                         // }

//                         DMatrix<double> computedFn = model.Psi()*model.f();
//                         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         std::ofstream filefn(solutions_path + "/fn_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                         if(filefn.is_open()){
//                             filefn << computedFn.format(CSVFormatfn);
//                             filefn.close();
//                         }

//                         if(model_type == "param"){
//                             DMatrix<double> computedBeta = model.beta(); 
//                             const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                             std::ofstream filebeta(solutions_path + "/beta_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                             if(filebeta.is_open()){
//                                 filebeta << computedBeta.format(CSVFormatBeta);
//                                 filebeta.close();
//                             }
//                         }

//                         idx++;
//                     }

//                 } else{
                    
//                     for(int idx_lambdas = 0; idx_lambdas < powers.size(); ++idx_lambdas){

//                         std::size_t idx = 0;
//                         for(double alpha : alphas){
//                             QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
//                             model.set_spatial_locations(space_locs);
//                             unsigned int alpha_int = alphas[idx]*100; 

//                             std::cout << "alpha=" << alpha_int << "%" << std::endl;  

//                             double lambda = lambdas_forced[idx_lambdas]; 
//                             std::cout << "lambda value = " << lambda << std::endl; 

//                             model.set_lambda_D(lambda);

//                             // set model data
//                             model.set_data(df);

//                             // solve smoothing problem
//                             model.init();
//                             model.solve();

//                             std::string gcv_summary_tmp = gcv_summary + "_" + powers_str[idx_lambdas];

//                             // Save solution
//                             DMatrix<double> computedF = model.f();
//                             const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                             std::ofstream filef(solutions_path + "/f_" + std::to_string(alpha_int) + gcv_summary_tmp + ".csv");
//                             if(filef.is_open()){
//                                 filef << computedF.format(CSVFormatf);
//                                 filef.close();
//                             }

//                             // DMatrix<double> computedG = model.g();
//                             // const static Eigen::IOFormat CSVFormatg(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                             // std::ofstream fileg(solutions_path + "/g_" + std::to_string(alpha_int) + gcv_summary_tmp + ".csv");
//                             // if(fileg.is_open()){
//                             //     fileg << computedG.format(CSVFormatg);
//                             //     fileg.close();
//                             // }

//                             DMatrix<double> computedFn = model.Psi()*model.f();
//                             const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                             std::ofstream filefn(solutions_path + "/fn_" + std::to_string(alpha_int) + gcv_summary_tmp + ".csv");
//                             if(filefn.is_open()){
//                                 filefn << computedFn.format(CSVFormatfn);
//                                 filefn.close();
//                             }

//                             if(model_type == "param"){
//                                 DMatrix<double> computedBeta = model.beta(); 
//                                 const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                                 std::ofstream filebeta(solutions_path + "/beta_" + std::to_string(alpha_int) + gcv_summary_tmp + ".csv");
//                                 if(filebeta.is_open()){
//                                     filebeta << computedBeta.format(CSVFormatBeta);
//                                     filebeta.close();
//                                 }
//                             }

//                             idx++;
//                         }



//                     }


//                 }



//             }
            
//             if(mult_est){

//                 std::cout << "-----------------------MULTIPLE running---------------" << std::endl;
                
//                 std::string solutions_path_single = solutions_path; 
//                 std::string solutions_path_mult = solutions_path_single + "/g_" + gamma0_str; 
//                 std::cout << "solutions_path_mult=" << solutions_path_mult << std::endl; 

//                 if(gcv_summary != "_fix"){

//                     MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//                     model.set_spatial_locations(space_locs);
//                     model.set_preprocess_option(false); 
//                     model.set_forcing_option(false);
//                     model.set_max_iter(max_it_convergence_loop); 

//                     // use optimal lambda to avoid possible numerical issues
//                     DMatrix<double> lambdas;
//                     DVector<double> lambdas_temp; 
//                     lambdas_temp.resize(alphas.size());
//                     for(std::size_t idx = 0; idx < alphas.size(); ++idx){
//                         unsigned int alpha_int = alphas[idx]*100;  
//                         std::ifstream fileLambdas(solutions_path_single + "/lambda_s_opt_" + std::to_string(alpha_int) + gcv_summary + ".csv");
//                         if(fileLambdas.is_open()){
//                             fileLambdas >> lambdas_temp(idx); 
//                             std::cout << "lambda read in multiple = " << lambdas_temp(idx) << std::endl; 
//                             fileLambdas.close();
//                         }
//                     }
//                     lambdas = lambdas_temp;                
//                     model.setLambdas_D(lambdas);

//                     // set model data
//                     model.set_data(df);

//                     // solve smoothing problem
//                     std::cout << "init multiple" << std::endl; 
//                     model.init();
//                     std::cout << "solve multiple" << std::endl; 
//                     model.solve();

//                     // Save solution
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solutions_path_mult + "/f_all" + gcv_summary + ".csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solutions_path_mult + "/fn_all" + gcv_summary + ".csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }
                    
//                     if(model_type == "param"){
//                         DMatrix<double> computedBeta = model.beta();
//                         const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         std::ofstream filebeta(solutions_path_mult + "/beta_all" + gcv_summary + ".csv");
//                         if(filebeta.is_open()){
//                             filebeta << computedBeta.format(CSVFormatbeta);
//                             filebeta.close();
//                         }
//                     }

//                 } else{

//                     MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//                     model.set_spatial_locations(space_locs);
//                     model.set_preprocess_option(false); 
//                     model.set_forcing_option(false);
//                     model.set_max_iter(max_it_convergence_loop); 

//                     // use optimal lambda to avoid possible numerical issues
//                     model.setLambdas_D(lambdas_forced_mult);

//                     // set model data
//                     model.set_data(df);

//                     // solve smoothing problem
//                     model.init();
//                     model.solve();

//                     // Save solution
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solutions_path_mult + "/f_all" + gcv_summary + ".csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solutions_path_mult + "/fn_all" + gcv_summary + ".csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }
                    
//                     if(model_type == "param"){
//                         DMatrix<double> computedBeta = model.beta();
//                         const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         std::ofstream filebeta(solutions_path_mult + "/beta_all" + gcv_summary + ".csv");
//                         if(filebeta.is_open()){
//                             filebeta << computedBeta.format(CSVFormatbeta);
//                             filebeta.close();
//                         }
//                     }



//                 }
                

//             }
            

//         }        





//     }


// }



// // run CV error 
// TEST(case_study_mqsrpde_run_obs_rip_cv_err, NO2) {

//     std::string CV_type = "";   // "" "2"

//     const std::vector<unsigned int> sims = {1,2,3,4,5,6,7,8,9,10};  
//     unsigned int num_folds; 

//     const std::string month = "gennaio";       // gennaio dicembre 

//     // several days or day fixed?
//     const std::string days_begin_end = "";   // "" "7_11"
//     const std::string day_begin = "7"; 
//     const std::string day_end = "11"; 

//     const std::string day_chosen = "11"; 
//     const std::string K = "24";    // maximum number of repeated observations per each location

//     if(CV_type == ""){
//         if(days_begin_end == "7_11"){
//             num_folds = 84;     
//         } else{
//             if(K=="1"){
//                 num_folds = 17;     
//             }
//             if(K=="2"){
//                 num_folds = 34;     
//             }
//             if(K=="3"){
//                 num_folds = 51;     
//             }
//             if(K=="5"){
//                 num_folds = 84;     
//             }
//             if(K=="10"){
//                 num_folds = 168;     
//             }
//             if(K=="24"){
//                 num_folds = 67;       // non segue il trend delle altre perchè per ragioni computazionali qui alziamo la folder size
//             }
//         }
//     } 

//     if(CV_type == "2"){
//         if(days_begin_end == "7_11"){
//             num_folds = 5;     
//         } else{
//             if(K=="3"){
//                 num_folds = 3;     
//             }
//             if(K=="5"){
//                 num_folds = 5;     
//             }
//             if(K=="10"){
//                 num_folds = 5;     
//             }
//         }
//     }



//     const std::string eps_string = "1e-1.5";   // "1e-0.25" "0"  "1e+0" "1e+0.5" "1e+1" "1e+2"

//     const std::string pde_type = "";  // ""  "b"  ---> ATT modifica giù
//     const std::string u_string = "1"; 
//     const std::string rescale = "_rescale";    // "" "_rescale" 
    
//     const std::string model_type = "nonparam";  // "nonparam" "param"
//     const std::string covariate_type = "dens.new_log.el.orig";   // dens.new_log.el.orig: too long path otherwise

//     std::string est_type = "mean";    // mean quantile

//     std::vector<std::string> model_types = {"", 
//                                             "_II_appr", 
//                                             "_fix_-6.0", "_fix_-5.5", "_fix_-5.0", 
//                                             "_fix_-4.5", "_fix_-4.0", "_fix_-3.5", "_fix_-3.0",
//                                             "_fix_-2.5", "_fix_-2.0", "_fix_-1.5", "_fix_-1.0"
//                                             }; 
                                            
//     // {"", "_II_appr", "_fix_-6.0", "_fix_-5.5", "_fix_-5.0", "_fix_-4.5", "_fix_-4.0", "_fix_-3.5", "_fix_-3.0", "_fix_-2.5", "_fix_-2.0", "_fix_-1.5", "_fix_-1.0"}; 

//     for(int sim : sims){

//         std::cout << "--------------------------------SIMULATION " << std::to_string(sim) << "--------------------------------" << std::endl; 

//         for(std::string gcv_summary : model_types){

//             // Marco 
//             std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
//             std::string path_data; 
//             if(days_begin_end == ""){
//                 path_data = path + "/data/MQSRPDE/NO2/" + month + "/day_" + day_chosen;  
//             } else{
//                 path_data = path + "/data/MQSRPDE/NO2/" + month + "/days_" + day_begin + "_" + day_end;  
//             }
//             std::string solutions_path; 

//             // // Ilenia 
//             // std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
//             // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen; 
//             // std::string solutions_path;

//             if(days_begin_end == ""){
//                 if(est_type == "mean"){
//                     if(model_type == "nonparam"){
//                         solutions_path = path + "/results/SRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type;
//                     } else{
//                         solutions_path = path + "/results/SRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + covariate_type;
//                     }

//                     if(pde_type == "b") 
//                         solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//                 }

//                 if(est_type == "quantile"){
//                     if(model_type == "nonparam"){
//                         solutions_path = path + "/results/MQSRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/eps_" + eps_string;
//                     } else{
//                         solutions_path = path + "/results/MQSRPDE/obs_rip" + K + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + covariate_type + "/eps_" + eps_string;
//                     }

//                     if(pde_type == "b")
//                         solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//                 }

//             } else{
//                 if(est_type == "mean"){
//                     if(model_type == "nonparam"){
//                         solutions_path = path + "/results/SRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type;
//                     } else{
//                         solutions_path = path + "/results/SRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type; + "/" + covariate_type;
//                     }

//                     if(pde_type == "b") 
//                         solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//                 }

//                 if(est_type == "quantile"){
//                     if(model_type == "nonparam"){
//                         solutions_path = path + "/results/MQSRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type + "/eps_" + eps_string;
//                     } else{
//                         solutions_path = path + "/results/MQSRPDE/" + month + "/days_" + day_begin +  "_" + day_end + "/" + model_type + "/" + covariate_type + "/eps_" + eps_string;
//                     }

//                     if(pde_type == "b")
//                         solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
//                 }

//             }
//             std::string CV_path = solutions_path + "/CV_error" + CV_type + "/lamb" + gcv_summary; 

//             std::cout << "CV_path path: " << CV_path << std::endl; 

//             // define spatial domain and regularizing PDE
//             MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_convex_hull");

//             // ATT: parameter cascading è SEMPRE quelli su modelli nonparametrici, anche quando ci sono covariate, 
//             //      tanto già c'è l'approssimazione che vale per la media e non per i quantili

//             // Laplacian 
//             if(pde_type == "b")
//                 std::cout << "ATT You want to run a model with transport field but you are using a PDE with laplacian only"; 
//             DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//             auto L = -laplacian<FEM>();

//             // // diffusion + transport 
//             // if(pde_type == "")
//             //     std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
//             // DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/b_" + u_string + "_opt_nonparametric.csv");
//             // DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_string + "_opt_nonparametric.csv");
//             // DiscretizedVectorField<2, 2> b(b_data);
//             // auto L = -laplacian<FEM>() + advection<FEM>(b);


//             PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//             std::cout << "--------------------------------CV validation model=" << gcv_summary << "--------------------------------" << std::endl; 
//             if(est_type == "mean"){

//                 // Read lambda 
//                 double lambda_S; 
//                 std::ifstream fileLambdaS_opt(CV_path + "/lambda.csv");
//                 if(fileLambdaS_opt.is_open()){
//                     fileLambdaS_opt >> lambda_S; 
//                     fileLambdaS_opt.close();
//                 }
//                 std::cout << "lambda=" << lambda_S << std::endl;

//                 for(int k = 1; k <= num_folds; ++k){

//                     std::string k_str = std::to_string(k); 
//                     std::cout << "fold #" << k_str << std::endl;

//                     SRPDE model(problem, Sampling::pointwise);

//                     // import data and locs from files
//                     DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X;  

//                     y = read_csv<double>(CV_path + "/sim_" + std::to_string(sim) + "/fold_" + k_str + "/y_train.csv"); 
//                     if(model_type == "param")
//                         X = read_csv<double>(CV_path + "/sim_" + std::to_string(sim) + "/fold_" + k_str + "/X_train.csv");

//                     space_locs = read_csv<double>(CV_path + "/sim_" + std::to_string(sim) + "/fold_" + k_str + "/locs_train.csv");   

//                     // // check dimensions
//                     // std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
//                     // std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
//                     // std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

//                     BlockFrame<double, int> df;
//                     df.insert(OBSERVATIONS_BLK, y);
//                     if(model_type == "param")
//                         df.insert(DESIGN_MATRIX_BLK, X);
                
//                     // set model's data
//                     model.set_spatial_locations(space_locs);
//                     model.set_lambda_D(lambda_S);     
//                     model.set_data(df);

//                     model.init();
//                     model.solve();

//                     // Save C++ solution 
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(CV_path + "/sim_" + std::to_string(sim) + "/fold_" + k_str + "/f.csv");
//                     if (filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(CV_path + "/sim_" + std::to_string(sim) + "/fold_" + k_str  + "/fn.csv");
//                     if (filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                     if(model_type == "param"){
//                         DMatrix<double> computedBeta = model.beta();
//                         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                         std::ofstream filebeta(CV_path + "/sim_" + std::to_string(sim) + "/fold_" + k_str  + "/beta.csv");
//                         if (filebeta.is_open()){
//                             filebeta << computedBeta.format(CSVFormatBeta);
//                             filebeta.close();
//                         }
//                     }

//                 }





//             }

//         }

//     }

// }



