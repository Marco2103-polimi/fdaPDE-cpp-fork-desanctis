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

    const std::string month = "dicembre";       // gennaio dicembre 
    const std::string day_chosen = "8";   // nota: single number WITHOUT zero padding

    const std::string eps_string = "1e-1.5";   // "1e-0.25" "0"  "1e+0" "1e+0.5" "1e+1" "1e+2"

    std::string pde_type = "tr";  // ""  "tr" 
    const std::string u_string = "1e+0"; 

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

    const std::string cov_strategy = "9"; 
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
    if(cov_strategy == "9"){
        covariate_type_for_data = "log.elev.original"; 
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
    std::string path_data = path + "/data/MQSRPDE/PM10/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

    if(est_type == "mean"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results_PM10/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results_PM10/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    if(est_type == "quantile"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results_PM10/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string;
        } else{
            solutions_path = path + "/results_PM10/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type + "/eps_" + eps_string;
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
        double step = 0.1; // coarse: 0.5    fine: 0.1 
        for(double x = -6.0; x <= -2.5; x += step) lambdas_1_5.push_back(std::pow(10, x)); 
        for(double x = -6.5; x <= -3.0; x += step) lambdas_10_25.push_back(std::pow(10, x));
        for(double x = -6.5; x <= -1.5; x += step) lambdas_30_70.push_back(std::pow(10, x)); 
        for(double x = -6.5; x <= -2.0; x += step) lambdas_75_90.push_back(std::pow(10, x)); 
        for(double x = -6.0; x <= -2.0; x += step) lambdas_95_99.push_back(std::pow(10, x));
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

    if(model_type == "nonparam"){
        b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + ".csv");
        u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + ".csv");      
    } else{
        b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + "_" + covariate_type_for_data + ".csv");
        u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + "_" + covariate_type_for_data + ".csv");    
    }

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

    const std::string month = "dicembre";       // gennaio dicembre 
    const std::string day_chosen = "8"; 

    const std::string eps_string = "1e-1.5";   // "0" "1e+0" "1e+1"

    std::string pde_type = "tr";  // "" "tr" "tr2"
    const std::string u_string = "1e+0"; 

    std::string est_type = "quantile";    // mean quantile
    bool single_est = true;
    bool mult_est = true;

    bool save_fpirls_iter = false; 
    std::vector<unsigned int> num_fpirls_iter_per_alpha = {200, 200, 200, 200, 200, 
                                                           200, 200, 200, 200, 200, 200, 200, 200, 
                                                           200, 
                                                           200, 200, 200, 200, 200, 200, 200, 200, 200, 
                                                           200, 200, 200, 200}; 

   // std::vector<unsigned int> num_fpirls_iter_per_alpha = {10, 10}; 

    const std::string fpirls_iter_strategy = "1"; 

    double gamma0 = 1.;   // initial crossing penalty parameter 
    std::string gamma0_str = "1"; 

    const std::string model_type = "param";  // "nonparam" "param"
    const std::string  cascading_model_type = "parametric";  // ok parametric2 anche per tr3
    if(cascading_model_type == "parametric" || cascading_model_type == "parametric2"){
        pde_type = pde_type + "_par"; 
    }
    
    
    const std::string cov_strategy = "9"; 
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
    if(cov_strategy == "9"){
        covariate_type_for_data = "log.elev.original"; 
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
    std::string path_data = path + "/data/MQSRPDE/PM10/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

     if(est_type == "mean"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results_PM10/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results_PM10/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    if(est_type == "quantile"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results_PM10/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/eps_" + eps_string;
        } else{
            solutions_path = path + "/results_PM10/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type + "/eps_" + eps_string;
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

    std::cout << "path b: " << path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + "_" + covariate_type_for_data + ".csv" << std::endl; 
    if(model_type == "nonparam"){
        b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + ".csv");
        u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + ".csv");      
    } else{
        b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type + "_" + covariate_type_for_data + ".csv");
        u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + "_" + covariate_type_for_data + ".csv");    
    }

    
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