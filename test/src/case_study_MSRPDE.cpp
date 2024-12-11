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
#include "../../fdaPDE/models/regression/msrpde.h"
using fdapde::models::SRPDE;
using fdapde::models::MSRPDE;

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
TEST(case_study_msrpde_gcv, NO2) {

    const std::string month = "gennaio";       // gennaio dicembre 
    const std::string day_chosen = "11"; 
    const std::string rescale_data = "_rescale";   // "" "_rescale"

    std::string pde_type = "tr";  // ""  "tr" 
    const std::string u_string = "1e-1"; 

    // const bool return_smoothing = false;    // if true, metti exact gcv!! 
    std::string gcv_type = "exact";   // "exact" "stochastic"  ---> MODIFICA ANCHE GIU'!

    const unsigned int num_fpirls_iter = 200;  

    std::size_t seed = 438172;
    unsigned int MC_run = 100; 
    const std::string model_type = "param";  // "nonparam" "param"
    const std::string  cascading_model_type = "parametric";  // "parametric" "nonparametric" 
    if(cascading_model_type == "parametric"){
        pde_type = pde_type + "_par"; 
    }

    std::string cov_strategy;
    if(model_type == "param"){
        cov_strategy = "7"; 
    } 
    const std::string covariate_type = "cov_" + cov_strategy; 
    std::string covariate_type_for_data = "";   // if model has covariates, it is overwritten below 
    if(cov_strategy == "1" || cov_strategy == "2"){
        covariate_type_for_data = "_dens.new_log.elev.original"; 
    }
    if(cov_strategy == "3"){
        covariate_type_for_data = "_log.dens_log.elev.original"; 
    }
    if(cov_strategy == "4"){
        covariate_type_for_data = "_log.dens.ch_log.elev.original"; 
    }
    if(cov_strategy == "5"){
        covariate_type_for_data = "_N.dens.new_N.elev.original"; 
    }
    if(cov_strategy == "6"){
        covariate_type_for_data = "_log.dens_N.elev.original"; 
    }
    if(cov_strategy == "7"){
        covariate_type_for_data = "_sqrt.dens_log.elev.original"; 
    }
    if(cov_strategy == "8"){
        covariate_type_for_data = "_sqrt.dens_sqrt.elev.original"; 
    }

    const std::string mesh_type = "canotto_fine";  // "square" "esagoni" "convex_hull" "CH_fine" "canotto" "canotto_fine"
    const std::string mesh_type_param_casc = mesh_type; 

    std::string est_type = "mixed";    // mean mixed

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/case_studies/mixed_NO2"; 
    std::string path_data = path + "/data/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

    if(est_type == "mean"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    if(est_type == "mixed"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/MSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results/MSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

        solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    std::cout << "solution path: " << solutions_path << std::endl; 
    
    // lambdas sequence 
    std::vector<double> lambdas; 
    DMatrix<double> lambdas_mat;
    double seq_start; double seq_end; double seq_by; 
    if(est_type == "mean"){
        seq_start = -3.5; 
        seq_end = -0.5; 
        seq_by = 0.05; 
    }
    if(est_type == "mixed"){
        seq_start = -4.0; 
        seq_end = 2.0; 
        seq_by = 0.1; 
    }

    if(est_type == "mean"){
        // if(return_smoothing){
        //     double lambda_S;  
        //     std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
        //     if(fileLambdaS_opt.is_open()){
        //         fileLambdaS_opt >> lambda_S; 
        //         fileLambdaS_opt.close();
        //     }
        //     lambdas.push_back(lambda_S);      
        // }
        // if(return_smoothing && lambdas.size() > 1){
        //     std::cout << "ERROR: you want S, but you are providing more lambdas" << std::endl; 
        // } 
    }

    for(double xs = seq_start; xs <= seq_end; xs += seq_by)
        lambdas.push_back(std::pow(10,xs));   


    // define lambda sequence as matrix 
    lambdas_mat.resize(lambdas.size(), 1); 
    for(auto i = 0; i < lambdas_mat.rows(); ++i){
        lambdas_mat(i,0) = lambdas[i]; 
    }
    std::cout << "dim lambdas mat: " << lambdas_mat.rows() << " " << lambdas_mat.cols() << std::endl;

    // define spatial domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X; DMatrix<double> Z; DVector<unsigned int> ids_groups; 

    y = read_csv<double>(path_data + "/y" + rescale_data + ".csv"); 
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;

    space_locs = read_csv<double>(path_data + "/locs" + ".csv");   
     std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;

    if(model_type == "param"){
        X = read_csv<double>(path_data + "/X" + covariate_type_for_data + ".csv");
        std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;
    }
    if(est_type == "mixed"){
        Z = read_csv<double>(path_data + "/Z.csv");
        std::cout << "dim Z " << Z.rows() << " " << Z.cols() << std::endl;
        std::cout << "max(Z) = " << Z.maxCoeff() << std::endl;

        ids_groups = read_csv<unsigned int>(path_data + "/ids_groups.csv");
        std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
        std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 
    }
      

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(model_type == "param")
        df.insert(DESIGN_MATRIX_BLK, X);
    if(est_type == "mixed")
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);
   

    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
    DMatrix<double, Eigen::RowMajor> b_data; 
    DMatrix<double> u; 
    b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type +  covariate_type_for_data + ".csv");
    u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + covariate_type_for_data + ".csv");    
    std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl;
    std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
    DiscretizedVectorField<2, 2> b(b_data);    
    auto L = -laplacian<FEM>() + advection<FEM>(b); 
    
    // Only Laplacian
    //auto L = -laplacian<FEM>();
    
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

        // // stochastic
        // auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
        // if(return_smoothing){
        //     std::cout << "ATTENTION: YOU WANT S, BUT STOCHASTIC GCV IS ACTIVATED"; 
        // }

        // exact
        auto GCV = model.gcv<ExactEDF>();
           
        // optimize GCV
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas_mat);
        SVector<1> best_lambda = opt.optimum();

        // if(!return_smoothing){
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

        // }
        
        // if(return_smoothing){
        //     // Save S
        //     DMatrix<double> computedS = GCV.S_get_gcv();
        //     Eigen::saveMarket(computedS, solutions_path + "/S.mtx");
        // }

    }

    if(est_type == "mixed"){
                            
        MSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise);
        model_gcv.set_spatial_locations(space_locs);
        model_gcv.set_data(df);
        model_gcv.set_ids_groups(ids_groups); 
        model_gcv.set_fpirls_max_iter(num_fpirls_iter); 

        model_gcv.init();


        // define GCV function and grid of \lambda_D values
        // stochastic
        auto GCV = model_gcv.gcv<ExactEDF>();
        // optimize GCV
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas_mat);
        
        double best_lambda = opt.optimum()(0,0);

        std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

        // Save lambda sequence 
        std::ofstream fileLambdaS(solutions_path + "/lambdas_seq.csv");
        for(std::size_t i = 0; i < lambdas.size(); ++i) 
            fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
        fileLambdaS.close();

        // Save lambda GCVopt for all alphas
        std::ofstream fileLambdaoptS(solutions_path + "/lambdas_opt.csv");
        if(fileLambdaoptS.is_open()){
            fileLambdaoptS << std::setprecision(16) << best_lambda;
            fileLambdaoptS.close();
        }

        // Save GCV 
        std::ofstream fileGCV_scores(solutions_path + "/score.csv");
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


// run 
TEST(case_study_msrpde_run, NO2) {

    const std::string month = "gennaio";       // gennaio dicembre 
    const std::string day_chosen = "11"; 
    const std::string rescale_data = "_rescale";   // "" "_rescale"

    std::string pde_type = "tr";  // ""  "tr" 
    const std::string u_string = "1e-1"; 

    const unsigned int num_fpirls_iter = 200;  

    std::size_t seed = 438172;
    unsigned int MC_run = 100; 
    const std::string model_type = "param";  // "nonparam" "param"
    const std::string  cascading_model_type = "parametric";  // "parametric" "nonparametric" 
    if(cascading_model_type == "parametric"){
        pde_type = pde_type + "_par"; 
    }

    std::string cov_strategy;
    if(model_type == "param"){
        cov_strategy = "7"; 
    } 
    const std::string covariate_type = "cov_" + cov_strategy; 
    std::string covariate_type_for_data = "";   // if model has covariates, it is overwritten below 
    if(cov_strategy == "1" || cov_strategy == "2"){
        covariate_type_for_data = "_dens.new_log.elev.original"; 
    }
    if(cov_strategy == "3"){
        covariate_type_for_data = "_log.dens_log.elev.original"; 
    }
    if(cov_strategy == "4"){
        covariate_type_for_data = "_log.dens.ch_log.elev.original"; 
    }
    if(cov_strategy == "5"){
        covariate_type_for_data = "_N.dens.new_N.elev.original"; 
    }
    if(cov_strategy == "6"){
        covariate_type_for_data = "_log.dens_N.elev.original"; 
    }
    if(cov_strategy == "7"){
        covariate_type_for_data = "_sqrt.dens_log.elev.original"; 
    }
    if(cov_strategy == "8"){
        covariate_type_for_data = "_sqrt.dens_sqrt.elev.original"; 
    }

    const std::string mesh_type = "canotto_fine";  // "square" "esagoni" "convex_hull" "CH_fine" "canotto" "canotto_fine"
    const std::string mesh_type_param_casc = mesh_type; 

    std::string est_type = "mixed";    // mean mixed

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/case_studies/mixed_NO2"; 
    std::string path_data = path + "/data/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

    if(est_type == "mean"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    if(est_type == "mixed"){
        if(model_type == "nonparam"){
            solutions_path = path + "/results/MSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/results/MSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + covariate_type;
        }

        solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    }

    std::cout << "solution path: " << solutions_path << std::endl; 
    

    // define spatial domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X; DMatrix<double> Z; DVector<unsigned int> ids_groups; 

    y = read_csv<double>(path_data + "/y" + rescale_data + ".csv"); 
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;

    space_locs = read_csv<double>(path_data + "/locs" + ".csv");   
     std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;

    if(model_type == "param"){
        X = read_csv<double>(path_data + "/X" + covariate_type_for_data + ".csv");
        std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;
    }
    if(est_type == "mixed"){
        Z = read_csv<double>(path_data + "/Z.csv");
        std::cout << "dim Z " << Z.rows() << " " << Z.cols() << std::endl;
        std::cout << "max(Z) = " << Z.maxCoeff() << std::endl;

        ids_groups = read_csv<unsigned int>(path_data + "/ids_groups.csv");
        std::cout << "dim ids_groups = " << ids_groups.size() << std::endl;
        std::cout << "max(ids_groups) = " << ids_groups.maxCoeff() << std::endl; 
    }
      

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(model_type == "param")
        df.insert(DESIGN_MATRIX_BLK, X);
    if(est_type == "mixed")
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);
   

    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
    DMatrix<double, Eigen::RowMajor> b_data; 
    DMatrix<double> u; 
    b_data  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/b_" + u_string + "_opt_" + cascading_model_type +  covariate_type_for_data + ".csv");
    u  = read_csv<double>(path_data + "/" + mesh_type_param_casc + "/u_" + u_string + "_opt_" + cascading_model_type  + covariate_type_for_data + ".csv");    
    std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl;
    std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
    DiscretizedVectorField<2, 2> b(b_data);    
    auto L = -laplacian<FEM>() + advection<FEM>(b); 
    
    // Only Laplacian
    //auto L = -laplacian<FEM>();
    
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

    if(est_type == "mixed"){

        MSRPDE<SpaceOnly> model(problem, Sampling::pointwise);
        model.set_spatial_locations(space_locs);
        model.set_fpirls_max_iter(num_fpirls_iter);
        model.set_ids_groups(ids_groups);  
        
        double lambda; 
        std::ifstream fileLambda(solutions_path + "/lambdas_opt.csv");
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
        std::ofstream filef(solutions_path + "/f.csv");
        if(filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
        }

        DMatrix<double> computedG = model.g();
        const static Eigen::IOFormat CSVFormatg(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream fileg(solutions_path + "/g.csv");
        if(fileg.is_open()){
            fileg << computedG.format(CSVFormatg);
            fileg.close();
        }

        DMatrix<double> computedFn = model.Psi()*model.f();
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(solutions_path + "/fn.csv");
        if(filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }     

        std::vector<DVector<double>> temp_bhat = model.b_hat(); 
        DMatrix<double> computed_b;
        computed_b.resize(temp_bhat.size(), model.p());   // m x p
        for(int i=0; i<temp_bhat.size(); ++i){
            computed_b.row(i) = temp_bhat[i]; 
        }
        const static Eigen::IOFormat CSVFormatb(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream fileb(solutions_path + "/b_random.csv");
        if(fileb.is_open()){
            fileb << computed_b.format(CSVFormatb);
            fileb.close();
        }

        double computedsigmahat = std::sqrt(model.sigma_sq_hat());
        std::ofstream filesigmahat(solutions_path + "/sigma_hat.csv");
        if(filesigmahat.is_open()){
            filesigmahat << std::setprecision(16) << computedsigmahat << "\n"; 
            filesigmahat.close();
        }

        DMatrix<double> computedsigma_b_hat = model.Sigma_b();
        const static Eigen::IOFormat CSVFormatsigma_b_hat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filesigma_b_hat(solutions_path + "/Sigma_b_hat.csv");
        if(filesigma_b_hat.is_open()){
            filesigma_b_hat << std::setprecision(16) << computedsigma_b_hat.format(CSVFormatsigma_b_hat);
            filesigma_b_hat.close();
        }


        if(model_type == "param"){
            DMatrix<double> computedBeta = model.beta(); 
            const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filebeta(solutions_path + "/beta.csv");
            if(filebeta.is_open()){
                filebeta << computedBeta.format(CSVFormatBeta);
                filebeta.close();
            }
        }


        // // Save Psi, R0 and R1 per l'inferenza       
        // SpMatrix<double> Psi_mat = model.Psi(fdapde::models::not_nan());
        // Eigen::saveMarket(Psi_mat, solutions_path_single + "/Psi" + ".mtx");

        // SpMatrix<double> R0_mat = model.R0(); 
        // Eigen::saveMarket(R0_mat, solutions_path_single + "/R0" + ".mtx");

        // SpMatrix<double> R1_mat = model.R1(); 
        // Eigen::saveMarket(R1_mat, solutions_path_single + "/R1" + ".mtx");
 

    }

}
        




