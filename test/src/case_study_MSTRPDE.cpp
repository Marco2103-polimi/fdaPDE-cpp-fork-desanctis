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
using fdapde::models::SpaceOnly;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;
using fdapde::models::SRPDE;
using fdapde::models::STRPDE;
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

// for k-fold
#include "../../fdaPDE/calibration/kfold_cv.h"
#include "../../fdaPDE/calibration/rmse.h"
using fdapde::calibration::KCV;
using fdapde::calibration::RMSE;



// gcv 
TEST(case_study_mstrpde_gcv, NO2) {

    const bool infraday_analysis = true; 
    std::string results_str; 
    std::string month; 
    std::string day_chosen; 
    if(infraday_analysis){
        month = "gennaio"; 
        day_chosen = "15"; 
        results_str = "results_infraday";
    } else{
        results_str = "results"; 
    }

    unsigned int num_folds;
    const std::string CV_type = "5-folds";     // "GCV"  "5-folds" "10-folds"
    unsigned int seed_kfold; 
    if(CV_type != "GCV"){
        num_folds = 5; 
        seed_kfold = 254; 
    }
    const bool shuffle_kfold = true; 

    std::string rescale_data = "_sqrt";   // "", "_rescale", oppure inserisci la trasformazione che vuoi 
    const bool normalized_loss_flag = true;  // true to normalize the loss

    std::string pde_type = "tr";  // ""  "tr" 
    const std::string u_string = "1e-1"; 

    const unsigned int num_fpirls_iter = 15;  

    // define time domain 
    double t0;
    double tf;
    if(!infraday_analysis){
        t0 = 0.0;
        tf = 22.0;
    } else{
        t0 = 0.0;
        tf = 23.0;
    }

    const unsigned int M = 6;  // number of time mesh nodes 
    const std::string M_string = "_M" + std::to_string(M);
    Triangulation<1, 1> time_mesh(t0, tf, M-1);  // interval [t0, tf] with M-1 knots

    std::size_t seed = 438172;
    unsigned int MC_run = 100; 
    const std::string model_type_root = "param";  // "nonparam" "param"
    const std::string model_type = model_type_root + M_string; 
    const std::string  cascading_model_type = "nonparametric";  // "parametric" "nonparametric" 
    if(cascading_model_type == "parametric"){
        pde_type = pde_type + "_par"; 
    }

    std::string cov_strategy;
    if(model_type_root == "param"){
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

    const std::string mesh_type = "canotto_fine";    // fine come la run !! 

    std::string est_type = "mixed";    // mean mixed


    bool has_fix_cov = (model_type == "param"); 
    bool has_rnd_cov = (est_type == "mixed"); 


    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/case_studies/mixed_NO2"; 

    std::string path_data;  
    if(!infraday_analysis){
        path_data = path + "/data/space-time";  
    } else{
        path_data = path + "/data_infraday/" + month + "/day_" + day_chosen;  
    }
   
    std::string solutions_path; 

    std::string sigla_model; 
    if(est_type == "mean"){
        sigla_model = "STRPDE"; 
    }
    if(est_type == "mixed"){
        sigla_model = "MSTRPDE"; 
    }


    if(!infraday_analysis){
        if(model_type_root == "nonparam"){
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + model_type + "/" + CV_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + model_type + "/" + CV_type + "/" + mesh_type + "/" + covariate_type;
        }
        if(pde_type != "")
            solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
    } else{
        if(model_type_root == "nonparam"){
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + CV_type + "/" + mesh_type;
        } else{
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + CV_type + "/" + mesh_type + "/" + covariate_type;
        }
        if(pde_type != "")
            solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string;             
    }


    std::cout << "data path: " << path_data << std::endl; 
    std::cout << "solution path: " << solutions_path << std::endl; 
    
    // lambdas sequence -> con loss normalizzate 
    std::vector<double> lambdas; 
    double seq_start_space; double seq_end_space; double seq_by_space; 
    double seq_start_time; double seq_end_time; double seq_by_time; 
    if(est_type == "mean"){
        seq_start_space = -6.5; 
        seq_end_space = -1.0; 
        seq_by_space = 0.1; 

        seq_start_time = -4.0; 
        seq_end_time = -4.0; 
        seq_by_time = 1.0; 
    }
    if(est_type == "mixed"){
        seq_start_space = -6.0; 
        seq_end_space = -4.0; 
        seq_by_space = 0.1; 

        seq_start_time = -8.0; 
        seq_end_time = -8.0; 
        seq_by_time = 2.0; 
    }

    std::vector<double> lambdas_d; std::vector<double> lambdas_t; std::vector<DVector<double>> lambdas_d_t;
    for(double xs = seq_start_space; xs <= seq_end_space; xs += seq_by_space)    
        lambdas_d.push_back(std::pow(10,xs));

    for(double xt = seq_start_time; xt <= seq_end_time; xt += seq_by_time)
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
    std::cout << "dim lambdas mat: " << lambdas_mat.rows() << " " << lambdas_mat.cols() << std::endl;
    std::cout << "max lambdas mat: " << lambdas_mat.maxCoeff() << std::endl;
    std::cout << "min lambdas mat: " << lambdas_mat.minCoeff() << std::endl;


    // std::vector<DVector<double>> lambdas_kfold;
    // for(double xs = seq_start_space; xs <= seq_end_space; xs += seq_by_space) 
    //     for(double xt = seq_start_time; xt <= seq_end_time; xt += seq_by_time) 
    //         lambdas_kfold.push_back(SVector<2>(std::pow(10, xs), std::pow(10, xt)));


    // define spatial domain
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);   

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> time_locs; 
    DMatrix<double> X; DMatrix<double> Z; DVector<unsigned int> ids_groups; 

    y = read_csv<double>(path_data + "/y" + rescale_data + ".csv");
    std::cout << "path y = " << path_data + "/y" + rescale_data + ".csv" << std::endl; 
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;

    // check number of missing values
    int count_na = 0;
    double max_y = -1000.; // for debug
    double min_y = 1000.; // for debug
    for (int i = 0; i < y.rows(); ++i) {
        for (int j = 0; j < y.cols(); ++j) {
            if(std::isnan(y(i,j))) {
                ++count_na;
            } else{
                if(y(i,j) > max_y)
                    max_y = y(i,j);
                if(y(i,j) < min_y)
                    min_y = y(i,j);
            }
        }
    }
    std::cout << "num missing values = " << count_na << std::endl;
    std::cout << "max y = " << max_y << std::endl;
    std::cout << "min y = " << min_y << std::endl;


    time_locs = read_csv<double>(path_data + "/time_locs.csv");  
    std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
    std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

    space_locs = read_csv<double>(path_data + "/locs.csv");  
    std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
    std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

    if(model_type_root == "param"){
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
    df.stack(OBSERVATIONS_BLK, y);  // ATT stack for space-time models
    if(model_type_root == "param")
        df.insert(DESIGN_MATRIX_BLK, X);
    if(est_type == "mixed")
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);
   

    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
    DMatrix<double, Eigen::RowMajor> b_data; 
    DMatrix<double> u; 
    std::cout << "path b: " << path_data + "/" + mesh_type + "/b_" + u_string + "_opt_" + cascading_model_type +  covariate_type_for_data + ".csv" << std::endl; 
    b_data  = read_csv<double>(path_data + "/" + mesh_type + "/b_" + u_string + "_opt_" + cascading_model_type +  covariate_type_for_data + ".csv");
    u  = read_csv<double>(path_data + "/" + mesh_type + "/u_" + u_string + "_opt_" + cascading_model_type  + covariate_type_for_data + ".csv");    
    std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl;
    std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
    DiscretizedVectorField<2, 2> b(b_data);    
    auto Ld = -laplacian<FEM>() + advection<FEM>(b); 
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);


    // // Only Laplacian
    // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1); // rhs 
    // auto Ld = -laplacian<FEM>();   
    // PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    


    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);


    // // Save quadrature nodes 
    // DMatrix<double> quad_nodes = space_penalty.quadrature_nodes();  
    // std::cout << "rows quad nodes = " << quad_nodes.rows() << std::endl; 
    // std::cout << "cols quad nodes = " << quad_nodes.cols() << std::endl; 
    // std::ofstream file_quad(path_data + "/" + mesh_type + "/quadrature_nodes.csv");
    // for(int i = 0; i < quad_nodes.rows(); ++i) {
    //     file_quad << quad_nodes(i, 0) << "," << quad_nodes(i, 1) << "\n";
    // }
    // file_quad.close();


    SVector<2> best_lambda;

    if(CV_type == "GCV"){
        std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 
    } else{
        std::cout << "-----------------------------" + CV_type + " STARTS------------------------" << std::endl; 
    }
       

    if(est_type == "mean"){

        STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);

        // set model's data
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);
        model.set_normalize_loss(normalized_loss_flag);
        
        model.set_data(df);
        model.init();


        if(CV_type == "GCV"){

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
            best_lambda = opt.optimum();

            std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

            // Save scores 
            std::ofstream fileGCV_scores(solutions_path + "/score.csv");
            std::cout << "dim GCV.gcvs() = " << GCV.gcvs().size() << std::endl;
            for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
            fileGCV_scores.close();


            std::ofstream fileGCV_edf(solutions_path + "/edf.csv");
            for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
            fileGCV_edf.close();

        } else{


            std::cout << "initialize KCV_" << std::endl; 
            
            KCV KCV_(num_folds, seed_kfold, shuffle_kfold);

            std::cout << "call fit KCV_" << std::endl; 
            KCV_.fit(model, lambdas_mat, RMSE(model), time_locs, space_locs, y, 
                    has_fix_cov = has_fix_cov, has_rnd_cov = has_rnd_cov);

            best_lambda = KCV_.optimum(); 
            
            // Save scores 
            std::ofstream fileKCV_scores(solutions_path + "/score.csv");
            std::cout << "len KCV_.avg_scores() = " << KCV_.avg_scores().size() << std::endl;
            for(std::size_t i = 0; i < KCV_.avg_scores().size(); ++i) 
                fileKCV_scores << std::setprecision(16) << KCV_.avg_scores()[i] << "\n"; 
            fileKCV_scores.close();


        }



        // Save lambda sequence 
        std::ofstream fileLambdaS(solutions_path + "/lambdas_S_seq.csv");
        for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
            fileLambdaS << std::setprecision(16) << lambdas_d[i] << "\n"; 
        fileLambdaS.close();

        std::ofstream fileLambda_T_Seq(solutions_path + "/lambdas_T_seq.csv");
        for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
            fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
        fileLambda_T_Seq.close();


        // Save lambda GCVopt for all alphas
        std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt.csv");
        if(fileLambdaoptS.is_open()){
          fileLambdaoptS << std::setprecision(16) << best_lambda[0];
          fileLambdaoptS.close();
        }
        std::ofstream fileLambdaoptT(solutions_path + "/lambda_t_opt.csv");
        if(fileLambdaoptT.is_open()){
          fileLambdaoptT << std::setprecision(16) << best_lambda[1];
          fileLambdaoptT.close();
        }


    }

    if(est_type == "mixed"){
                            
        MSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise);    
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);
        model.set_normalize_loss(normalized_loss_flag);
        
        // set model 
        model.set_data(df);
        model.set_ids_groups(ids_groups); 

        model.set_fpirls_max_iter(num_fpirls_iter); 

        model.init();


        if(CV_type == "GCV"){

            // define GCV function and grid of \lambda_D values
            auto GCV = model.gcv<ExactEDF>();
            // optimize GCV
            Grid<fdapde::Dynamic> opt;
            opt.optimize(GCV, lambdas_mat);
            
            best_lambda = opt.optimum();

            std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

            // Save GCV 
            std::ofstream fileGCV_scores(solutions_path + "/score.csv");
            std::cout << "dim GCV.gcvs() = " << GCV.gcvs().size() << std::endl;
            for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
            fileGCV_scores.close();


            std::ofstream fileGCV_edf(solutions_path + "/edf.csv");
            for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
            fileGCV_edf.close();


        } else{

            KCV KCV_(num_folds, seed_kfold, shuffle_kfold);
            KCV_.fit(model, lambdas_mat, RMSE(model), time_locs, space_locs, y,
                    has_fix_cov = has_fix_cov, has_rnd_cov = has_rnd_cov);

            best_lambda = KCV_.optimum(); 
            
            // Save scores 
            std::ofstream fileKCV_scores(solutions_path + "/score.csv");
            std::cout << "len KCV_.avg_scores() = " << KCV_.avg_scores().size() << std::endl;
            for(std::size_t i = 0; i < KCV_.avg_scores().size(); ++i) 
                fileKCV_scores << std::setprecision(16) << KCV_.avg_scores()[i] << "\n"; 
            fileKCV_scores.close();


        }


        // Save lambda sequence 
        std::ofstream fileLambdaS(solutions_path + "/lambdas_S_seq.csv");
        for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
            fileLambdaS << std::setprecision(16) << lambdas_d[i] << "\n"; 
        fileLambdaS.close();

        std::ofstream fileLambda_T_Seq(solutions_path + "/lambdas_T_seq.csv");
        for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
            fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
        fileLambda_T_Seq.close();


        // Save lambda GCVopt 
        std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt.csv");
        if(fileLambdaoptS.is_open()){
          fileLambdaoptS << std::setprecision(16) << best_lambda[0];
          fileLambdaoptS.close();
        }
        std::ofstream fileLambdaoptT(solutions_path + "/lambda_t_opt.csv");
        if(fileLambdaoptT.is_open()){
          fileLambdaoptT << std::setprecision(16) << best_lambda[1];
          fileLambdaoptT.close();
        }


    }


}


// run 
TEST(case_study_mstrpde_run, NO2) {

    const bool infraday_analysis = true; 
    std::string results_str; 
    std::string month; 
    std::string day_chosen; 
    if(infraday_analysis){
        month = "gennaio"; 
        day_chosen = "15"; 
        results_str = "results_infraday";
    } else{
        results_str = "results"; 
    }

    unsigned int num_folds;
    const std::string CV_type = "5-folds";     // "GCV"  "5-folds"
    if(CV_type != "GCV"){
        num_folds = 5; 
    }


    std::string rescale_data = "_sqrt";   // "", "_rescale", oppure inserisci la trasformazione che vuoi 
    const bool normalized_loss_flag = true;  // true to normalize the loss

    std::string pde_type = "tr";  // ""  "tr" 
    const std::string u_string = "1e-1"; 

    const unsigned int num_fpirls_iter = 15;  

    // define time domain 
    double t0;
    double tf;
    if(!infraday_analysis){
        t0 = 0.0;
        tf = 22.0;
    } else{
        t0 = 0.0;
        tf = 23.0;
    }

    const unsigned int M = 6;  // number of time mesh nodes 
    const std::string M_string = "_M" + std::to_string(M);
    Triangulation<1, 1> time_mesh(t0, tf, M-1);  // interval [t0, tf] with M-1 knots

    std::size_t seed = 438172;
    unsigned int MC_run = 100; 
    const std::string model_type_root = "param";  // "nonparam" "param"
    const std::string model_type = model_type_root + M_string; 
    const std::string  cascading_model_type = "nonparametric";  // "parametric" "nonparametric" 
    if(cascading_model_type == "parametric"){
        pde_type = pde_type + "_par"; 
    }

    std::string cov_strategy;
    if(model_type_root == "param"){
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

    const std::string mesh_type = "canotto_fine";  // la run sulla mesh fine!
    const std::string mesh_gcv_type = "canotto_fine";  // fine come la run !! 

    std::string est_type = "mixed";    // mean mixed

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/case_studies/mixed_NO2"; 
 
    std::string path_data;  
    if(!infraday_analysis){
        path_data = path + "/data/space-time";  
    } else{
        path_data = path + "/data_infraday/" + month + "/day_" + day_chosen;  
    }


    std::string solutions_path; std::string solutions_path_gcv; 

    std::string sigla_model; 
    if(est_type == "mean"){
        sigla_model = "STRPDE"; 
    }
    if(est_type == "mixed"){
        sigla_model = "MSTRPDE"; 
    }

    if(!infraday_analysis){
        if(model_type_root == "nonparam"){
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + model_type + "/" + CV_type + "/" + mesh_type;
            solutions_path_gcv = path + "/" + results_str + "/" + sigla_model + "/" + model_type + "/" + CV_type + "/" + mesh_gcv_type;
        } else{
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + model_type + "/" + CV_type + "/" + mesh_type + "/" + covariate_type;
            solutions_path_gcv = path + "/" + results_str + "/" + sigla_model + "/" + model_type + "/" + CV_type + "/" + mesh_gcv_type + "/" + covariate_type;
        }
        if(pde_type != ""){
            solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string; 
            solutions_path_gcv = solutions_path_gcv + "/pde_" + pde_type + "/u_" + u_string; 
        }
            
    } else{
        if(model_type_root == "nonparam"){
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + CV_type + "/" + mesh_type;
            solutions_path_gcv = path + "/" + results_str + "/" + sigla_model + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + CV_type + "/" + mesh_gcv_type;
        } else{
            solutions_path = path + "/" + results_str + "/" + sigla_model + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + CV_type + "/" + mesh_type + "/" + covariate_type;
            solutions_path_gcv = path + "/" + results_str + "/" + sigla_model + "/" + month + "/day_" + day_chosen + "/" + model_type + "/" + CV_type + "/" + mesh_gcv_type + "/" + covariate_type;
        }
        if(pde_type != ""){
            solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string;
            solutions_path_gcv = solutions_path_gcv + "/pde_" + pde_type + "/u_" + u_string;
        }
            
    }

    std::cout << "solution path: " << solutions_path << std::endl; 

    // define spatial domain
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> time_locs; 
    DMatrix<double> X; DMatrix<double> Z; DVector<unsigned int> ids_groups; 

    y = read_csv<double>(path_data + "/y" + rescale_data + ".csv"); 
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;

    time_locs = read_csv<double>(path_data + "/time_locs.csv");  
    std::cout << "dim time locs = " << time_locs.rows() << ";" << time_locs.cols() << std::endl;
    std::cout << "max(time_locs) = " << time_locs.maxCoeff() << std::endl; 

    space_locs = read_csv<double>(path_data + "/locs.csv");  
    std::cout << "dim space_locs locs = " << space_locs.rows() << ";" << space_locs.cols() << std::endl;
    std::cout << "max(space_locs) = " << space_locs.maxCoeff() << std::endl;

    if(model_type_root == "param"){
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
    df.stack(OBSERVATIONS_BLK, y);
    if(model_type_root == "param")
        df.insert(DESIGN_MATRIX_BLK, X);
    if(est_type == "mixed")
        df.insert(DESIGN_MATRIX_RANDOM_BLK, Z);
   

    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
    DMatrix<double, Eigen::RowMajor> b_data; 
    DMatrix<double> u; 
    b_data  = read_csv<double>(path_data + "/" + mesh_type + "/b_" + u_string + "_opt_" + cascading_model_type +  covariate_type_for_data + ".csv");
    u  = read_csv<double>(path_data + "/" + mesh_type + "/u_" + u_string + "_opt_" + cascading_model_type  + covariate_type_for_data + ".csv");    
    std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl;
    std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
    DiscretizedVectorField<2, 2> b(b_data);    
    auto Ld = -laplacian<FEM>() + advection<FEM>(b); 
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    

    // // Only Laplacian
    // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1); // rhs 
    // auto Ld = -laplacian<FEM>();   
    // PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    

    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);


    // // Save quadrature nodes 
    // DMatrix<double> quad_nodes = problem.quadrature_nodes();  
    // std::cout << "rows quad nodes = " << quad_nodes.rows() << std::endl; 
    // std::cout << "cols quad nodes = " << quad_nodes.cols() << std::endl; 
    // std::ofstream file_quad(path_data + "/" + mesh_type + "/quadrature_nodes.csv");
    // for(int i = 0; i < quad_nodes.rows(); ++i) {
    //     file_quad << quad_nodes(i, 0) << "," << quad_nodes(i, 1) << "\n";
    // }
    // file_quad.close();

    std::cout << "-----------------------------RUN STARTS------------------------" << std::endl; 

    if(est_type == "mean"){

        STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);

        // set model's data
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);
        model.set_normalize_loss(normalized_loss_flag);
        
        model.set_data(df);

        // read lambdas
        double lambda_D;  
        double lambda_T;  

        std::ifstream fileLambdaS_gcv(solutions_path_gcv + "/lambda_s_opt.csv");
        if(fileLambdaS_gcv.is_open()){
            fileLambdaS_gcv >> lambda_D; 
            fileLambdaS_gcv.close();
        }
        std::ifstream fileLambdaT(solutions_path_gcv + "/lambda_t_opt.csv");
        if(fileLambdaT.is_open()){
            fileLambdaT >> lambda_T; 
            fileLambdaT.close();
        }

        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);

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

        DMatrix<double> computedFn = model.Psi()*model.f(); // ATTENZIONE: la soluzione avrà zeri nelle righe missing ! 
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(solutions_path + "/fn.csv");
        if(filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }     

        if(model_type_root == "param"){
            DMatrix<double> computedBeta = model.beta(); 
            const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filebeta(solutions_path + "/beta.csv");
            if(filebeta.is_open()){
                filebeta << computedBeta.format(CSVFormatBeta);
                filebeta.close();
            }
        }



    }

    if(est_type == "mixed"){
                            
        MSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise);    
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);
        model.set_normalize_loss(normalized_loss_flag);
        
        // set model 
        model.set_data(df);
        model.set_ids_groups(ids_groups); 
        model.set_fpirls_max_iter(num_fpirls_iter); 

        // read lambdas
        double lambda_D;  
        double lambda_T;  

        std::ifstream fileLambdaS_gcv(solutions_path_gcv + "/lambda_s_opt.csv");
        if(fileLambdaS_gcv.is_open()){
            fileLambdaS_gcv >> lambda_D; 
            fileLambdaS_gcv.close();
        }
        std::ifstream fileLambdaT(solutions_path_gcv + "/lambda_t_opt.csv");
        if(fileLambdaT.is_open()){
            fileLambdaT >> lambda_T; 
            fileLambdaT.close();
        }
        std::cout << "lambda_D = " << lambda_D << std::endl;
        std::cout << "lambda_T = " << lambda_T << std::endl;

        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);


        model.init();
        model.solve();


        // Save solution
        DMatrix<double> computedF = model.f();
        // std::cout << "computedF max = " << (computedF.rowwise().lpNorm<1>()).maxCoeff() << std::endl; 
        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filef(solutions_path + "/f.csv");
        if(filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
        }

        DMatrix<double> computedFn = model.Psi()*model.f();   // ATTENZIONE: la soluzione avrà zeri nelle righe missing ! 
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(solutions_path + "/fn.csv");
        if(filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }

        if(model_type_root == "param"){
            DMatrix<double> computedBeta = model.beta(); 
            const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filebeta(solutions_path + "/beta.csv");
            if(filebeta.is_open()){
                filebeta << computedBeta.format(CSVFormatBeta);
                filebeta.close();
            }
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



    }

}
        




