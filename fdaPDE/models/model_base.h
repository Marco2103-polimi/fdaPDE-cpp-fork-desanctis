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

#ifndef __MODEL_BASE_H__
#define __MODEL_BASE_H__

#include <fdaPDE/utils.h>
using fdapde::core::BlockFrame;

#include "model_macros.h"
#include "model_traits.h"
#include "model_runtime.h"

namespace fdapde {

// supported resolution strategies
struct monolithic {};
struct iterative {};
struct sequential {};
  
namespace models {

// abstract base interface for any fdaPDE statistical model.
template <typename Model> class ModelBase {
   public:
    // ModelBase() = default;
    ModelBase() {
        std::cout << "calling model base constructor..." << std::endl;
    }
    // full model stack initialization
    void init() {
        if (model().runtime().query(runtime_status::require_penalty_init)) { model().init_regularization(); }
        if (model().runtime().query(runtime_status::require_functional_basis_evaluation)) {
            model().init_sampling(true);   // init \Psi matrix, always force recomputation
        }
	model().analyze_data();    // specific data-dependent initialization requested by Model
	if (model().runtime().query(runtime_status::require_psi_correction)) { model().correct_psi(); }
        model().init_model();
	// clear all dirty bits in blockframe
	for(const auto& BLK : df_.dirty_cols()) df_.clear_dirty_bit(BLK);
    }
    // setters
    void set_data(const BlockFrame<double, int>& df, bool reindex = false) {
        std::cout << "model base set data here 0" << std::endl; 
        df_ = df;
        std::cout << "model base set data here 1" << std::endl; 
        // insert an index row (if not yet present or requested)
        if (!df_.has_block(INDEXES_BLK) || reindex) {
            std::cout << "model base set data here 2" << std::endl; 
            int n = df_.rows();
            std::cout << "model base set data here 3" << std::endl; 
            DMatrix<int> idx(n, 1);
            std::cout << "model base set data here 4" << std::endl; 
            for (int i = 0; i < n; ++i) idx(i, 0) = i;
            std::cout << "model base set data here 5" << std::endl; 
            df_.insert(INDEXES_BLK, idx);
            std::cout << "model base set data here 6" << std::endl; 
        }
	model().runtime().set(runtime_status::require_data_stack_update);
    std::cout << "model base set data here 7" << std::endl; 
    }
    void set_lambda(const DVector<double>& lambda) {   // dynamic sized version of set_lambda provided by upper layers
	model().set_lambda_D(lambda[0]);
	if constexpr(is_space_time<Model>::value) model().set_lambda_T(lambda[1]);
    }
    // getters
    const BlockFrame<double, int>& data() const { 
        std::cout << "calling data() const in model_base" << std::endl; 
        return df_;
    }
    BlockFrame<double, int>& data() { 
        std::cout << "calling data() non-const in model_base" << std::endl; 
        return df_; 
    }   // direct write-access to model's internal data storage
    const DMatrix<int>& idx() const { 
        std::cout << "idx() in modelbase is calling get.." << std::endl; 
        return df_.get<int>(INDEXES_BLK);
    }   // data indices
    int n_locs() const { return model().n_spatial_locs() * model().n_temporal_locs(); }
    DVector<double> lambda(int) const {   // int supposed to be fdapde::Dynamic
        fdapde_assert(!is_empty(model().lambda()));
        return model().lambda();
    }
    // access to model runtime status
    model_runtime_handler& runtime() { return runtime_; }
    const model_runtime_handler& runtime() const { return runtime_; }

    virtual ~ModelBase() = default;
   protected:
    BlockFrame<double, int> df_ {};      // blockframe for data storage
    model_runtime_handler runtime_ {};   // model's runtime status
  
    // getter to underlying model object
    inline Model& model() { return static_cast<Model&>(*this); }
    inline const Model& model() const { return static_cast<const Model&>(*this); }
};

// set boundary conditions on problem's linear system
// BUG: not working - fix needed due to SparseBlockMatrix interface
// template <typename Model> void ModelBase<Model>::set_dirichlet_bc(SpMatrix<double>& A, DMatrix<double>& b) {
//     int n = A.rows() / 2;

//     for (int i = 0; i < n; ++i) {
//         if (pde_->domain().is_on_boundary(i)) {
//             A.row(i) *= 0;          // zero all entries of this row
//             A.coeffRef(i, i) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

//             A.row(i + n) *= 0;
//             A.coeffRef(i + n, i + n) = 1;

//             // boundaryDatum is a pair (nodeID, boundary value)
//             double boundaryDatum = pde_->boundaryData().empty() ? 0 : pde_->boundaryData().at(i)[0];
//             b.coeffRef(i, 0) = boundaryDatum;   // impose boundary value
//             b.coeffRef(i + n, 0) = 0;
//         }
//     }
//     return;
// }

}   // namespace models
}   // namespace fdapde

#endif   // __MODEL_BASE_H__
