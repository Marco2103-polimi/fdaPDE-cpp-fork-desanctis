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

#ifndef __SRPDE_H__
#define __SRPDE_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;

namespace fdapde {
namespace models {

class SRPDE : public RegressionBase<SRPDE, SpaceOnly> {
   private:
    using Base = RegressionBase<SRPDE, SpaceOnly>;
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
   public:
    IMPORT_REGRESSION_SYMBOLS
    using Base::lambda_D;   // smoothing parameter in space
    using Base::n_basis;    // number of spatial basis
    using Base::runtime;    // runtime model status
    using RegularizationType = SpaceOnly;
    using This = SRPDE;
    static constexpr int n_lambda = 1;
    // constructor
    SRPDE() = default;
    SRPDE(const Base::PDE& pde, Sampling s) : Base(pde, s) {};

    void init_model() {
        if (runtime().query(runtime_status::is_lambda_changed)) {
             
            // std::cout << "here 0 in init_model srpde in IF1" << std::endl;
            // if (W().nonZeros() == 0) {
            //     std::cout << "The sparse matrix is empty (no non-zero elements)." << std::endl;
            // } else {
            //     std::cout << "The sparse matrix has non-zero elements." << std::endl;
            // }
            // std::cout << "range W() in init_model srpde:" << (Eigen::SparseMatrix<double>(W())).coeffs().minCoeff() << ";" << (Eigen::SparseMatrix<double>(W())).coeffs().maxCoeff() << std::endl; 
            
            std::cout << "lambda_D() in srpde = " << std::setprecision(16) << lambda_D() << std::endl;
            
            // assemble system matrix for nonparameteric part
            A_ = SparseBlockMatrix<double, 2, 2>(
              -PsiTD() * W() * Psi(), lambda_D() * R1().transpose(),
	      lambda_D() * R1(),      lambda_D() * R0()            );
            invA_.compute(A_);
            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(n_basis(), 0, n_basis(), 1) = lambda_D() * u();
            return;
        }
        if (runtime().query(runtime_status::require_W_update)) {
            std::cout << "here 0 in init_model srpde in IF2" << std::endl;
            if (W().nonZeros() == 0) {
                std::cout << "The sparse matrix is empty (no non-zero elements)." << std::endl;
            } else {
                std::cout << "The sparse matrix has non-zero elements." << std::endl;
            }
            std::cout << "range W() in init_model srpde:" << (Eigen::SparseMatrix<double>(W())).coeffs().minCoeff() << ";" << (Eigen::SparseMatrix<double>(W())).coeffs().maxCoeff() << std::endl; 
            std::cout << "dim W() = " << W().rows() << ";" << W().cols() << std::endl;
            std::cout << "dim Psi() = " << Psi().rows() << ";" << Psi().cols() << std::endl;
            std::cout << "dim PsiTD() = " << PsiTD().rows() << ";" << PsiTD().cols() << std::endl;
            std::cout << "dim A_.block(0, 0) = " << A_.block(0, 0).rows() << ";" << A_.block(0, 0).cols() << std::endl;
                        
            // adjust north-west block of matrix A_ only
            A_.block(0, 0) = -PsiTD() * W() * Psi();
            std::cout << "here 1 in init_model srpde in IF2" << std::endl;
            invA_.compute(A_);
            std::cout << "here 2 in init_model srpde in IF2" << std::endl;
            return;
        }
    }
    void solve() {
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;
        if (!has_covariates()) {   // nonparametric case
            // update rhs of SR-PDE linear system
            b_.block(0, 0, n_basis(), 1) = -PsiTD() * W() * y();
            // solve linear system A_*x = b_
            sol = invA_.solve(b_);
            f_ = sol.head(n_basis());
        } else {   // parametric case
            std::cout << "srpde solve parametric case" << std::endl; 
            // update rhs of SR-PDE linear system
            b_.block(0, 0, n_basis(), 1) = -PsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z
            // matrices U and V for application of woodbury formula
            U_ = DMatrix<double>::Zero(2 * n_basis(), q());
            U_.block(0, 0, n_basis(), q()) = PsiTD() * W() * X();
            V_ = DMatrix<double>::Zero(q(), 2 * n_basis());
            V_.block(0, 0, q(), n_basis()) = X().transpose() * W() * Psi();
            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
            sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
            // store result of smoothing
            f_ = sol.head(n_basis());
            beta_ = invXtWX().solve(X().transpose() * W()) * (y() - Psi() * f_);
        }
        // store PDE misfit
        g_ = sol.tail(n_basis());
        return;
    }
    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }
    // getters
    const SparseBlockMatrix<double, 2, 2>& A() const { return A_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    virtual ~SRPDE() = default;
};

}   // namespace models
}   // namespace fdapde

#endif   // __SRPDE_H__
