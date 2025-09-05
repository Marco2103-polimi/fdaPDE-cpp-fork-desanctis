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


#include <fdaPDE/fdapde.h>

#include <iostream>
#include <string>
#include <vector>

using namespace fdapde;

int test_02();

int main(){
    test_02();  
    return 0; 
}

// test 1
//    mesh:         c-shaped
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
int test_02() {

    // geometry 
    using PointT = Eigen::Matrix<double, 2, 1>;

    Eigen::Matrix<double, Dynamic, Dynamic> points = read_csv<double>("my_data/mesh/c_shaped_242/points.csv").as_matrix();
    Eigen::Matrix<int, Dynamic, Dynamic> elements = read_csv<int>("my_data/mesh/c_shaped_242/elements.csv").as_matrix();
    Eigen::Matrix<int, Dynamic, Dynamic> boundary = read_csv<int>("my_data/mesh/c_shaped_242/boundary.csv").as_matrix();

    elements.array() -= 1; // non necessario

    std::string datadir = "my_data/msr/02/";
    Triangulation<2, 2> D(points, elements, boundary);

    GeoFrame data(D);

    // data 
    auto& l = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    l.load_csv<double>(datadir + "response.csv");
    l.load_csv<double>(datadir + "design_matrix.csv");

    // physics 
    FeSpace Vh(D, P1<1>);   // functional space definition

    // trial and test function definition
    TrialFunction f(Vh);
    TestFunction v(Vh);

    auto a = integral(D)(dot(grad(f), grad(v)));

    // homogeneous forcing linear form
    ZeroField<2> u;
    auto F = integral(D)(u * v);

    // modeling
    MSRPDE m("y ~ x1 + x2 + f", data, fe_ls_elliptic(a, F));  
    // M: mi serve che gia' nel constructor venga letto group_ids_ altrimenti 
    //    se uso il setter dopo il constructor, ha gia' istanziato matrici dei pesi vuote e' da assert error

    // Eigen::Matrix<unsigned int, Dynamic, 1> ids_groups = read_csv<unsigned int>(datadir + "ids_groups.csv").as_matrix();
    // std::cout << "in test main: ids_groups.size() = " << ids_groups.size() << std::endl;
    // m.set_ids_groups(ids_groups);

    // // calibration
    // std::vector<double> lambda_grid = {1e-4, 1e-3, 1e-2, 1e-1};
    // GridSearch<1> opt;
    // opt.optimize(m.gcv(), lambda_grid);

    // // fit at optimal smoothing level
    // m.fit(opt.optimum());

    double lambda = read_csv<double>(datadir + "lambda.csv", false, false).as_matrix()(0,0);
    std::cout << "lambda read from file = " << lambda << std::endl;
    m.fit(lambda);
    write_csv(datadir + "f.csv", m.f());

    return 0;
}
