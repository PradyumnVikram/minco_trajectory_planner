/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef GEO_UTILS_HPP
#define GEO_UTILS_HPP

#include "quickhull.hpp"
#include "sdlp.hpp"

#include <Eigen/Eigen>

#include <cfloat>
#include <cstdint>
#include <set>
#include <chrono>

namespace geo_utils
{

    // Each row of hPoly is defined by h0, h1, h2, h3 as
    // h0*x + h1*y + h2*z + h3 <= 0
    inline bool findInterior(const Eigen::MatrixX3d &hPoly,
                             Eigen::Vector2d &interior)
    {
        const int m = hPoly.rows();

        Eigen::MatrixX3d A(m, 3);
        Eigen::VectorXd b(m);
        Eigen::Vector3d c, x;
        const Eigen::ArrayXd hNorm = hPoly.leftCols<2>().rowwise().norm();
        A.leftCols<2>() = hPoly.leftCols<2>().array().colwise() / hNorm;
        A.rightCols<1>().setConstant(1.0);
        b = -hPoly.rightCols<1>().array() / hNorm;
        c.setZero();
        c(2) = -1.0;

        const double minmaxsd = sdlp::linprog<3>(c, A, b, x);
        interior = x.head<2>();

        return minmaxsd < 0.0 && !std::isinf(minmaxsd);
    }

    inline bool overlap(const Eigen::MatrixX3d &hPoly0,
                        const Eigen::MatrixX3d &hPoly1,
                        const double eps = 1.0e-6)

    {
        const int m = hPoly0.rows();
        const int n = hPoly1.rows();
        Eigen::MatrixX3d A(m + n, 3);
        Eigen::Vector3d c, x;
        Eigen::VectorXd b(m + n);
        A.leftCols<2>().topRows(m) = hPoly0.leftCols<2>();
        A.leftCols<2>().bottomRows(n) = hPoly1.leftCols<2>();
        A.rightCols<1>().setConstant(1.0);
        b.topRows(m) = -hPoly0.rightCols<1>();
        b.bottomRows(n) = -hPoly1.rightCols<1>();
        c.setZero();
        c(2) = -1.0;

        const double minmaxsd = sdlp::linprog<3>(c, A, b, x);

        return minmaxsd < -eps && !std::isinf(minmaxsd);
    }

    struct filterLess {
        bool operator()(const Eigen::Vector2d &l,
                        const Eigen::Vector2d &r) const {
            return l(0) < r(0) || (l(0) == r(0) && l(1) < r(1));
        }
    };

    inline void filterVs(const Eigen::Matrix2Xd &rV,
                         const double &epsilon,
                         Eigen::Matrix2Xd &fV)
    {
        const double mag = std::max(fabs(rV.maxCoeff()), fabs(rV.minCoeff()));
        const double res = mag * std::max(fabs(epsilon) / mag, DBL_EPSILON);
        std::set<Eigen::Vector2d, filterLess> filter;
        fV = rV;
        int offset = 0;
        Eigen::Vector2d quanti;
        for (int i = 0; i < rV.cols(); i++)
        {
            quanti = (rV.col(i) / res).array().round();
            if (filter.find(quanti) == filter.end())
            {
                filter.insert(quanti);
                fV.col(offset) = rV.col(i);
                offset++;
            }
        }
        fV = fV.leftCols(offset).eval();
        return;
    }

    // Each row of hPoly is defined by h0, h1, h2, h3 as
    // h0*x + h1*y + h2*z + h3 <= 0
    // proposed epsilon is 1.0e-6
    inline void enumerateVs(const Eigen::MatrixX3d &hPoly,
                        const Eigen::Vector2d &inner,
                        Eigen::Matrix2Xd &vPoly,
                        const double epsilon = 1.0e-6)
{
    Eigen::VectorXd b = -hPoly.rightCols<1>() - hPoly.leftCols<2>() * inner;
    Eigen::Matrix<double, 2, -1, Eigen::ColMajor> A =
        (hPoly.leftCols<2>().array().colwise() / b.array()).transpose();

    quickhull::QuickHull<double> qh;
    const double qhullEps = std::min(epsilon, quickhull::defaultEps<double>());
    auto cvxHull = qh.getConvexHull(A.data(), A.cols(), false, true, qhullEps);
    const auto &idBuffer = cvxHull.getIndexBuffer();

    const int hNum = idBuffer.size();
    Eigen::Matrix2Xd rV(2, hNum);
    for (int i = 0; i < hNum; ++i)
        rV.col(i) = A.col(idBuffer[i]);

    filterVs(rV, epsilon, vPoly);
    vPoly = (vPoly.colwise() + inner).eval();
}

    // Each row of hPoly is defined by h0, h1, h2, h3 as
    // h0*x + h1*y + h2*z + h3 <= 0
    // proposed epsilon is 1.0e-6
    inline bool enumerateVs(const Eigen::MatrixX3d &hPoly,
                            Eigen::Matrix2Xd &vPoly,
                            const double epsilon = 1.0e-6)
    {
        Eigen::Vector2d inner;
        if (findInterior(hPoly, inner))
        {
            enumerateVs(hPoly, inner, vPoly, epsilon);
            return true;
        }
        else
        {
            return false;
        }
    }

} // namespace geo_utils

#endif
