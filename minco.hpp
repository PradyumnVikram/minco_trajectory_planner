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

#ifndef MINCO_HPP
#define MINCO_HPP

#include "trajectory.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <vector>

namespace minco
{

    // The banded system class is used for solving
    // banded linear system Ax=b efficiently.
    // A is an N*N band matrix with lower band width lowerBw
    // and upper band width upperBw.
    // Banded LU factorization has O(N) time complexity.
    class BandedSystem
    {
    public:
        // The size of A, as well as the lower/upper
        // banded width p/q are needed
        inline void create(const int &n, const int &p, const int &q)
        {
            // In case ofso re-creating before destroying
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
            return;
        }

        inline void destroy()
        {
            if (ptrData != nullptr)
            {
                delete[] ptrData;
                ptrData = nullptr;
            }
            return;
        }

    private:
        int N;
        int lowerBw;
        int upperBw;
        // Compulsory nullptr initialization here
        double *ptrData = nullptr;

    public:
        // Reset the matrix to zero
        inline void reset(void)
        {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
            return;
        }

        // The band matrix is stored as suggested in "Matrix Computation"
        inline const double &operator()(const int &i, const int &j) const
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline double &operator()(const int &i, const int &j)
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        // This function conducts banded LU factorization in place
        // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
        inline void factorizeLU()
        {
            int iM, jM;
            double cVl;
            for (int k = 0; k <= N - 2; k++)
            {
                iM = std::min(k + lowerBw, N - 1);
                cVl = operator()(k, k);
                for (int i = k + 1; i <= iM; i++)
                {
                    if (operator()(i, k) != 0.0)
                    {
                        operator()(i, k) /= cVl;
                    }
                }
                jM = std::min(k + upperBw, N - 1);
                for (int j = k + 1; j <= jM; j++)
                {
                    cVl = operator()(k, j);
                    if (cVl != 0.0)
                    {
                        for (int i = k + 1; i <= iM; i++)
                        {
                            if (operator()(i, k) != 0.0)
                            {
                                operator()(i, j) -= operator()(i, k) * cVl;
                            }
                        }
                    }
                }
            }
            return;
        }

        // This function solves Ax=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solve(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; j++)
            {
                iM = std::min(j + lowerBw, N - 1);
                for (int i = j + 1; i <= iM; i++)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; j--)
            {
                b.row(j) /= operator()(j, j);
                iM = std::max(0, j - upperBw);
                for (int i = iM; i <= j - 1; i++)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            return;
        }

        // This function solves ATx=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solveAdj(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; j++)
            {
                b.row(j) /= operator()(j, j);
                iM = std::min(j + upperBw, N - 1);
                for (int i = j + 1; i <= iM; i++)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; j--)
            {
                iM = std::max(0, j - lowerBw);
                for (int i = iM; i <= j - 1; i++)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            return;
        }
    };

    // MINCO for s=2 and non-uniform time
    class MINCO_S2NU
    {
    public:
        MINCO_S2NU() = default;
        ~MINCO_S2NU() { A.destroy(); }

    private:
        int N;
        Eigen::Matrix<double, 3, 2> headPV;
        Eigen::Matrix<double, 3, 2> tailPV;
        BandedSystem A;
        Eigen::MatrixX3d b;
        Eigen::VectorXd T1;
        Eigen::VectorXd T2;
        Eigen::VectorXd T3;

    public:
        inline void setConditions(const Eigen::Matrix3d &headState,
                                  const Eigen::Matrix3d &tailState,
                                  const int &pieceNum)
        {
            N = pieceNum;
            headPV = headState.leftCols<2>();
            tailPV = tailState.leftCols<2>();
            A.create(4 * N, 4, 4);
            b.resize(4 * N, 3);
            T1.resize(N);
            T2.resize(N);
            T3.resize(N);
            return;
        }

        inline void setParameters(const Eigen::Matrix3Xd &inPs,
                                  const Eigen::VectorXd &ts)
        {
            T1 = ts;
            T2 = T1.cwiseProduct(T1);
            T3 = T2.cwiseProduct(T1);

            A.reset();
            b.setZero();

            A(0, 0) = 1.0;
            A(1, 1) = 1.0;
            b.row(0) = headPV.col(0).transpose();
            b.row(1) = headPV.col(1).transpose();

            for (int i = 0; i < N - 1; i++)
            {
                A(4 * i + 2, 4 * i + 2) = 2.0;
                A(4 * i + 2, 4 * i + 3) = 6.0 * T1(i);
                A(4 * i + 2, 4 * i + 6) = -2.0;
                A(4 * i + 3, 4 * i) = 1.0;
                A(4 * i + 3, 4 * i + 1) = T1(i);
                A(4 * i + 3, 4 * i + 2) = T2(i);
                A(4 * i + 3, 4 * i + 3) = T3(i);
                A(4 * i + 4, 4 * i) = 1.0;
                A(4 * i + 4, 4 * i + 1) = T1(i);
                A(4 * i + 4, 4 * i + 2) = T2(i);
                A(4 * i + 4, 4 * i + 3) = T3(i);
                A(4 * i + 4, 4 * i + 4) = -1.0;
                A(4 * i + 5, 4 * i + 1) = 1.0;
                A(4 * i + 5, 4 * i + 2) = 2.0 * T1(i);
                A(4 * i + 5, 4 * i + 3) = 3.0 * T2(i);
                A(4 * i + 5, 4 * i + 5) = -1.0;

                b.row(4 * i + 3) = inPs.col(i).transpose();
            }

            A(4 * N - 2, 4 * N - 4) = 1.0;
            A(4 * N - 2, 4 * N - 3) = T1(N - 1);
            A(4 * N - 2, 4 * N - 2) = T2(N - 1);
            A(4 * N - 2, 4 * N - 1) = T3(N - 1);
            A(4 * N - 1, 4 * N - 3) = 1.0;
            A(4 * N - 1, 4 * N - 2) = 2 * T1(N - 1);
            A(4 * N - 1, 4 * N - 1) = 3 * T2(N - 1);

            b.row(4 * N - 2) = tailPV.col(0).transpose();
            b.row(4 * N - 1) = tailPV.col(1).transpose();

            A.factorizeLU();
            A.solve(b);

            return;
        }

        inline void getTrajectory(Trajectory<3> &traj) const
        {
            traj.clear();
            traj.reserve(N);
            for (int i = 0; i < N; i++)
            {
                traj.emplace_back(T1(i),
                                  b.block<4, 3>(4 * i, 0)
                                      .transpose()
                                      .rowwise()
                                      .reverse());
            }
            return;
        }

        inline void getEnergy(double &energy) const
        {
            energy = 0.0;
            for (int i = 0; i < N; i++)
            {
                energy += 4.0 * b.row(4 * i + 2).squaredNorm() * T1(i) +
                          12.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) * T2(i) +
                          12.0 * b.row(4 * i + 3).squaredNorm() * T3(i);
            }
            return;
        }

        inline const Eigen::MatrixX3d &getCoeffs(void) const
        {
            return b;
        }

        inline void getEnergyPartialGradByCoeffs(Eigen::MatrixX3d &gdC) const
        {
            gdC.resize(4 * N, 3);
            for (int i = 0; i < N; i++)
            {
                gdC.row(4 * i + 3) = 12.0 * b.row(4 * i + 2) * T2(i) +
                                     24.0 * b.row(4 * i + 3) * T3(i);
                gdC.row(4 * i + 2) = 8.0 * b.row(4 * i + 2) * T1(i) +
                                     12.0 * b.row(4 * i + 3) * T2(i);
                gdC.block<2, 3>(4 * i, 0).setZero();
            }
            return;
        }

        inline void getEnergyPartialGradByTimes(Eigen::VectorXd &gdT) const
        {
            gdT.resize(N);
            for (int i = 0; i < N; i++)
            {
                gdT(i) = 4.0 * b.row(4 * i + 2).squaredNorm() +
                         24.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) * T1(i) +
                         36.0 * b.row(4 * i + 3).squaredNorm() * T2(i);
            }
            return;
        }

        inline void propogateGrad(const Eigen::MatrixX3d &partialGradByCoeffs,
                                  const Eigen::VectorXd &partialGradByTimes,
                                  Eigen::Matrix3Xd &gradByPoints,
                                  Eigen::VectorXd &gradByTimes)

        {
            gradByPoints.resize(3, N - 1);
            gradByTimes.resize(N);
            Eigen::MatrixX3d adjGrad = partialGradByCoeffs;
            A.solveAdj(adjGrad);

            for (int i = 0; i < N - 1; i++)
            {
                gradByPoints.col(i) = adjGrad.row(4 * i + 3).transpose();
            }

            Eigen::Matrix<double, 4, 3> B1;
            Eigen::Matrix<double, 2, 3> B2;
            for (int i = 0; i < N - 1; i++)
            {
                // negative jerk
                B1.row(0) = -6.0 * b.row(i * 4 + 3);

                // negative velocity
                B1.row(1) = -(b.row(i * 4 + 1) +
                              2.0 * T1(i) * b.row(i * 4 + 2) +
                              3.0 * T2(i) * b.row(i * 4 + 3));
                B1.row(2) = B1.row(1);

                // negative acceleration
                B1.row(3) = -(2.0 * b.row(i * 4 + 2) +
                              6.0 * T1(i) * b.row(i * 4 + 3));

                gradByTimes(i) = B1.cwiseProduct(adjGrad.block<4, 3>(4 * i + 2, 0)).sum();
            }

            // negative velocity
            B2.row(0) = -(b.row(4 * N - 3) +
                          2.0 * T1(N - 1) * b.row(4 * N - 2) +
                          3.0 * T2(N - 1) * b.row(4 * N - 1));

            // negative acceleration
            B2.row(1) = -(2.0 * b.row(4 * N - 2) +
                          6.0 * T1(N - 1) * b.row(4 * N - 1));

            gradByTimes(N - 1) = B2.cwiseProduct(adjGrad.block<2, 3>(4 * N - 2, 0)).sum();

            gradByTimes += partialGradByTimes;
        }
    };


}
#endif
