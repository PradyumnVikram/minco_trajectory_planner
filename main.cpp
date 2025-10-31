//g++ -I//home/azidozide/cpp_libraries/eigen-3.4.1 -std=c++17 main.cpp -o main && "/home/azidozide/projects/MINCO/source_minco/"main
#include <iostream>
#include "minco.hpp"
#include <Eigen/Dense>

int main()
{
    using namespace minco;

    const int N = 3;

    // HEAD and TAIL states for s = 3 (position, velocity, acceleration)
    //  - column 0 = position (x,y,z)
    //  - column 1 = velocity (x,y,z)
    //  - column 2 = acceleration (x,y,z)
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.setZero();
    tailPVA.setZero();

    //start at origin, stop at (3,0,0) with zero vel/acc at both ends
    headPVA.col(0) = Eigen::Vector3d(0.0, 0.0, 0.0); // pos
    headPVA.col(1) = Eigen::Vector3d(0.0, 0.0, 0.0); // vel
    headPVA.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0); // acc

    tailPVA.col(0) = Eigen::Vector3d(3.0, 0.0, 6.0); // pos
    tailPVA.col(1) = Eigen::Vector3d(0.0, 0.0, 0.0); // vel
    tailPVA.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0); // acc

    // Intermediate sparse points q (3 x (N-1)). Provide exactly N-1 columns.
    Eigen::Matrix3Xd inPs(3, N - 1);
    inPs.col(0) = Eigen::Vector3d(0.0, 1.0, 0.0); // waypoint between seg0 & seg1
    inPs.col(1) = Eigen::Vector3d(2.0, 0.0, 0.0); // waypoint between seg1 & seg2

    // Segment durations: one duration per segment (length N)
    Eigen::VectorXd ts(N);
    ts << 1.0, 2.0, 1.0; // example durations for segments 0..2

    // Instantiate and configure MINCO for s=3
    MINCO_S3NU minco;
    minco.setConditions(headPVA, tailPVA, N); 
    minco.setParameters(inPs, ts);

    // Retrieve coefficients matrix: each segment occupies 6 rows (a0..a5), 3 columns (x,y,z)
    const Eigen::MatrixX3d &coeffs = minco.getCoeffs();
    const int coeffsPerSeg = 6; // s=3 -> 6 coefficients per segment (degree 5 polynomial)
    for (int seg = 0; seg < N; ++seg)
    {
        std::cout << "Segment " << seg << " coefficients (rows = a0..a5; columns = x,y,z):\n";
        std::cout << coeffs.block(coeffsPerSeg * seg, 0, coeffsPerSeg, 3) << "\n\n";
    }

    return 0;
}