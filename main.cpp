//#g++ -I/home/azidozide/cpp_libraries/eigen-3.4.1/ main.cpp -o main && "/home/azidozide/projects/MINCO/source_minco/"main
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include<chrono>

#include <fstream>

#include "geo_utils.hpp"
#include "gcopter.hpp"
#include "minco.hpp"
#include "lbfgs.hpp"

using namespace std;

// Helper: construct an axis-aligned box as a 6x4 H-matrix.
// Row format: [nx, ny, nz, d] representing nx*x + ny*y + nz*z + d <= 0 for interior.
Eigen::Matrix<double, 6, 4> boxToH(const Eigen::Vector3d &center, const Eigen::Vector3d &half)
{
    Eigen::Matrix<double, 6, 4> H;
    // +x plane: x <= cx + ex  ->  1*x + 0*y + 0*z + (-(cx+ex)) <= 0
    H.row(0) << 1.0, 0.0, 0.0, -(center(0) + half(0));
    // -x plane: x >= cx - ex  ->  -1*x + 0*y + 0*z + (cx - ex) <= 0
    H.row(1) << -1.0, 0.0, 0.0, center(0) - half(0);

    // +y
    H.row(2) << 0.0, 1.0, 0.0, -(center(1) + half(1));
    // -y
    H.row(3) << 0.0, -1.0, 0.0, center(1) - half(1);

    // +z
    H.row(4) << 0.0, 0.0, 1.0, -(center(2) + half(2));
    // -z
    H.row(5) << 0.0, 0.0, -1.0, center(2) - half(2);

    return H;
}

double* return_pos(double time, const Eigen::Matrix<double, 3, 6> &coeff_matrix){
    double* pos_coord = (double* )malloc(3*sizeof(double));
    for(int i=0; i<3; i++){
    pos_coord[i] = coeff_matrix(i, 0) + time*coeff_matrix(i, 1) + time*time*coeff_matrix(i, 2) + time*time*time*coeff_matrix(i, 3) + time*time*time*time*coeff_matrix(i, 4) + time*time*time*time*time*coeff_matrix(i, 5);
    }
    return pos_coord;
}

double* return_vel(double time, const Eigen::Matrix<double, 3, 6> &coeff_matrix){
    double* vel_coord = (double* )malloc(3*sizeof(double));
    for(int i=0; i<3; i++){
    vel_coord[i] = coeff_matrix(i, 1) + 2*time*coeff_matrix(i, 2) + 3*time*time*coeff_matrix(i, 3) + 4*time*time*time*coeff_matrix(i, 4) + 5*time*time*time*time*coeff_matrix(i, 5);
    }
    return vel_coord;
}

double* return_acc(double time, const Eigen::Matrix<double, 3, 6> &coeff_matrix){
    double* acc_coord = (double* )malloc(3*sizeof(double));
    for(int i=0; i<3; i++){
    acc_coord[i] = 2*coeff_matrix(i, 2) + 6*time*coeff_matrix(i, 3) + 12*time*time*coeff_matrix(i, 4) + 20*time*time*time*coeff_matrix(i, 5);
    }
    return acc_coord;
}

int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    // boundaries (P, V, A)
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.setZero();
    tailPVA.setZero();
    headPVA.col(0) = Eigen::Vector3d(0.0, 0.0, 0.0);   // start pos
    tailPVA.col(0) = Eigen::Vector3d(8.0, 0.0, 0.0);   // end pos

    // build a chain of overlapping corridor boxes along straight line
    // Use the exact PolyhedraH type expected by GCOPTER:
    std::vector<Eigen::MatrixX4d> safeCorridor;
    const int numBoxes = 6;
    for (int i = 0; i < numBoxes; ++i) {
        double alpha = double(i) / double(max(1, numBoxes - 1));
        Eigen::Vector3d center = (1.0 - alpha) * headPVA.col(0) + alpha * tailPVA.col(0);
        Eigen::Vector3d half(1.5, 2.0, 1.0); // corridor width/height (tune)
        Eigen::Matrix<double,6,4> H = boxToH(center, half);
        // Eigen::Matrix<double,6,4> converts to Eigen::MatrixX4d when pushed
        safeCorridor.push_back(H);
    }

    // tuning parameters for GCOPTER
    double timeWeight = 5;
    double lengthPerPiece = 0.5;     // piece length control (tune)
    double smoothingFactor = 0.2;    // mu for smoothed L1
    int integralResolution = 6;      // samples per segment for penalties (higher -> stronger enforcement)

    Eigen::VectorXd magnitudeBounds(5); // [v_max, omg_max, theta_max, thrust_min, thrust_max]
    magnitudeBounds << 4.0, 10.0, 0.5, -7.0, 7.0;

    Eigen::VectorXd penaltyWeights(5);  // [pos_w, vel_w, omg_w, theta_w, thrust_w]
    penaltyWeights << 300.0, 1.0, 0.5, 0.5, 0.1;

    Eigen::VectorXd physicalParams(6); // flatness parameters: [mass, g, drag_hor, drag_ver, parasitic, speed_smooth]
    physicalParams << 0.5, 9.81, 0.0, 0.0, 0.0, 1.0;

    // Create GCOPTER SFC object and setup
    gcopter::GCOPTER_PolytopeSFC sfc;
    bool ok = sfc.setup(timeWeight,
                        headPVA,
                        tailPVA,
                        safeCorridor,
                        lengthPerPiece,
                        smoothingFactor,
                        integralResolution,
                        magnitudeBounds,
                        penaltyWeights,
                        physicalParams);
    if (!ok) {
        cerr << "SFC setup failed (processCorridor may be unable to enumerate vertices). Exiting.\n";
        return 1;
    }

    // Run the optimizer (internally uses lbfgs and attachPenaltyFunctional)
    Trajectory<5> traj;
    double relCostTol = 1e-4;
    double cost = sfc.optimize(traj, relCostTol);

    cout << "SFC optimization finished, cost = " << cost << "\n";

    // Use the returned Trajectory to get durations, junctions and coefficients
    int pieceNum = traj.getPieceNum();
    if (pieceNum == 0) {
        cerr << "Optimization returned empty trajectory.\n";
        return 1;
    }

    // Optimized times (per-segment durations)
    Eigen::VectorXd opt_times = traj.getDurations();
    cout << "Optimized times:\n" << opt_times.transpose() << "\n";

    // Junction positions: 3 x (pieceNum + 1) [start, junc1, junc2, ..., end]
    Eigen::Matrix3Xd positions = traj.getPositions();
    cout << "Junction positions (columns: start ... end):\n" << positions << "\n";

    // Inner (intermediate) points: exclude first and last columns
    if (positions.cols() >= 3) {
        Eigen::Matrix3Xd innerPoints = positions.middleCols(1, positions.cols() - 2);
        cout << "Optimized inner points (columns):\n" << innerPoints << "\n";
    } else {
        cout << "No inner points (single segment trajectory).\n";
    }

    fstream fout_csv;
    fout_csv.open("trajectory.csv", ios::out | ios::app);
    double time_stamp = 0.0;
    double duration;
    double increment = 0.001;
    double* position = (double* )malloc(3*sizeof(double));
    double* velocity = (double* )malloc(3*sizeof(double));
    double* acceleration = (double* )malloc(3*sizeof(double));
    // Per-segment coefficient matrices (Piece<D>::getCoeffMat()): 3 x (D+1)
    for (int seg = 0; seg < pieceNum; ++seg) {
        const auto &cMat = traj[seg].getCoeffMat(); // 3 x (D+1)
        cout << "Segment " << seg << " duration: " << traj[seg].getDuration() << "\n";
        cout << "Segment " << seg << " coeffs (cols = a0..a" << cMat.cols()-1 << "):\n" << cMat << "\n\n";
    }

    int seg = 0;
    const auto &cMat = traj[seg].getCoeffMat();
    duration += traj[seg].getDuration();

    while(seg<pieceNum){
        if(time_stamp>duration){
            seg += 1;
            const auto &cMat = traj[seg].getCoeffMat();
            duration += traj[seg].getDuration();
        }
        
        position = return_pos(time_stamp, cMat);
        velocity = return_vel(time_stamp, cMat);
        acceleration = return_acc(time_stamp, cMat);
        fout_csv<<time_stamp<<","<<position[0]<<","<<position[1]<<","<<position[2]<<","<<velocity[0]<<","<<velocity[1]<<","<<velocity[2]<<","<<acceleration[0]<<","<<acceleration[1]<<","<<acceleration[2]<<"\n";
        time_stamp += increment;
    }
    fout_csv.close();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration_.count() << " microseconds" << std::endl;

    return 0;
}