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

    Eigen::MatrixXd readTrajectoryCSVPositions(const std::string& filename)
    {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            throw std::runtime_error("Unable to open file " + filename);
        }

        std::vector<Eigen::Vector3d> positions;
        std::string line;
        while (std::getline(infile, line)) {
            if(line.empty()) continue;
            std::stringstream ss(line);
            std::string val;

            // Read time (ignored)
            std::getline(ss, val, ',');
            // Read positions
            double pos[3];
            for(int i=0; i<3; ++i) {
                if(!std::getline(ss, val, ',')) {
                    // End of line
                    throw std::runtime_error("Malformed CSV line: not enough values");
                }
                pos[i] = std::stod(val);
            }
            // Ignore the rest (velocity, acceleration)
            positions.emplace_back(pos[0], pos[1], pos[2]);
        }
        infile.close();

        // Copy into Eigen matrix
        int N = static_cast<int>(positions.size());
        Eigen::MatrixXd posMat(3, N);
        for(int i=0; i<N; ++i) {
            posMat.col(i) = positions[i];
        }
        return posMat;
    }


    // Helper: construct an axis-aligned box as a 6x4 H-matrix.
    // Row format: [nx, ny, nz, d] representing nx*x + ny*y + d <= 0 for interior.
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

double* return_pos(double time, const Eigen::Matrix<double, 3, 4> &coeff_matrix){
    double* pos_coord = (double*)malloc(3*sizeof(double));
    for (int i = 0; i < 3; i++) {
        // pos(t) = c0 + t*c1 + t^2*c2 + t^3*c3
        pos_coord[i] = coeff_matrix(i, 3)
                     + time * coeff_matrix(i, 2)
                     + time*time * coeff_matrix(i, 1)
                     + time*time*time * coeff_matrix(i, 0);
    }
    return pos_coord;
}

double* return_vel(double time, const Eigen::Matrix<double, 3, 4> &coeff_matrix){
    double* vel_coord = (double*)malloc(3*sizeof(double));
    for (int i = 0; i < 3; i++) {
        // derivative: v(t) = c1 + 2 t c2 + 3 t^2 c3
        vel_coord[i] = coeff_matrix(i, 2)
                     + 2*time * coeff_matrix(i, 1)
                     + 3*time*time * coeff_matrix(i, 0);
    }
    return vel_coord;
}

double* return_acc(double time, const Eigen::Matrix<double, 3, 4> &coeff_matrix){
    double* acc_coord = (double*)malloc(3*sizeof(double));
    for (int i = 0; i < 3; i++) {
        // acceleration: a(t) = 2 c2 + 6 t c3
        acc_coord[i] = 2*coeff_matrix(i, 1)
                     + 6*time * coeff_matrix(i, 0);
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
        tailPVA.col(0) = Eigen::Vector3d(8.0, 1.0, 0.0);   // end pos
        Eigen::Vector3d shift(0.0, 1.5, 0.0);

        // build a chain of overlapping corridor boxes along straight line
        // Use the exact PolyhedraH type expected by GCOPTER:
        std::vector<Eigen::MatrixX4d> safeCorridor;
        const int numBoxes = 24;
        Eigen::Vector3d center;
        double amplitude = 3; // Amplitude of the wave (adjust as needed)
        double wavelength = 8.0; // Distance over which one wave cycle occurs

        for (int i = 0; i < numBoxes; ++i) {
            double alpha = double(i) / double(std::max(8, numBoxes - 1));
            double x = alpha * 8.0; // x-coordinate along the path
            double y = amplitude * sin(2.0 * M_PI * x / wavelength); // y-coordinate oscillates

            center = Eigen::Vector3d(x, y, 0.0); // z remains 0 for a 2D wave

            Eigen::Vector3d half(1.5, 2.0, 1.0); // corridor width/height (tune)
            Eigen::Matrix<double,6,4> H = boxToH(center, half);
            //std::cout << half << " " << center << " done" << std::endl;
            safeCorridor.push_back(H);
        }

        // tuning parameters for GCOPTER
        double timeWeight = 5;
        double lengthPerPiece = 0.5;     // piece length control (tune)
        double smoothingFactor = 0.2;    // mu for smoothed L1
        int integralResolution = 6;      // samples per segment for penalties (higher -> stronger enforcement)

        Eigen::VectorXd magnitudeBounds(6); // [v_max, omg_max, theta_max, thrust_min, thrust_max]
        magnitudeBounds << 1.0, 10.0, 0.5, -2.0, 2.0, 6.0;

        Eigen::VectorXd penaltyWeights(6);  // [pos_w, vel_w, omg_w, theta_w, thrust_w]
        penaltyWeights << 0.5, 0.5, 0.5, 0.5, 0.1, 1.0;

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
            std::cerr << "SFC setup failed (processCorridor may be unable to enumerate vertices). Exiting.\n";
            return 1;
        }
        Eigen::MatrixXd otherPos = readTrajectoryCSVPositions("trajectory_extra.csv");
        std::vector<Trajectory<3>> otherTrajs(1);  // Single other agent trajectory
        std::vector<double> sampleTimes;
        std::vector<Eigen::Matrix<double,3,4>> sampleCoeffs;

        double dt = 0.001;
        for(int i=0; i<otherPos.cols()-1; i++) {
            sampleTimes.push_back(dt);
            Eigen::Matrix<double,3,4> pieceCoeff; pieceCoeff.setZero();
            pieceCoeff.col(3) = otherPos.col(i);  // Constant position piece
            sampleCoeffs.push_back(pieceCoeff);
        }

        otherTrajs[0] = Trajectory<3>(sampleTimes, sampleCoeffs);
        double C_sw = 1.5; // example safe separation
        Eigen::Matrix3d ellipsoid = Eigen::Matrix3d::Identity(); // can tune per axis
        sfc.setSwarmObstacleParams(otherTrajs, C_sw, ellipsoid);
        // Run the optimizer (internally uses lbfgs and attachPenaltyFunctional)
        Trajectory<3> traj;
        double relCostTol = 1e-4;
        double cost = sfc.optimize(traj, relCostTol);

        //cout << "SFC optimization finished, cost = " << cost << "\n";

        // Use the returned Trajectory to get durations, junctions and coefficients
        int pieceNum = traj.getPieceNum();
        if (pieceNum == 0) {
            std::cerr << "Optimization returned empty trajectory.\n";
            return 1;
        }

        // Optimized times (per-segment durations)
        Eigen::VectorXd opt_times = traj.getDurations();
        //cout << "Optimized times:\n" << opt_times.transpose() << "\n";

        // Junction positions: 3 x (pieceNum + 1) [start, junc1, junc2, ..., end]
        Eigen::Matrix3Xd positions = traj.getPositions();
        //cout << "Junction positions (columns: start ... end):\n" << positions << "\n";

        // Inner (intermediate) points: exclude first and last columns
        if (positions.cols() >= 3) {
            Eigen::Matrix3Xd innerPoints = positions.middleCols(1, positions.cols() - 2);
            //cout << "Optimized inner points (columns):\n" << innerPoints << "\n";
        }

        auto end_time_csv = std::chrono::high_resolution_clock::now();
        //CSV
        auto duration_csv = std::chrono::duration_cast<std::chrono::microseconds>(end_time_csv - start_time);
        std::cout << "Execution time: " << duration_csv.count() << " microseconds" << std::endl;

        std::fstream fout_csv;
        fout_csv.open("trajectory.csv", std::ios::out | std::ios::app);
        double time_stamp = 0.0;
        double duration;
        double accumulated_stamps = 0.0;
        double prev_time_stamp = 0.0;
        double increment = 0.001;
        double* position = (double* )malloc(3*sizeof(double));
        double* velocity = (double* )malloc(3*sizeof(double));
        double* acceleration = (double* )malloc(3*sizeof(double));
        // Per-segment coefficient matrices (Piece<D>::getCoeffMat()): 3 x (D+1)

        int seg = 0;
        const auto* cMat = &traj[seg].getCoeffMat(); // pointer to current segment's coeff mat

        duration = traj[seg].getDuration();

        while (seg < pieceNum) {
            if (time_stamp > duration) {
                seg += 1;
                if (seg == pieceNum) { break; }
                cMat = &traj[seg].getCoeffMat();
                prev_time_stamp = accumulated_stamps;
                accumulated_stamps += time_stamp - increment - prev_time_stamp;
                duration += traj[seg].getDuration();
            }
            double t_rel = time_stamp - accumulated_stamps;

            position = return_pos(t_rel, *cMat);
            velocity = return_vel(t_rel, *cMat);
            acceleration = return_acc(t_rel, *cMat);

            fout_csv << time_stamp << ","
                    << position[0] << "," << position[1] << "," << position[2] << ","
                    << velocity[0] << "," << velocity[1] << "," << velocity[2] << ","
                    << acceleration[0] << "," << acceleration[1] << "," << acceleration[2] << "\n";

            time_stamp += increment;
        }

        fout_csv.close();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Execution time: " << duration_.count() << " microseconds" << std::endl;

        return 0;
    }