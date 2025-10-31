#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "minco.hpp"
#include "lbfgs.hpp"

using namespace minco;

// Structure passed to L-BFGS evaluate callback
struct OptData {
    MINCO_S3NU *minco;
    int N;                // number of segments
    double time_weight;   // optional weight for penalizing total time (set 0 if unused)
};

// L-BFGS evaluate function: x -> cost, g
// x layout: [log_ts (N)] [inPs_flattened (3*(N-1))]  (column-major flatten of inPs)
static double lbfgs_evaluate(void *instance,
                             const Eigen::VectorXd &x,
                             Eigen::VectorXd &g)
{
    OptData *d = reinterpret_cast<OptData*>(instance);
    int N = d->N;
    const int np = N - 1; // number of intermediate sparse points

    // Map log times and map flattened inPs
    Eigen::Map<const Eigen::VectorXd> log_ts(x.data(), N);
    Eigen::VectorXd ts = log_ts.array().exp(); // enforce positivity: ts > 0

    Eigen::Map<const Eigen::VectorXd> inps_flat(x.data() + N, 3 * np);
    Eigen::Matrix3Xd inPs(3, np);
    // Eigen maps are column-major by default; copy by columns
    for (int c = 0; c < np; ++c) {
        inPs.col(c) = inps_flat.segment(3*c, 3);
    }

    // Set MINCO parameters and evaluate energy + partial gradients
    d->minco->setParameters(inPs, ts);

    double energy = 0.0;
    d->minco->getEnergy(energy);

    Eigen::MatrixX3d partialGradByCoeffs;
    Eigen::VectorXd partialGradByTimes;
    d->minco->getEnergyPartialGradByCoeffs(partialGradByCoeffs);
    d->minco->getEnergyPartialGradByTimes(partialGradByTimes);

    // Propagate gradient from coefficients/times to sparse points and times
    Eigen::Matrix3Xd gradByPoints; // 3 x (N-1)
    Eigen::VectorXd gradByTimes;   // N
    d->minco->propogateGrad(partialGradByCoeffs, partialGradByTimes,
                            gradByPoints, gradByTimes);

    // Optionally add time penalty term: weight * sum(ts)
    double cost = energy;
    if (d->time_weight != 0.0) {
        cost += d->time_weight * ts.sum();
        gradByTimes.array() += d->time_weight;
    }

    // Build gradient g for optimizer consistent with x layout.
    g.resize(x.size());
    g.setZero();

    // Chain rule for log_ts: ts = exp(log_ts)
    // d cost / d log_ts = (d cost / d ts) * (d ts / d log_ts) = gradByTimes * ts
    g.segment(0, N) = gradByTimes.cwiseProduct(ts);

    // Gradients for inPs: flatten gradByPoints in the same column-major order
    for (int c = 0; c < np; ++c) {
        g.segment(N + 3*c, 3) = gradByPoints.col(c);
    }

    return cost;
}

int main()
{
    const int N = 3; // number of segments in your example

    // Boundary states (P, V, A) as you had them
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.setZero();
    tailPVA.setZero();

    headPVA.col(0) = Eigen::Vector3d(0.0, 0.0, 0.0); // pos
    headPVA.col(1) = Eigen::Vector3d(0.0, 0.0, 0.0); // vel
    headPVA.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0); // acc

    tailPVA.col(0) = Eigen::Vector3d(3.0, 0.0, 6.0); // pos
    tailPVA.col(1) = Eigen::Vector3d(0.0, 0.0, 0.0); // vel
    tailPVA.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0); // acc

    // initial intermediate points
    Eigen::Matrix3Xd inPs(3, N - 1);
    inPs.col(0) = Eigen::Vector3d(0.0, 1.0, 0.0);
    inPs.col(1) = Eigen::Vector3d(2.0, 0.0, 0.0);

    // initial times guess (positive)
    Eigen::VectorXd ts(N);
    ts << 1.0, 2.0, 1.0;

    // Instantiate MINCO for s=3 and initialize conditions
    MINCO_S3NU minco;
    minco.setConditions(headPVA, tailPVA, N);

    // Build initial optimizer vector x = [log_ts, inPs_flat]
    Eigen::VectorXd x(N + 3*(N-1));
    Eigen::Map<Eigen::VectorXd> log_ts_map(x.data(), N);
    log_ts_map = ts.array().log();
    for (int c = 0; c < N-1; ++c) {
        x.segment(N + 3*c, 3) = inPs.col(c);
    }

    // Create OptData
    OptData data;
    data.minco = &minco;
    data.N = N;
    data.time_weight = 25.0; // tune if you want to penalize total time

    // L-BFGS parameters
    lbfgs::lbfgs_parameter_t params;
    params.mem_size = 10;
    params.past = 3;
    params.delta = 1e-6;
    params.g_epsilon = 0.0; // use past/delta stopping for potentially nonsmooth cases

    // run optimizer
    double final_cost;
    int ret = lbfgs::lbfgs_optimize(x,
                                    final_cost,
                                    &lbfgs_evaluate,
                                    nullptr,   // no stepbound
                                    nullptr,   // no progress callback
                                    &data,
                                    params);

    std::cout << "lbfgs returned: " << ret << " (" << lbfgs::lbfgs_strerror(ret) << ")\n";
    std::cout << "final cost: " << final_cost << "\n";

    // Recover optimized ts and inPs from x
    Eigen::VectorXd opt_ts = x.segment(0, N).array().exp();
    Eigen::Matrix3Xd opt_inPs(3, N-1);
    for (int c = 0; c < N-1; ++c) {
        opt_inPs.col(c) = x.segment(N + 3*c, 3);
    }

    // Set final parameters and extract coefficients
    minco.setParameters(opt_inPs, opt_ts);
    const Eigen::MatrixX3d &coeffs = minco.getCoeffs();
    const int coeffsPerSeg = 6; // for s=3 (6 coefficients per segment)
    for (int seg = 0; seg < N; ++seg)
    {
        std::cout << "Segment " << seg << " coefficients:\n";
        std::cout << coeffs.block(coeffsPerSeg * seg, 0, coeffsPerSeg, 3) << "\n\n";
    }

    std::cout << "Optimized times:\n" << opt_ts.transpose() << "\n";
    std::cout << "Optimized intermediate points:\n";
    for (int c = 0; c < N-1; ++c)
        std::cout << opt_inPs.col(c).transpose() << "\n";

    return 0;
}