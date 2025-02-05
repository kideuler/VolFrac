#include "Grid.hpp"
#include <random>

int main(int argc, char** argv) {

    int nruns = 1000;
    if (argc > 0){
        nruns = atoi(argv[1]);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    double mean = -4; 
    double stddev = 1; 

    std::lognormal_distribution<double> dist(mean, stddev);

    std::cout << "--- Beginning the generation of "<< nruns << " points of training data ---" << std::endl;

    for (int n = 0; n<nruns; n++){
        // random point on unit square
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        // random normal vector
        double nx = (double)rand() / RAND_MAX;
        double ny = (double)rand() / RAND_MAX;
        double nrm = sqrt(nx*nx + ny*ny);
        nx /= nrm;
        ny /= nrm;

        // random curvature on a log scale
        double K = dist(gen);
        std::cout << x << " " << y << " K=" << K << std::endl;
    }

    std::cout << "--- Finished the generation of training data ---" << std::endl;
    return 0;
}