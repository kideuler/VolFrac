#include "Grid.hpp"
#include <random>
#include <fstream>
#include <iomanip> 

int main(int argc, char** argv) {

    int nruns = 20000;
    if (argc > 1){
        nruns = atoi(argv[1]);
    }

    std::cout << "--- Beginning the generation of "<< nruns << " points of training data ---" << std::endl;

    // create file to store the data
    std::fstream fid;
    fid.open("VolFracData.dat", std::ios::out);

    for (int n = 0; n<nruns; n++){
        // random point on unit square
        double x = (double)rand() / RAND_MAX;
        //x = 3.0*x - 1.0;
        double y = (double)rand() / RAND_MAX;
        //y = 3.0*y - 1.0;

        // random normal vector
        double nx = 2.0*((double)rand() / RAND_MAX)-1.0;
        double ny = 2.0*((double)rand() / RAND_MAX)-1.0;
        double nrm = sqrt(nx*nx + ny*ny);
        nx /= nrm;
        ny /= nrm;

        // random curvature on a log scale
        double K = 0;
        switch (n % 6) {
            case 0:
                K = 0.0001;
                break;
            case 1:
                K = 0.001;
                break;
            case 2:
                K = 0.01;
                break;
            case 3:
                K = 0.001;
                break;
            case 4:
                K = 0.0001;
                break;
            case 5:
                K = 0.00001;
        }
        K *= 5*(double)rand() / RAND_MAX;
        

        // compute intersection area with unit square
        double R = fabs(1/K);
        vertex C{(x+R*nx),(y+R*ny)};

        double area = ComputeCircleBoxIntersection(C, R, 0.0, 1.0, 0.0, 1.0);

        if (area < 0.0 || area > 1.0){
            continue;
        }

        // print the data
        fid << std::fixed << std::setprecision(10)
            << x << ", " << y << ", " << nx << ", " << ny << ", " 
            << K << ", " << area << std::endl;
    }

    std::cout << "--- Finished the generation of training data ---" << std::endl;
    fid.close();
    return 0;
}