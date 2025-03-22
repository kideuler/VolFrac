#include "Grid.hpp"
#include <random>
#include <fstream>
#include <iomanip> 

int main(int argc, char** argv) {

    int nruns = 200000;
    if (argc > 1){
        nruns = atoi(argv[1]);
    }

    std::cout << "--- Beginning the generation of "<< nruns << " points of training data ---" << std::endl;

    // create file to store the data
    std::fstream fid;
    fid.open("VolFracData_Plane.dat", std::ios::out);

    int ncirc = 10000;

    for (int n = 0; n<nruns; n++){
        // random point on unit square
        double x = (double)rand() / RAND_MAX;
        x = 2.0*x - 0.5;
        double y = (double)rand() / RAND_MAX;
        y = 2.0*y - 0.5;

        // random normal vector
        double nx = 2.0*((double)rand() / RAND_MAX)-1.0;
        double ny = 2.0*((double)rand() / RAND_MAX)-1.0;
        double nrm = sqrt(nx*nx + ny*ny);
        nx /= nrm;
        ny /= nrm;

        // compute intersection area with unit square
        double planes[1][3] = {
            {nx, ny, -(nx*x + ny*y)}
        };
        double area = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 1);

        if (area < 0.0 || area > 1.0){
            continue;
        }

        // print the data
        fid << std::fixed << std::setprecision(10)
            << x << ", " << y << ", " << nx << ", " << ny << ", " << area << std::endl;
    }

    std::cout << "--- Finished the generation of training data ---" << std::endl;
    fid.close();
    return 0;
}