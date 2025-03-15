#include "Bezier.hpp"
#include "Grid.hpp"
#include <iomanip>
#include <random>   // For modern C++ random number generation
#include <chrono>   // For seeding with current time

using namespace std;
const double pi = 3.14159265358979323846;

int main(int argc, char** argv) {

    int np = 1024; // grid size
    int nbez = 50; // number of random points from bezier curve
    if (argc > 1) {
        np = atoi(argv[1]);
    }
    if (argc > 2) {
        nbez = atoi(argv[2]);
    }

    std::cout << "--- Beginning the generation of training data from "<< np <<" size grid and " << nbez << " random points making a bezier curve ---" << std::endl;

    // Setup modern random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);  // Mersenne Twister PRNG
    std::uniform_real_distribution<double> distribution(0.0, 0.5); // Range [0.0, 0.5] for radius

    // Create random points around circle to form the bezier curve
    vector<array<double,2>> Control_points;
    for (int n = 0; n < nbez; n++){
        double t = 2*pi*double(n)/double(nbez);

        double r = distribution(generator);

        double x = r*cos(t)+0.5;
        double y = r*sin(t)+0.5;

        Control_points.push_back({x,y});
    }
    Control_points.push_back(Control_points[0]); // add first point to complete loop

    BezierCurve bezier(Control_points);


    BBox box{0.0, 1.0, 0.0, 1.0};
    Grid grid(box, np, np);
    coords coordinates;
    KDTree<5> tree;
    segment_ids segments;

    int nsegs = 10000;
    double h = 1.0/double(np-1);
    for (int i = 0; i < nsegs; i++) {
        double t = double(i)/double(nsegs);
        
        array<double,2> P = bezier.evaluate(t);
        array<double,2> D = bezier.derivative(t);
        array<double,2> DD = bezier.secondDerivative(t);
        double K = (D[0]*DD[1] - D[1]*DD[0]) / pow(D[0]*D[0] + D[1]*D[1], 1.5);
        
        double arr[5] = {D[0], D[1], DD[0], DD[1], K};

        tree.Insert(P, arr);
        coordinates.push_back(P);
        segments.push_back({i, (i + 1) % nsegs});
    }

    IntervalTree<Axis::Y> shape(segments, coordinates);
    grid.AddShape(std::make_unique<IntervalTree<Axis::Y>>(shape));
    grid.AddTree(tree);

    grid.ComputeVolumeFractionsTraining("VolFracData.dat");
}