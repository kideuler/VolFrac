#ifndef GRID_HPP
#define GRID_HPP

#include "IntervalTree.hpp"
#include "KDTree.hpp"
#include "CircleVolFrac.hpp"
#include "TorchWrapper.hpp"

using namespace std;

struct BBox {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
};

struct cell {
    array<int,4> indices;
    double volume;
    double volfrac;
};

class Grid {
    public:
        Grid(){}; // Default constructor

        Grid(BBox box, int nx, int ny); // Constructor

        void AddShape(const IntervalTree<Axis::Y> &bdy); // Add a shape to the grid

        void AddTree(const KDTree<5> &tree); // Add a kdtre if point data to the grid

        void ComputeVolumeFractions(int npaxis); // Compute the volume fractions of the cells

        void ComputeVolumeFractionsCurv(); // Compute the volume fractions of the cells using circle method

        double ComputeTotalVolume(); // Compute the total volume of the grid

    private:
        BBox box;
        int nx;
        int ny;
        double dx;
        double dy;
        vector<IntervalTree<Axis::Y>> shapes;
        vector<KDTree<5>> kd_trees;
        vector<bool> inflags;

    public:
        vector<vertex> points;
        vector<cell> cells;

};

#endif