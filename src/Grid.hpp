#ifndef GRID_HPP
#define GRID_HPP

#include "IntervalTree.hpp"
#include "KDTree.hpp"
#include "CircleVolFrac.hpp"

#ifdef USE_TORCH
#include "TorchWrapper.hpp"
#endif

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
    vector<int> point_indices;
    bool crosses_boundary = false;
};

class Grid {
    public:
        Grid(){}; // Default constructor

        Grid(BBox box, int nx, int ny); // Constructor

        void AddShape(const IntervalTree<Axis::Y> &bdy); // Add a shape to the grid

        void AddTree(const KDTree<5> &tree); // Add a kdtre if point data to the grid

        void ComputeVolumeFractions(); // Compute the volume fractions of the cells using zero order method

        void ComputeVolumeFractions(int npaxis); // Compute the volume fractions of the cells

        void ComputeVolumeFractionsCurv(); // Compute the volume fractions of the cells using circle method

#ifdef USE_TORCH
        void ComputeVolumeFractionsAI(); // Compute the volume fractions of the cells using Neural Network model
#endif
        double ComputeTotalVolume(); // Compute the total volume of the grid

        void ZeroVolumeFractions(); // Set all volume fractions to zero

        void ResetBox(BBox box, int nx, int ny); // Reset the box and grid size

    private:
        BBox box;
        int nx;
        int ny;
        double dx;
        double dy;
        int ncellsx;
        int ncellsy;
        int first_cell_index = 0;
        vector<IntervalTree<Axis::Y>> shapes;
        vector<KDTree<5>> kd_trees;
        vector<bool> inflags;

    public:
        vector<vertex> points;
        vector<cell> cells;
#ifdef USE_TORCH
        TorchWrapper *model = nullptr;
#endif
};

#endif