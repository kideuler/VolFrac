#ifndef GRID_HPP
#define GRID_HPP

#include "IntervalTree.hpp"
#include "KDTree.hpp"
#include "CircleVolFrac.hpp"
#include "Model.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>

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
    double volfrac = 0.0;
    int8_t loc_type = 0;
    double closest_data[5] {0.0, 0.0, 0.0, 0.0, 0.0};
    array<double,2> closest_point = {0.0, 0.0};
};

class Grid {
    public:
        Grid(){}; // Default constructor

        Grid(BBox box, int nx, int ny); // Constructor

        void AddShape(std::unique_ptr<IntervalTree<Axis::Y>> bdy); // Add a shape to the grid

        void AddTree(const KDTree<5> &tree); // Add a kdtre if point data to the grid

        void ComputeVolumeFractions(); // Compute the volume fractions of the cells using zero order method

        void ComputeVolumeFractions(int npaxis); // Compute the volume fractions of the cells

        void ComputeVolumeFractionsCurv(); // Compute the volume fractions of the cells using circle method

        void ComputeVolumeFractionsAI(); // Compute the volume fractions of the cells using Neural Network model

        void PreComputeClosestPoints(); // Precompute the closest points to the shapes

        void ComputeVolumeFractionsTraining(const std::string &filename);

        double ComputeTotalVolume(); // Compute the total volume of the grid

        void ZeroVolumeFractions(); // Set all volume fractions to zero

        void ResetBox(BBox box, int nx, int ny); // Reset the box and grid size

        void ExportToVTK(const std::string& filename);

        void addModel(const std::string& filename);

    private:
        BBox box;
        int nx;
        int ny;
        double dx;
        double dy;
        int ncellsx;
        int ncellsy;
        vector<std::unique_ptr<IntervalTree<Axis::Y>>> shapes;
        vector<KDTree<5>> kd_trees;
        vector<bool> inflags;

    public:
        vector<vertex> points;
        vector<cell> cells;

        Model *model = nullptr;
};

#endif