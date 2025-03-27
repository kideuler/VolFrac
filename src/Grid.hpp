#ifndef GRID_HPP
#define GRID_HPP

#include "IntervalTree.hpp"
#include "KDTree.hpp"
#include "CircleVolFrac.hpp"
#include "PolyVolFrac.hpp"
#include "Model.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

/**
 * @struct BBox
 * @brief Represents a 2D bounding box with min/max coordinates
 */
struct BBox {
    double x_min; ///< Minimum x-coordinate of the box
    double x_max; ///< Maximum x-coordinate of the box
    double y_min; ///< Minimum y-coordinate of the box
    double y_max; ///< Maximum y-coordinate of the box
};

/**
 * @struct cell
 * @brief Represents a cell in the computational grid with volume fraction information
 */
struct cell {
    array<int,4> indices;            ///< Indices of the four corner vertices
    double volume;                   ///< Cell volume
    double volfrac = 0.0;            ///< Volume fraction (0.0-1.0)
    int8_t loc_type = 0;             ///< Location type flag (0: outside, 1: boundary, 2: inside)
    double closest_data[5] {0.0, 0.0, 0.0, 0.0, 0.0}; ///< Data from closest point on boundary [distance, normal_x, normal_y, curvature, flag]
    array<double,2> closest_point = {0.0, 0.0}; ///< Coordinates of closest point on boundary
    bool has_discontinuity = false;  ///< Flag indicating if cell contains a discontinuity
    int dc_index = -1;              ///< Index of discontinuity in grid's discontinuities array
};

/**
 * @class Grid
 * @brief A 2D computational grid for calculating volume fractions of shapes
 * 
 * This class provides functionality for computing volume fractions using various methods
 * including zero-order, curvature-based, polygon clipping, and neural network approaches.
 */
class Grid {
    public:
        /**
         * @brief Default constructor
         */
        Grid(){};

        /**
         * @brief Constructs a grid with specified bounding box and resolution
         * 
         * @param box Bounding box specifying domain extents
         * @param nx Number of grid points in x direction
         * @param ny Number of grid points in y direction
         */
        Grid(BBox box, int nx, int ny);

        /**
         * @brief Add a shape to the grid using an interval tree representation
         * 
         * @param bdy Unique pointer to an interval tree representing the shape boundary
         */
        void AddShape(std::unique_ptr<IntervalTree<Axis::Y>> bdy);

        /**
         * @brief Add a KD-tree with point data to the grid
         * 
         * @param tree KD-tree containing point data with additional attributes
         */
        void AddTree(const KDTree<5> &tree);

        /**
         * @brief Compute volume fractions using a zero-order method
         * 
         * Uses point-in-polygon tests at cell corners to determine volume fractions.
         */
        void ComputeVolumeFractions();

        /**
         * @brief Compute volume fractions using a higher-resolution sampling approach
         * 
         * @param npaxis Number of sample points per axis within each cell
         */
        void ComputeVolumeFractions(int npaxis);

        /**
         * @brief Compute volume fractions using curvature-based circle approximation
         * 
         * Approximates the boundary with circles based on curvature data.
         */
        void ComputeVolumeFractionsCurv();

        /**
         * @brief Compute volume fractions using polygon clipping with planes
         * 
         * Uses Sutherland-Hodgman clipping to compute exact intersections.
         */
        void ComputeVolumeFractionsPlane();

        /**
         * @brief Compute volume fractions using a neural network model
         * 
         * Applies a pre-trained neural network to predict volume fractions.
         */
        void ComputeVolumeFractionsAI();

        /**
         * @brief Precompute closest points to shape boundaries for each cell
         * 
         * Stores distance, normal vector, and curvature information.
         */
        void PreComputeClosestPoints();

        /**
         * @brief Generate training data for machine learning models
         * 
         * @param filename Output filename for the generated training data
         */
        void ComputeVolumeFractionsTraining(const std::string &filename);

        /**
         * @brief Calculate total volume of shape within the grid
         * 
         * @return Total volume (area in 2D) occupied by the shape
         */
        double ComputeTotalVolume();

        /**
         * @brief Reset all volume fractions to zero
         */
        void ZeroVolumeFractions();

        /**
         * @brief Reinitialize the grid with a new bounding box and resolution
         * 
         * @param box New bounding box defining the domain
         * @param nx New number of grid points in x direction
         * @param ny New number of grid points in y direction
         */
        void ResetBox(BBox box, int nx, int ny);

        /**
         * @brief Export grid data to VTK format for visualization
         * 
         * @param filename Output filename for the VTK file
         */
        void ExportToVTK(const std::string& filename);

        /**
         * @brief Add a neural network model for volume fraction prediction
         * 
         * @param filename Path to the saved neural network model file
         */
        void addModel(const std::string& filename);

    private:
        BBox box;         ///< Bounding box of the computational domain
        int nx;           ///< Number of grid points in x direction
        int ny;           ///< Number of grid points in y direction
        double dx;        ///< Grid spacing in x direction
        double dy;        ///< Grid spacing in y direction
        int ncellsx;      ///< Number of cells in x direction
        int ncellsy;      ///< Number of cells in y direction
        vector<std::unique_ptr<IntervalTree<Axis::Y>>> shapes; ///< Shape boundaries represented as interval trees
        vector<KDTree<5>> kd_trees; ///< KD-trees for closest point queries
        vector<bool> inflags;      ///< In/out flags for grid points

    public:
        vector<vertex> points; ///< Grid point coordinates
        vector<cell> cells;    ///< Cell data including volume fractions
        std::vector<std::array<double,6>> discontinuities; ///< Discontinuity data for handling sharp features

        Model *model = nullptr; ///< Pointer to neural network model for AI-based predictions

        /**
         * @brief Static flag to control parallel execution
         * 
         * When set to true, OpenMP parallelization is disabled and all operations run serially.
         */
        static bool forceSerialExecution;
};

inline bool Grid::forceSerialExecution = false;

#endif