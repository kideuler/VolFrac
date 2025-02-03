#include "Grid.hpp"
#include "CircleVolFrac.hpp"
#include "gtest/gtest.h"

TEST(ComputeVolFracCircleTest, ComputeCircleBoxIntersection) {
    
    BBox box{-0.01, 1.01, -0.01, 1.01};
    Grid grid(box, 50, 50);

    vertex C{0.5, 0.5};
    double r = 0.5;

    const double exact = M_PI*r*r;
    
    for (int i = 0; i<grid.cells.size(); i++) {
        cell cell = grid.cells[i];
        double x_min = grid.points[cell.indices[0]][0];
        double x_max = grid.points[cell.indices[1]][0];
        double y_min = grid.points[cell.indices[0]][1];
        double y_max = grid.points[cell.indices[2]][1];
        double area = ComputeCircleBoxIntersection(C, r, x_min, x_max, y_min, y_max);
        grid.cells[i].volfrac = area / (cell.volume);
    }

    double total_volume = grid.ComputeTotalVolume();
    ASSERT_NEAR(total_volume, exact, 1e-14);
}