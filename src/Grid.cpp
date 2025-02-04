#include "Grid.hpp"

Grid::Grid(BBox box, int nx, int ny){
    this->box = box;
    this->nx = nx;
    this->ny = ny;
    this->dx = (box.x_max - box.x_min) / (nx-1);
    this->dy = (box.y_max - box.y_min) / (ny-1);

    // make a list of points
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            points.push_back({box.x_min + i*dx, box.y_min + j*dy});
        }
    }

    // make list of cells
    for (int j = 0; j < ny-1; j++) {
        for (int i = 0; i < nx-1; i++) {
            cell C{{i + j*nx, i+1 + j*nx, i+1 + (j+1)*nx, i + (j+1)*nx}, dx*dy, 0.0};
            cells.push_back(C);
        }
    }
}

void Grid::AddShape(const IntervalTree<Axis::Y> &bdy){
    shapes.push_back(bdy);
}

void Grid::AddTree(const KDTree<5> &tree) {
    kd_trees.push_back(tree);
}

void Grid::ComputeVolumeFractions(int npaxis){
    if (shapes.size() == 0 || points.size() == 0 || cells.size() == 0) {
        return;
    }

    int npoints = nx*ny;
    inflags.resize(npoints, false);

    // determine which points are inside the shapes
    for (int i = 0; i < npoints; i++) {
        inflags[i] = shapes[0].QueryPoint(points[i]) % 2 == 1;
    }

    // compute the volume fractions
    double dx_in = dx / (npaxis-1);
    double dy_in = dy / (npaxis-1);
    for (int i = 0; i < cells.size(); i++) {
        int count = 0;
        for (int j = 0; j < 4; j++) {
            if (inflags[cells[i].indices[j]]) {
                count++;
            }
        }
        
        if (count == 0) {
            cells[i].volfrac = 0.0;
        } else if (count == 4) {
            cells[i].volfrac = 1.0;
        } else {
            double total_points = double(npaxis*npaxis);
            int fine_count = 0;
            for (int j = 0; j < npaxis; j++) {
                for (int k = 0; k < npaxis; k++) {
                    vertex P = {points[cells[i].indices[0]][0] + k*dx_in, points[cells[i].indices[0]][1] + j*dy_in};
                    if (shapes[0].QueryPoint(P) % 2 == 1) {
                        fine_count++;
                    }
                }
            }
            cells[i].volfrac = double(fine_count) / total_points;
        }
    }
}

double Grid::ComputeTotalVolume(){
    double total_volume = 0.0;
    for (int i = 0; i < cells.size(); i++) {
        total_volume += cells[i].volume * cells[i].volfrac;
    }
    return total_volume;
}