#include "Grid.hpp"

Grid::Grid(BBox box, int nx, int ny){
    this->box = box;
    this->nx = nx;
    this->ny = ny;
    this->ncellsx = nx-1;
    this->ncellsy = ny-1;
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

void Grid::AddShape(std::unique_ptr<IntervalTree<Axis::Y>> bdy){
    shapes.push_back(std::move(*bdy));

    int npoints = nx*ny;
    inflags.resize(npoints, false);

    // determine which points are inside the shapes
    for (int i = 0; i < npoints; i++) {
        inflags[i] = shapes[0].QueryPoint(points[i]) % 2 == 1;
    }

    for (int i = 0; i < shapes[0].seg_ids.size(); i++) {
        int v1 = shapes[0].seg_ids[i][0];
        int v2 = shapes[0].seg_ids[i][1];

        vertex P1 = shapes[0].coordinates[v1];
        vertex P2 = shapes[0].coordinates[v2];

        // find cell index that contains P1 and P2
        int i1 = int((P1[0] - box.x_min) / dx);
        int j1 = int((P1[1] - box.y_min) / dy);
        int c1 = i1 + j1*(nx-1);
        int i2 = int((P2[0] - box.x_min) / dx);
        int j2 = int((P2[1] - box.y_min) / dy);
        int c2 = i2 + j2*(nx-1);

        cells[c1].crosses_boundary = true;
        cells[c2].crosses_boundary = true;

        // get all cells that the line between P1 and P2 crosses
        int x1 = i1, y1 = j1;
        int x2 = i2, y2 = j2;
        int dx_line = abs(x2 - x1);
        int dy_line = abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx_line - dy_line;

        int x = x1, y = y1;
        while (true) {
            int cell_index = x + y*(nx-1);
            cells[cell_index].crosses_boundary = true;
            if (x == x2 && y == y2)
                break;
            int e2 = 2 * err;
            if (e2 > -dy_line) {
                err -= dy_line;
                x += sx;
            }
            if (e2 < dx_line) {
                err += dx_line;
                y += sy;
            }
        }
    }

    // loop through all cells and add adjacent cells to cross boundary true
    vector<bool> visited(cells.size(), false);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].crosses_boundary) {
            int x = i % (nx-1);
            int y = i / (nx-1);
            if (x > 0) {
                visited[i-1] = true;
            }
            if (x < nx-1) {
                visited[i+1] = true;
            }
            if (y > 0) {
                visited[i-(nx-1)] = true;
            }
            if (y < ny-1) {
                visited[i+(nx-1)] = true;
            }
        }
    }

    for (int i = 0; i < cells.size(); i++) {
        if (visited[i]) {
            cells[i].crosses_boundary = true;
        }
    }
}

void Grid::AddTree(const KDTree<5> &tree) {
    kd_trees.push_back(tree);
}

void Grid::ComputeVolumeFractions(){
    if (shapes.size() == 0 || points.size() == 0 || cells.size() == 0) {
        return;
    }

    int count = 0;
    for (int i = 0; i < cells.size(); i++) {
        count = 0;
        for (int j = 0; j < 4; j++) {
            if (inflags[cells[i].indices[j]]) {
                count++;
            }
        }

        if (count == 4) {
            cells[i].volfrac = 1.0;
        } else if (count == 0) {
            cells[i].volfrac = 0.0;
        } else {
            cells[i].volfrac = 0.5;
        }
    }
}

void Grid::ComputeVolumeFractions(int npaxis){
    if (shapes.size() == 0 || points.size() == 0 || cells.size() == 0) {
        return;
    }

    // compute the volume fractions
    double dx_in = dx / (npaxis-1);
    double dy_in = dy / (npaxis-1);
    for (int i = 0; i < cells.size(); i++) {

        if (!cells[i].crosses_boundary) {
            int count = 0;
            for (int j = 0; j < 4; j++) {
                if (inflags[cells[i].indices[j]]) {
                    count++;
                }
            }
            cells[i].volfrac = double(count) / 4.0;
            continue;
        }
        
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

void Grid::ComputeVolumeFractionsCurv(){
    if (points.size() == 0 || cells.size() == 0 || kd_trees.size() == 0) {
        return;
    }

    for (int i = 0; i < cells.size(); i++) {
        if (!cells[i].crosses_boundary) {
            int count = 0;
            for (int j = 0; j < 4; j++) {
                if (inflags[cells[i].indices[j]]) {
                    count++;
                }
            }
            cells[i].volfrac = double(count) / 4.0;
            continue;
        }

        cell cell = cells[i];
        double x_min = points[cell.indices[0]][0];
        double x_max = points[cell.indices[1]][0];
        double y_min = points[cell.indices[0]][1];
        double y_max = points[cell.indices[2]][1];

        vertex cell_center{(x_min + x_max) / 2, (y_min + y_max) / 2};
        vertex P; 
        double data[5];
        kd_trees[0].Search(cell_center, P, data); // data constains derivative and curvature information
        double R = fabs(1.0/data[4]);
        vertex C{(P[0]-R*data[1]), (P[1]+R*data[0])};

        double area = ComputeCircleBoxIntersection(C, R, x_min, x_max, y_min, y_max);

        double volfrac = area/cell.volume;
        cells[i].volfrac = area/cell.volume;
    }
}

#ifdef USE_TORCH
void Grid::ComputeVolumeFractionsAI(){
    if (points.size() == 0 || cells.size() == 0 || kd_trees.size() == 0 || model == nullptr) {
        return;
    }

    for (int i = 0; i < cells.size(); i++) {
        if (!cells[i].crosses_boundary) {
            int count = 0;
            for (int j = 0; j < 4; j++) {
                if (inflags[cells[i].indices[j]]) {
                    count++;
                }
            }
            cells[i].volfrac = double(count) / 4.0;
            continue;
        }
        cell cell = cells[i];
        double x_min = points[cell.indices[0]][0];
        double x_max = points[cell.indices[1]][0];
        double y_min = points[cell.indices[0]][1];
        double y_max = points[cell.indices[2]][1];

        double dx = x_max - x_min;
        double dy = y_max - y_min;

        vertex cell_center{(x_min + x_max) / 2, (y_min + y_max) / 2};
        vertex P; 
        double data[5];
        kd_trees[0].Search(cell_center, P, data); // data constains derivative and curvature information
        P[0] = (P[0] - x_min) / dx;
        P[1] = (P[1] - y_min) / dy;
        double dx_dt = data[0]/dx;
        double dy_dt = data[1]/dy;
        double dxx_dt = data[2]/dx;
        double dyy_dt = data[3]/dy;
        double K = (dx_dt*dyy_dt - dy_dt*dxx_dt) / pow(dx_dt*dx_dt + dy_dt*dy_dt, 1.5);
        double norm = sqrt(dx_dt*dx_dt + dy_dt*dy_dt);
        dx_dt /= norm;
        dy_dt /= norm;

        // First check if circle and box intersect at all
        double R = fabs(1/K);
        vertex normal{-dy_dt, dx_dt};
        vertex C{(P[0]+R*normal[0]), (P[1]+R*normal[1])};

        //double volfrac = ComputeCircleBoxIntersection(C, R, 0.0, 1.0, 0.0, 1.0);
        double volfrac = model->Predict(P[0], P[1], -dy_dt, dx_dt, fabs(K));
        if (K<0){
            volfrac = 1.0 - volfrac;
        }
        cells[i].volfrac = volfrac;
    }
}
#endif

double Grid::ComputeTotalVolume(){
    double total_volume = 0.0;
    for (int i = 0; i < cells.size(); i++) {
        total_volume += cells[i].volume * cells[i].volfrac;
    }
    return total_volume;
}

void Grid::ZeroVolumeFractions(){
    for (int i = 0; i < cells.size(); i++) {
        cells[i].volfrac = 0.0;
    }
}

void Grid::ResetBox(BBox box, int nx, int ny){
    points.clear();
    cells.clear();
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

    if (shapes.size() > 0) {
        int npoints = nx*ny;
        inflags.resize(npoints, false);

        // determine which points are inside the shapes
        for (int i = 0; i < npoints; i++) {
            inflags[i] = shapes[0].QueryPoint(points[i]) % 2 == 1;
        }

        for (int i = 0; i < shapes[0].seg_ids.size(); i++) {
            int v1 = shapes[0].seg_ids[i][0];
            int v2 = shapes[0].seg_ids[i][1];

            vertex P1 = shapes[0].coordinates[v1];
            vertex P2 = shapes[0].coordinates[v2];

            // find cell index that contains P1 and P2
            int i1 = int((P1[0] - box.x_min) / dx);
            int j1 = int((P1[1] - box.y_min) / dy);
            int c1 = i1 + j1*(nx-1);
            int i2 = int((P2[0] - box.x_min) / dx);
            int j2 = int((P2[1] - box.y_min) / dy);
            int c2 = i2 + j2*(nx-1);

            cells[c1].crosses_boundary = true;
            cells[c2].crosses_boundary = true;

            // get all cells that the line between P1 and P2 crosses
            int x1 = i1, y1 = j1;
            int x2 = i2, y2 = j2;
            int dx_line = abs(x2 - x1);
            int dy_line = abs(y2 - y1);
            int sx = (x1 < x2) ? 1 : -1;
            int sy = (y1 < y2) ? 1 : -1;
            int err = dx_line - dy_line;

            int x = x1, y = y1;
            while (true) {
                int cell_index = x + y*(nx-1);
                cells[cell_index].crosses_boundary = true;
                if (x == x2 && y == y2)
                    break;
                int e2 = 2 * err;
                if (e2 > -dy_line) {
                    err -= dy_line;
                    x += sx;
                }
                if (e2 < dx_line) {
                    err += dx_line;
                    y += sy;
                }
            }
        }

        // loop through all cells and add adjacent cells to cross boundary true
        vector<bool> visited(cells.size(), false);
        for (int i = 0; i < cells.size(); i++) {
            if (cells[i].crosses_boundary) {
                int x = i % (nx-1);
                int y = i / (nx-1);
                if (x > 0) {
                    visited[i-1] = true;
                }
                if (x < nx-2) {
                    visited[i+1] = true;
                }
                if (y > 0) {
                    visited[i-(nx-1)] = true;
                }
                if (y < ny-2) {
                    visited[i+(nx-1)] = true;
                }
            }
        }

        for (int i = 0; i < cells.size(); i++) {
            if (visited[i]) {
                cells[i].crosses_boundary = true;
            }
        }
    }
}