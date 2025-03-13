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
    shapes.push_back(std::move(bdy));

    int npoints = nx*ny;
    inflags.resize(npoints, false);

    // determine which points are inside the shapes
    for (int i = 0; i < npoints; i++) {
        inflags[i] = shapes[0]->QueryPoint(points[i]) % 2 == 1;
    }

    for (size_t i = 0; i < shapes[0]->seg_ids.size(); i++) {
        int v1 = shapes[0]->seg_ids[i][0];
        int v2 = shapes[0]->seg_ids[i][1];

        vertex P1 = shapes[0]->coordinates[v1];
        vertex P2 = shapes[0]->coordinates[v2];

        // find cell index that contains P1 and P2
        int i1 = int((P1[0] - box.x_min) / dx);
        int j1 = int((P1[1] - box.y_min) / dy);
        int c1 = i1 + j1*(nx-1);
        int i2 = int((P2[0] - box.x_min) / dx);
        int j2 = int((P2[1] - box.y_min) / dy);
        int c2 = i2 + j2*(nx-1);

        cells[c1].loc_type = 1;
        cells[c2].loc_type = 1;

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
            cells[cell_index].loc_type = 1;
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
    for (int n = 0; n < 2; n++) {
        vector<bool> visited(cells.size(), false);
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].loc_type == 1) {
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

        for (size_t i = 0; i < cells.size(); i++) {
            if (visited[i]) {
                cells[i].loc_type = 1;
            }
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
    for (size_t i = 0; i < cells.size(); i++) {
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
    for (size_t i = 0; i < cells.size(); i++) {

        if (!(cells[i].loc_type == 1)) {
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
                if (shapes[0]->QueryPoint(P) % 2 == 1) {
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

    for (size_t i = 0; i < cells.size(); i++) {
        if (!(cells[i].loc_type == 1)) {
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
        vertex N;
        double data[5];
        kd_trees[0].Search(cell_center, P, data); // data constains derivative and curvature information
        double R = fabs(1.0/data[4]);
        N[0] = -data[1];
        N[1] = data[0];
        if (data[4] < 0.0) {
            N[0] = -N[0];
            N[1] = -N[1];
        }
        vertex C{(P[0]+R*N[0]), (P[1]+R*N[1])};

        double area = ComputeCircleBoxIntersection(C, R, x_min, x_max, y_min, y_max);

        double volfrac = area/cells[i].volume;
        if (volfrac > 1.0){
            volfrac = 1.0;
        }
        if (data[4] < 0.0) {
            volfrac = 1.0 - volfrac;
        }
        cells[i].volfrac = volfrac;
    }
}

#ifdef USE_TORCH
void Grid::ComputeVolumeFractionsAI(){
    if (points.size() == 0 || cells.size() == 0 || kd_trees.size() == 0 || model == nullptr) {
        return;
    }

    for (int i = 0; i < cells.size(); i++) {
        if (!(cells[i].loc_type == 1)) {
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
    for (size_t i = 0; i < cells.size(); i++) {
        total_volume += cells[i].volume * cells[i].volfrac;
    }
    return total_volume;
}

void Grid::ZeroVolumeFractions(){
    for (size_t i = 0; i < cells.size(); i++) {
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
            inflags[i] = shapes[0]->QueryPoint(points[i]) % 2 == 1;
        }

        for (size_t i = 0; i < shapes[0]->seg_ids.size(); i++) {
            int v1 = shapes[0]->seg_ids[i][0];
            int v2 = shapes[0]->seg_ids[i][1];

            vertex P1 = shapes[0]->coordinates[v1];
            vertex P2 = shapes[0]->coordinates[v2];

            // find cell index that contains P1 and P2
            int i1 = int((P1[0] - box.x_min) / dx);
            int j1 = int((P1[1] - box.y_min) / dy);
            int c1 = i1 + j1*(nx-1);
            int i2 = int((P2[0] - box.x_min) / dx);
            int j2 = int((P2[1] - box.y_min) / dy);
            int c2 = i2 + j2*(nx-1);

            cells[c1].loc_type = 1;
            cells[c2].loc_type = 1;

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
                cells[cell_index].loc_type = 1;
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
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells[i].loc_type == 1) {
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

        for (size_t i = 0; i < cells.size(); i++) {
            if (visited[i]) {
                cells[i].loc_type = 1;
            }
        }
    }
}

void Grid::ExportToVTK(const std::string& filename) {
    // Open file for writing
    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write VTK header
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Grid with volume fractions\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";

    // Write points
    vtkFile << "POINTS " << points.size() << " float\n";
    for (const auto& point : points) {
        vtkFile << point[0] << " " << point[1] << " 0.0\n";
    }

    // Write cells
    vtkFile << "CELLS " << cells.size() << " " << cells.size() * 5 << "\n";
    for (const auto& cell : cells) {
        vtkFile << "4 " << cell.indices[0] << " " << cell.indices[1] << " " 
                << cell.indices[2] << " " << cell.indices[3] << "\n";
    }

    // Write cell types (9 = VTK_QUAD)
    vtkFile << "CELL_TYPES " << cells.size() << "\n";
    for (size_t i = 0; i < cells.size(); i++) {
        vtkFile << "9\n";
    }

    // Write cell data (volume fractions)
    vtkFile << "CELL_DATA " << cells.size() << "\n";
    vtkFile << "SCALARS volume_fraction float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& cell : cells) {
        vtkFile << cell.volfrac << "\n";
    }
    
    // Add location type data
    vtkFile << "SCALARS location_type int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& cell : cells) {
        vtkFile << cell.loc_type << "\n";
    }

    vtkFile.close();
    
    // Write the boundary curves to a separate file if we have shapes
    if (!shapes.empty()) {
        std::string curvesFilename = filename.substr(0, filename.find_last_of('.')) + "_curves.vtk";
        std::ofstream curvesFile(curvesFilename);
        
        if (!curvesFile.is_open()) {
            std::cerr << "Failed to open file: " << curvesFilename << std::endl;
            return;
        }
        
        curvesFile << "# vtk DataFile Version 3.0\n";
        curvesFile << "Boundary curves\n";
        curvesFile << "ASCII\n";
        curvesFile << "DATASET POLYDATA\n";
        
        // Count total number of points in all shapes
        size_t totalPoints = 0;
        for (const auto& shape : shapes) {
            totalPoints += shape->coordinates.size();
        }
        
        // Write points for the boundary
        curvesFile << "POINTS " << totalPoints << " float\n";
        for (const auto& shape : shapes) {
            for (const auto& point : shape->coordinates) {
                curvesFile << point[0] << " " << point[1] << " 0.0\n";
            }
        }
        
        // Count total number of segments
        size_t totalSegments = 0;
        for (const auto& shape : shapes) {
            totalSegments += shape->seg_ids.size();
        }
        
        // Write lines for the boundary
        curvesFile << "LINES " << totalSegments << " " << totalSegments * 3 << "\n";
        size_t pointOffset = 0;
        for (const auto& shape : shapes) {
            for (const auto& seg : shape->seg_ids) {
                curvesFile << "2 " << (seg[0] + pointOffset) << " " << (seg[1] + pointOffset) << "\n";
            }
            pointOffset += shape->coordinates.size();
        }
        
        curvesFile.close();
        std::cout << "Boundary curves exported successfully to: " << curvesFilename << std::endl;
    }
    
    std::cout << "VTK file exported successfully to: " << filename << std::endl;
}