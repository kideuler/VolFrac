#include "TestFramework.hpp"
#include "LatexTable.hpp"
#include "chrono"

const double pi = 3.14159265358979323846;
using namespace std;

int main(){

    TorchWrapper model("../../models/VolFrac.pt", "../../models/normalization.pt");
    
    BBox box{0.3, 0.7, 0.3, 0.7};
    Grid grid = CreateGrid(box, 5, 5, 0, 200);
    grid.model = &model;
    vector<int> sizes = {32,64,128,256,512,1024};
    vector<vector<double>> data;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds;
    for (int i = 0; i < sizes.size(); i++){
        vector<double> row;
        grid.ResetBox(box, sizes[i], sizes[i]);
        row.push_back(sizes[i]);

        // pib 3
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractions(3);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();

        // pib 10
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractions(10);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();

        // circle method
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractionsCurv();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();

        // AI method
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractionsAI();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();

        data.push_back(row);

        std::cout << row[0] << " " << row[1] << " " << row[2] << " " << row[3] << " " << row[4] << std::endl;
    }

    return 0;
}