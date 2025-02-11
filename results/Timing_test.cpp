#include "TestFramework.hpp"
#include "LatexTable.hpp"
#include "chrono"

const double pi = 3.14159265358979323846;
using namespace std;

int main(int argc, char** argv){

#ifdef USE_TORCH
    TorchWrapper model("../../models/VolFrac.pt", "../../models/normalization.pt");
#else
    std::cout << "Torch not enabled" << std::endl;
#endif

    int np = 100;
    if (argc > 1){
        np = atoi(argv[1]);
    }
    
    BBox box{0.3, 0.7, 0.3, 0.7};
    Grid grid = CreateGrid(box, 5, 5, 0, np);
#ifdef USE_TORCH
    grid.model = &model;
#endif
    vector<int> sizes = {32,64,128,256,512,1024,2048,4096};
    vector<string> headers = {"Sizes","PIB 3", "PIB 5", "PIB 10", "OscCircle", "AI"};
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

        // pib 5
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractions(5);
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
#ifdef USE_TORCH
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractionsAI();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();
#endif
        data.push_back(row);

        std::cout << row[0] << " " << row[1] << " " << row[2] << " " << row[3] << " " << row[4] << " " << row[5] << std::endl;
    }

    // np as string
    string np_str = to_string(np);

    WriteLatexTable("Timing_table_"+np_str+".tex", headers, data,"Timing data for volume fraction methods on ellipse in seconds. The ellipse is discretized with "+np_str+" points.");

    return 0;
}