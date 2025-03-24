#include "TestFramework.hpp"
#include "LatexTable.hpp"
#include "chrono"

int column_width = 15;
using namespace std;

int main(int argc, char** argv){

    int np = 100;
    if (argc > 1){
        np = atoi(argv[1]);
    }
    
    BBox box{0.3, 0.7, 0.3, 0.7};
    Grid grid = CreateGrid(box, 5, 5, 0, np);
    grid.addModel("model.dat");
    // grid.forceSerialExecution = true;

    vector<int> sizes = {32,64,128,256,512,1024,2048,4096};
    vector<string> headers = {"Sizes","PIB 5", "PIB 50", "Plane Clipping", "OscCircle","AI"};
    for (const auto& header : headers) {
        cout << setw(column_width) << header << " ";
    }
    cout << endl;

    vector<vector<double>> data;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds;
    for (int i = 0; i < sizes.size(); i++){
        vector<double> row;
        grid.ResetBox(box, sizes[i], sizes[i]);
        row.push_back(sizes[i]);
        grid.PreComputeClosestPoints();

        // start = std::chrono::high_resolution_clock::now();
        // grid.ComputeVolumeFractions();
        // end = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // seconds = duration.count() / 1000000.0;
        // row.push_back(seconds);
        // grid.ZeroVolumeFractions();

        // pib 10
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractions(5);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();

        // pib 20
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractions(50);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        row.push_back(seconds);
        grid.ZeroVolumeFractions();

        // plane clipping
        start = std::chrono::high_resolution_clock::now();
        grid.ComputeVolumeFractionsPlane();
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

        // print the row
        for (const auto& value : row) {
            cout << setw(column_width) << value << " ";
        }
        cout << endl;
    }

    // np as string
    string np_str = to_string(np);

    WriteLatexTable("Timing_table_"+np_str+".tex", headers, data,"Timing data for volume fraction initialization methods on ellipse in seconds. The ellipse is discretized with "+np_str+" points.\\label{tab:timing_"+np_str+"}");

    return 0;
}