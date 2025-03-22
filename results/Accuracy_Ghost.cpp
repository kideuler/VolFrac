#include "TestFramework.hpp"
#include "LatexTable.hpp"

int column_width = 15;
using namespace std;

int main() {

    BBox box{-0.7, 1.1, -0.4, 1.8};
    Grid grid = CreateGrid(box, 5, 5, 3, 100000);
    grid.addModel("model.dat");

    vector<int> sizes = {32,64,128,256,512,1024,2048,4096};

    vector<string> sizes_str;
    for (int size : sizes) {
        sizes_str.push_back(to_string(size));
    }

    vector<string> headers = {"Sizes","PIB 10", "PIB 20", "Plane Clipping", "OscCircle", "AI"};
    for (const auto& header : headers) {
        cout << setw(column_width) << header << " ";
    }
    cout << endl;
    vector<vector<double>> data;
    double result;
    double exact = -(-0.5*pi*(0.25 + 1.0/pi) - 0.583333333333338 + 0.04*pi);
    std::cout << "Exact area: " << exact << std::endl;
    for (int i = 0; i < sizes.size(); i++) {
        vector<double> row;
        grid.ResetBox(box, sizes[i], sizes[i]);
        grid.PreComputeClosestPoints();
        row.push_back(sizes[i]);

        // cross
        // grid.ComputeVolumeFractions();
        // result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        // row.push_back(result);
        // grid.ZeroVolumeFractions();

        // pib 10
        grid.ComputeVolumeFractions(10);
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // pib 10
        grid.ComputeVolumeFractions(20);
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // Plane Clipping
        grid.ComputeVolumeFractionsPlane();
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // OscCircle
        grid.ComputeVolumeFractionsCurv();
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // AI
        grid.ComputeVolumeFractionsAI();
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();
        data.push_back(row);

        // print the row
        for (const auto& value : row) {
            cout << setw(column_width) << value << " ";
        }
        cout << endl;
    }

    std::string filename = "AccuracyTable_Ghost.tex";

    WriteLatexTable(filename, headers, data, "Accuracy of Volume Fraction Methods on the ghost geometry in terms of relative error of total volume.\\label{tab:accuracy_ghost}");

    return 0;
}