#include "TestFramework.hpp"
#include "LatexTable.hpp"

int column_width = 15;
using namespace std;

int main(int argc, char** argv) {

    int shape = 0;
    if (argc > 1) {
        shape = atoi(argv[1]);
    }
    double exacts[3] = {pi*0.02, 0.0075*pi, 7*pi/900.0};
    vector<string> shape_names = {"Ellipse", "Flower", "Petals"};

    BBox box{0.3, 0.7, 0.3, 0.7};
    Grid grid = CreateGrid(box, 5, 5, shape, 200000);
    double exact = exacts[shape];

    grid.addModel("model.dat");

    vector<int> sizes = {32,64,128,256,512,1024,2048,4096};
    vector<string> sizes_str;
    for (int size : sizes) {
        sizes_str.push_back(to_string(size));
    }

    vector<string> headers = {"Sizes","PIB 5", "PIB 50", "Plane Clipping", "OscCircle", "AI"};

    for (const auto& header : headers) {
        cout << setw(column_width) << header << " ";
    }
    cout << endl;
    vector<vector<double>> data;
    double result;
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

        // pib 3
        grid.ComputeVolumeFractions(5);
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // pib 10
        grid.ComputeVolumeFractions(50);
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // plane clipping
        grid.ComputeVolumeFractionsPlane();
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // circle method
        grid.ComputeVolumeFractionsCurv();
        result = fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // AI method
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

    std::string filename = "AccuracyTable_"+shape_names[shape]+".tex";

    WriteLatexTable(filename, headers, data, "Accuracy of Volume Fraction Methods on "+shape_names[shape]+" in terms of percent error of total volume.\\label{tab:accuracy_"+shape_names[shape]+"}");
    


    return 0;
}