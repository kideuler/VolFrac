#include "TestFramework.hpp"
#include "LatexTable.hpp"

const double pi = 3.14159265358979323846;
using namespace std;

int main(int argc, char** argv) {

    TorchWrapper model("../../models/VolFrac.pt", "../../models/normalization.pt");

    int shape = 0;
    if (argc > 1) {
        shape = atoi(argv[1]);
    }
    double exacts[3] = {pi*0.02, 0.0075*pi, 7*pi/900.0};
    vector<string> shape_names = {"Ellipse", "Flower", "Petals"};

    BBox box{0.3, 0.7, 0.3, 0.7};
    Grid grid = CreateGrid(box, 5, 5, shape, 5000);
    double exact = exacts[shape];
    grid.model = &model;
    vector<int> sizes = {32,64,128,256,512,1024};
    vector<string> sizes_str;
    for (int size : sizes) {
        sizes_str.push_back(to_string(size));
    }
    vector<string> headers = {"Sizes","PIB 3", "PIB 10", "OscCircle", "AI"};
    vector<vector<double>> data;
    double result;
    for (int i = 0; i < sizes.size(); i++) {
        vector<double> row;
        grid.ResetBox(box, sizes[i], sizes[i]);
        row.push_back(sizes[i]);

        // pib 3
        grid.ComputeVolumeFractions(3);
        result = 100.0*fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // pib 10
        grid.ComputeVolumeFractions(10);
        result = 100.0*fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // circle method
        grid.ComputeVolumeFractionsCurv();
        result = 100.0*fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        // AI method
        grid.ComputeVolumeFractionsAI();
        result = 100.0*fabs(grid.ComputeTotalVolume() - exact) / exact;
        row.push_back(result);
        grid.ZeroVolumeFractions();

        data.push_back(row);

        // print the row
        cout << row[0] << " " << row[1] << " " << row[2] << " " << row[3] << " " << row[4] << endl;
    }

    WriteLatexTable("AccuracyTable_"+shape_names[shape]+".tex", headers, data, "Accuracy of Volume Fraction Methods on "+shape_names[shape]+" in terms of percent error of total volume");
    


    return 0;
}