#include "TestFramework.hpp"
#include "LatexTable.hpp"

int column_width = 15;
using namespace std;

int main(){
    int nruns = 100000;

    Model model("model.dat");
    double x1 = 0.0; double x2 = 1.0;

    vector<double> totals = {0.0, 0.0, 0.0, 0.0, 0.0};

    for (int n = 0; n < nruns; n++){
        vector<double> row;

        double y1 = (double)rand() / RAND_MAX;
        double y2 = (double)rand() / RAND_MAX;

        double slope = (y2 - y1)/(x2-x1);
        double intercept = y1 - slope*x1;
        vertex P({0.5, slope*0.5 + intercept});

        double exact_area = 0.5 * (y1 + y2) * (x2 - x1);

        vertex normal({slope, 1});
        double nrm = sqrt(normal[0]*normal[0] + normal[1]*normal[1]);
        normal[0] /= nrm;
        normal[1] /= nrm;

        double k = 1e-5; // approximate curvature

        // Use Plane Clipping method to compute the exact volume fraction
        double planes[1][3] = {{normal[0], normal[1], -P[0]*normal[0] - P[1]*normal[1]}};
        double volfracPlane = PlaneBoxIntersection(0.0, 1.0, 0.0, 1.0, planes, 1);

        // Use Neural Network to compute the volume fraction
        double input[5] = {P[0], P[1], normal[0], normal[1], k};
        double volfracAI = 1.0 - model.Predict(input);
        
        // Use Osculating Circle method to compute the exact volume fraction
        vertex C({P[0] + (1.0/k)*normal[0], P[1] + (1.0/k)*normal[1]});
        double volfracOsc = 1.0-ComputeCircleBoxIntersection(C, 1.0/k, 0.0, 1.0, 0.0, 1.0);

        // Compute PIB10
        int np = 5;
        double T = double(np*np);
        double p = 0.0;
        for (int i = 0; i < np; i++){
            for (int j = 0; j < np; j++){
                double x = double(i) / double(np-1);
                double y = double(j) / double(np-1);
                
                // if point is under the line
                if (y < slope*x + intercept){
                    p += 1.0;
                }
            }
        }
        double volfracPIB3 = p / T;

        // Compute PIB50
        np = 50;
        T = double(np*np);
        p = 0.0;
        for (int i = 0; i < np; i++){
            for (int j = 0; j < np; j++){
                double x = double(i) / double(np-1);
                double y = double(j) / double(np-1);
                
                // if point is under the line
                if (y < slope*x + intercept){
                    p += 1.0;
                }
            }
        }
        double volfracPIB10 = p / T;

        row.push_back(fabs(volfracPIB3 - exact_area) / exact_area );
        row.push_back(fabs(volfracPIB10 - exact_area) / exact_area);
        row.push_back(fabs(volfracPlane - exact_area) / exact_area);
        row.push_back(fabs(volfracOsc - exact_area) / exact_area);
        row.push_back(fabs(volfracAI - exact_area) / exact_area);

        totals[0] += row[0];
        totals[1] += row[1];
        totals[2] += row[2];
        totals[3] += row[3];
        totals[4] += row[4];

    }

    totals[0] /= nruns;
    totals[1] /= nruns;
    totals[2] /= nruns;
    totals[3] /= nruns;
    totals[4] /= nruns;

    vector<string> headers = {"PIB 5", "PIB 50", "Plane Clipping", "OscCircle", "AI"};

    for (const auto& header : headers) {
        cout << setw(column_width) << header << " ";
    }
    cout << endl;

    for (const auto& value : totals) {
        cout << setw(column_width) << value << " ";
    }
    cout << endl;

    // Make the table
    vector<vector<double>> data = {totals};

    std::string filename = "Accuracy_line_test.tex";


    WriteLatexTable(filename, headers, data, "Volume fraction perecnt error for a straight line with curvature approximation of 1e-5.\\label{tab:accuracy_line}");


    return 0;
}