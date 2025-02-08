#ifndef __LATEX_TABLE_HPP__
#define __LATEX_TABLE_HPP__

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

void WriteLatexTable(string filename, vector<string> col_headers, vector<vector<double>> data, string title) {
    ofstream fid;
    fid.open(filename);

    // Begin table environment
    fid << "\\begin{table}[h!]" << endl;
    fid << "\\centering" << endl;
    
    // Add title if provided
    if (title != "") {
        fid << "\\caption{" << title << "}" << endl;
    }

    // Begin tabular environment
    fid << "\\begin{tabular}{|";
    for (size_t i = 0; i < col_headers.size(); i++) {
        fid << "c|";
    }
    fid << "}" << endl;
    fid << "\\hline" << endl;

    // Write column headers
    for (size_t i = 0; i < col_headers.size(); i++) {
        fid << col_headers[i];
        if (i < col_headers.size() - 1) fid << " & ";
    }
    fid << " \\\\" << endl;
    fid << "\\hline" << endl;

    // Write data rows
    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); j++) {
            fid << row[j];
            if (j < row.size() - 1) fid << " & ";
        }
        fid << " \\\\" << endl;
        fid << "\\hline" << endl;
    }

    // Close table environment
    fid << "\\end{tabular}" << endl;
    fid << "\\end{table}" << endl;

    fid.close();
}

#endif