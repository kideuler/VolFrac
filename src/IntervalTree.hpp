#ifndef __INTERVALTREE_HPP__
#define __INTERVALTREE_HPP__

#include <array>
#include <vector>
#include <iostream>
#include <deque>
#include <cmath>

using namespace std;
typedef array<double,2> vertex;
typedef vector<vertex> coords;
typedef vector<array<int,2>> segment_ids;

namespace Axis {
    const static short X = 0;
    const static short Y = 1;
}

template <short T>
class Segment {
public:
    Segment(){}; // Default constructor
    Segment(vertex start, vertex end) {
        this->start = start;
        this->end = end;
        this->midpoint[0] = (start[0] + end[0]) / 2;
        this->midpoint[1] = (start[1] + end[1]) / 2;
        this->low = std::min(start[T], end[T]);
        this->high = std::max(start[T], end[T]);
    }

    vertex start;
    vertex end;
    vertex midpoint;
    double low;
    double high;
};

template <short T>
class IntervalNode {
public:
    IntervalNode() : left(nullptr), right(nullptr) {} // Default constructor
    deque<Segment<T>> segments;
    IntervalNode* left;
    IntervalNode* right;
    vector<double> low;
    vector<double> high;
    vector<double> off_low;
    double center;

    ~IntervalNode() {
        if (left != nullptr) {
            delete left;
        }
        if (right != nullptr) {
            delete right;
        }
    }

    void Construct(){
        if (segments.empty()) {
            return;
        }

        // find the center of the segments
        center = 0.0;
        for (auto segment : segments) {
            center += segment.midpoint[T];
        }
        center /= (double)segments.size();

        // Split the segments into left and right children
        left = new IntervalNode<T>();
        right = new IntervalNode<T>();
        while (!segments.empty()) {
            Segment<T> segment = segments.front();
            segments.pop_front();
            if (segment.high < center) {
                left->segments.push_back(segment);
            } else if (segment.low > center) {
                right->segments.push_back(segment);
            } else {
                low.push_back(segment.low);
                high.push_back(segment.high);
                off_low.push_back(std::min(segment.start[1-T], segment.end[1-T]));
            }
        }

        // Recursively construct the left and right children
        left->Construct();
        right->Construct();
    }

    void Print() {
        cout << "Center: " << center << endl;
        cout << "Low: [";
        for (auto l : low) {
            cout << l << " ";
        }
        cout << "]" << endl;
        cout << "High: [";
        for (auto h : high) {
            cout << h << " ";
        }
        cout << "]" << endl;
        cout << "Left: ";
        if (left != nullptr) {
            left->Print();
        } else {
            cout << "nullptr" << endl;
        }
        cout << "Right: ";
        if (right != nullptr) {
            right->Print();
        } else {
            cout << "nullptr" << endl;
        }
    }


    int QueryPoint(vertex P){

        // Check if the point is in the node
        int count = 0;
        for (int i = 0; i < low.size(); i++) {
            if (low[i] < P[T] && P[T] <= high[i] && P[1-T] < off_low[i]) {
                count++;
            }
        }

        // Recursively check the children
        if (P[T] <= center && left != nullptr) {
            count += left->QueryPoint(P);
        }
        if (P[T] > center && right != nullptr) {
            count += right->QueryPoint(P);
        }

        return count;
    }
};

template <short T>
class IntervalTree {
    public:
        segment_ids seg_ids;
        coords coordinates;

        IntervalTree(segment_ids segs, coords coordinates){
            this->seg_ids = segs;
            this->coordinates = coordinates;
            int nsegments = segs.size();
            this->segments.resize(nsegments);
            for (int i = 0; i<nsegments; i++){
                this->segments[i] = Segment<T>(coordinates[segs[i][0]], coordinates[segs[i][1]]);
            }

            root = new IntervalNode<T>();
            // copy the segments to the root node
            for (auto segment : this->segments) {
                root->segments.push_back(segment);
            }
            root->Construct();
        };

        int QueryPoint(vertex P){
            return root->QueryPoint(P);
        }
    

    private:
        vector<Segment<T>> segments;
        IntervalNode<T>* root;
};

#endif