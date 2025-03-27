#ifndef __INTERVALTREE_HPP__
#define __INTERVALTREE_HPP__

#include <array>
#include <vector>
#include <iostream>
#include <deque>
#include <cmath>

/**
 * @file IntervalTree.hpp
 * @brief Implements an interval tree data structure for efficient point-in-polygon testing
 *
 * This file provides an implementation of an interval tree optimized for fast point containment
 * queries in 2D polygons. The implementation uses the ray-casting algorithm (even-odd rule)
 * to determine if a point is inside a polygon by counting intersections.
 */

using namespace std;

/**
 * @typedef vertex
 * @brief A 2D point represented as an array of two doubles [x,y]
 */
typedef array<double,2> vertex;

/**
 * @typedef coords
 * @brief A collection of vertices representing points in 2D space
 */
typedef vector<vertex> coords;

/**
 * @typedef segment_ids
 * @brief A collection of index pairs referencing vertices that form segments
 */
typedef vector<array<int,2>> segment_ids;

/**
 * @namespace Axis
 * @brief Constants for specifying the sorting axis for the interval tree
 */
namespace Axis {
    const static short X = 0; ///< X-axis constant
    const static short Y = 1; ///< Y-axis constant
}

/**
 * @class Segment
 * @brief Represents a line segment between two points with precomputed properties
 *
 * Stores a line segment defined by start and end points, and precomputes properties
 * needed for efficient interval tree operations including midpoint and axis-aligned bounds.
 *
 * @tparam T The axis along which this segment will be sorted (0 for X, 1 for Y)
 */
template <short T>
class Segment {
public:
    /**
     * @brief Default constructor
     */
    Segment(){};
    
    /**
     * @brief Constructs a segment from two vertices
     *
     * @param start Starting point of the segment
     * @param end Ending point of the segment
     */
    Segment(vertex start, vertex end) {
        this->start = start;
        this->end = end;
        this->midpoint[0] = (start[0] + end[0]) / 2;
        this->midpoint[1] = (start[1] + end[1]) / 2;
        this->low = std::min(start[T], end[T]);
        this->high = std::max(start[T], end[T]);
    }

    vertex start;     ///< Starting point of the segment
    vertex end;       ///< Ending point of the segment
    vertex midpoint;  ///< Midpoint of the segment
    double low;       ///< Minimum value of the segment along the sorting axis
    double high;      ///< Maximum value of the segment along the sorting axis
};

/**
 * @class IntervalNode
 * @brief A node in the interval tree containing segments
 *
 * Each node stores segments that intersect its center coordinate along the sorting axis.
 * Segments entirely to the left or right of the center are stored in child nodes.
 *
 * @tparam T The axis along which segments are sorted (0 for X, 1 for Y)
 */
template <short T>
class IntervalNode {
public:
    /**
     * @brief Default constructor that initializes an empty node
     */
    IntervalNode() : left(nullptr), right(nullptr) {}
    
    deque<Segment<T>> segments;  ///< Segments to be distributed during construction
    IntervalNode* left;          ///< Left child containing segments with high values below center
    IntervalNode* right;         ///< Right child containing segments with low values above center
    vector<double> low;          ///< Low values of segments that cross the center
    vector<double> high;         ///< High values of segments that cross the center
    vector<double> off_low;      ///< Values on the perpendicular axis for segments crossing center
    double center;               ///< Center coordinate that splits segments

    // Commented out to prevent memory leaks in incomplete implementation
    // /**
    //  * @brief Destructor that recursively deletes all child nodes
    //  */
    // ~IntervalNode() {
    //     if (left != nullptr) {
    //         delete left;
    //     }
    //     if (right != nullptr) {
    //         delete right;
    //     }
    // }

    /**
     * @brief Constructs the interval tree from segments
     *
     * Distributes segments into this node and its children based on whether they
     * are entirely to the left, entirely to the right, or crossing the center.
     */
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

    /**
     * @brief Prints the node structure to the standard output for debugging
     */
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

    /**
     * @brief Counts how many segments a ray from the query point to infinity crosses
     *
     * Implements the ray-casting algorithm by counting segments that lie
     * on the positive x-axis from the query point. An odd count indicates
     * the point is inside the polygon.
     *
     * @param P The query point to test
     * @return Number of segments crossed by a ray from P in the positive direction
     */
    int QueryPoint(vertex P){
        // Check if the point is in the node
        int count = 0;
        for (unsigned long i = 0; i < low.size(); i++) {
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

/**
 * @class IntervalTree
 * @brief A spatial data structure for efficient point-in-polygon testing
 *
 * Implements a specialized interval tree that enables fast point-in-polygon tests
 * by preprocessing a set of line segments. The tree partitions segments based on
 * their projection along one axis.
 *
 * @tparam T The axis along which the tree is organized (0 for X, 1 for Y)
 */
template <short T>
class IntervalTree {
    public:
        segment_ids seg_ids;        ///< Original segment indices
        coords coordinates;         ///< Original vertex coordinates
        std::vector<std::array<double,5>> data = {}; ///< Optional data associated with segments

        /**
         * @brief Constructs an interval tree from a set of segments
         *
         * @param segs Vector of segment indices referencing vertex pairs
         * @param coordinates Vector of vertex coordinates
         */
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

        /**
         * @brief Constructs an interval tree with associated segment data
         *
         * Similar to the standard constructor but allows attaching additional data
         * to each segment, such as normal vectors or curvature information.
         *
         * @param segs Vector of segment indices referencing vertex pairs
         * @param coordinates Vector of vertex coordinates
         * @param data Additional data associated with each segment (e.g., normal vectors)
         */
        IntervalTree(segment_ids segs, coords coordinates, std::vector<std::array<double,5>> data){
            this->seg_ids = segs;
            this->coordinates = coordinates;
            this->data = data;
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

        /**
         * @brief Tests if a point is inside the polygon defined by the tree's segments
         *
         * Uses the ray-casting algorithm (even-odd rule) to determine if the point
         * is inside the polygon.
         *
         * @param P The query point to test
         * @return An integer where odd values indicate the point is inside
         */
        int QueryPoint(vertex P){
            return root->QueryPoint(P);
        }
    
    private:
        vector<Segment<T>> segments; ///< Internal representation of all segments
        IntervalNode<T>* root;      ///< Root node of the interval tree
};

#endif // __INTERVALTREE_HPP__