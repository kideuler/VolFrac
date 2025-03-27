#ifndef __KDTREE_HPP__
#define __KDTREE_HPP__

#include "IntervalTree.hpp"

/**
 * @file KDTree.hpp
 * @brief Implements a k-d tree data structure for efficient nearest neighbor searches
 *
 * This file provides an implementation of a k-d tree specialized for 2D space,
 * allowing fast nearest neighbor queries. Each node can store additional data
 * associated with points in the tree.
 */

/**
 * @brief Calculates the squared Euclidean distance between two vertices
 *
 * Helper function that computes the squared distance between two 2D points
 * without taking the square root, for efficiency in comparisons.
 *
 * @param a First vertex
 * @param b Second vertex
 * @return The squared Euclidean distance between the points
 */
inline double squaredDistance(const vertex &a, const vertex &b) {
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
}

/**
 * @class KDNode
 * @brief Node in a k-d tree storing a point and associated data
 *
 * Each node in the k-d tree contains a 2D point, associated data, and pointers
 * to child nodes. The tree alternates between splitting on x and y coordinates
 * as depth increases.
 *
 * @tparam DATA_SIZE Size of the data array associated with each point
 */
template <short DATA_SIZE>
class KDNode {
public:
    /**
     * @brief Default constructor initializing pointers to null
     */
    KDNode() : left(nullptr), right(nullptr), parent(nullptr) {}
    
    vertex point;                ///< The 2D point stored in this node
    double data[DATA_SIZE];      ///< Array of data values associated with the point
    KDNode* left;               ///< Pointer to left child (points with smaller coordinate value)
    KDNode* right;              ///< Pointer to right child (points with larger coordinate value)
    KDNode* parent;             ///< Pointer to parent node
    int depth = -1;             ///< Depth of this node in the tree (-1 for uninitialized nodes)

    /**
     * @brief Destructor that recursively deletes child nodes
     *
     * Ensures proper cleanup of the entire subtree when a node is deleted.
     */
    ~KDNode() {
        if (left != nullptr) {
            delete left;
        }
        if (right != nullptr) {
            delete right;
        }
    }

    /**
     * @brief Inserts a point and its associated data into the subtree rooted at this node
     *
     * The insertion algorithm alternates between comparing x and y coordinates
     * based on the depth of the node in the tree.
     *
     * @param P The point to insert
     * @param data Array of data values associated with the point
     * @param depth Current depth in the tree (determines splitting axis)
     */
    void Insert(vertex P, double data[DATA_SIZE], unsigned int depth) {
        if (this->depth == -1) {
            this->point = P;
            for (int i = 0; i < DATA_SIZE; i++) {
                this->data[i] = data[i];
            }
            this->depth = depth;
            return;
        }

        unsigned int axis = depth % 2;

        if (P[axis] < point[axis]) {
            if (left == nullptr) {
                left = new KDNode();
                left->parent = this;
                left->Insert(P, data, depth + 1);
            } else {
                left->Insert(P, data, depth + 1);
            }
        } else {
            if (right == nullptr) {
                right = new KDNode();
                right->parent = this;
                right->Insert(P, data, depth + 1);
            } else {
                right->Insert(P, data, depth + 1);
            }
        }
    }

    /**
     * @brief Recursively searches the subtree for the nearest neighbor to a query point
     *
     * Implements the standard k-d tree nearest neighbor search algorithm, which
     * traverses down the tree and then backtracks, checking if branches on the other
     * side of splitting planes need to be explored.
     *
     * @param P The query point
     * @param best_radius Reference to the current best (squared) distance
     * @param best_point Reference to store the closest point found
     * @param best_data Array to store the data associated with the closest point
     */
    void Search_Down(vertex P, double &best_radius, vertex &best_point, double *best_data) {
        if (this->depth == -1) {
            return;
        }

        double distance = squaredDistance(point, P);
        if (distance < best_radius) {
            best_radius = distance;
            best_point = point;
            for (int i = 0; i < DATA_SIZE; i++) {
                best_data[i] = data[i];
            }
        }

        unsigned int axis = depth % 2;
        double axis_diff_sq = (point[axis] - P[axis]) * (point[axis] - P[axis]);

        if (P[axis] < point[axis]) {
            if (left != nullptr) {
                left->Search_Down(P, best_radius, best_point, best_data);
            }
            if (right != nullptr && axis_diff_sq < best_radius) {
                right->Search_Down(P, best_radius, best_point, best_data);
            }
        } else {
            if (right != nullptr) {
                right->Search_Down(P, best_radius, best_point, best_data);
            }
            if (left != nullptr && axis_diff_sq < best_radius) {
                left->Search_Down(P, best_radius, best_point, best_data);
            }
        }
    }

    /**
     * @brief Prints the node information and recursively prints child nodes
     *
     * Debugging function that prints the point coordinates, associated data,
     * depth, and recursively prints the left and right subtrees.
     */
    void Print() {
        cout << "Point: (" << point[0] << ", " << point[1] << ")" << endl;
        cout << "Data: [";
        for (int i = 0; i < DATA_SIZE; i++) {
            cout << data[i] << " ";
        }
        cout << "]" << endl;
        cout << "Depth: " << depth << endl;
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
};

/**
 * @class KDTree
 * @brief A k-d tree implementation for efficient nearest neighbor searches in 2D space
 *
 * This class provides a k-d tree data structure that partitions 2D space
 * recursively along alternating dimensions. It supports efficient nearest
 * neighbor queries and can store additional data with each point.
 *
 * @tparam DATA_SIZE Size of the data array associated with each point
 */
template <short DATA_SIZE>
class KDTree {
public:
    /**
     * @brief Default constructor initializing an empty tree
     */
    KDTree() : root(nullptr) {}
    
    KDNode<DATA_SIZE>* root;    ///< Root node of the tree

    /**
     * @brief Inserts a point and its associated data into the tree
     *
     * @param P The point to insert
     * @param data Array of data values associated with the point
     */
    void Insert(vertex P, double data[DATA_SIZE]) {
        if (root == nullptr) {
            root = new KDNode<DATA_SIZE>();
        }
        root->Insert(P, data, 0);
    }

    /**
     * @brief Searches for the nearest neighbor to a query point
     *
     * Finds the point in the tree that is closest to the query point
     * according to Euclidean distance.
     *
     * @param P The query point
     * @param best_point Reference to store the closest point found
     * @param best_data Array to store the data associated with the closest point
     */
    void Search(vertex P, vertex &best_point, double *best_data) {
        if (root == nullptr) {
            return;
        }
        double best_radius = 1e34;  // Initialize with a very large value
        root->Search_Down(P, best_radius, best_point, best_data);
    }

    /**
     * @brief Prints the entire tree structure
     *
     * Debugging function that prints all nodes in the tree in a hierarchical format.
     */
    void Print() {
        if (root != nullptr) {
            root->Print();
        }
    }
};

#endif // __KDTREE_HPP__