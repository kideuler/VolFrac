#ifndef __KDTREE_HPP__
#define __KDTREE_HPP__

#include "IntervalTree.hpp"

// Inline helper function to compute squared distance between two vertices.
inline double squaredDistance(const vertex &a, const vertex &b) {
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
}

template <short DATA_SIZE>
class KDNode {
public:
    KDNode() : left(nullptr), right(nullptr), parent(nullptr) {} // Default constructor
    vertex point;
    double data[DATA_SIZE];
    KDNode* left;
    KDNode* right;
    KDNode* parent;
    int depth = -1;

    ~KDNode() { // Destructor
        if (left != nullptr) {
            delete left;
        }
        if (right != nullptr) {
            delete right;
        }
    }

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

template <short DATA_SIZE>
class KDTree {
public:
    KDTree() : root(nullptr) {} // Default constructor
    KDNode<DATA_SIZE>* root;

    void Insert(vertex P, double data[DATA_SIZE]) {
        if (root == nullptr) {
            root = new KDNode<DATA_SIZE>();
        }
        root->Insert(P, data, 0);
    }

    void Search(vertex P, vertex &best_point, double *best_data) {
        if (root == nullptr) {
            return;
        }
        double best_radius = 1e34;
        root->Search_Down(P, best_radius, best_point, best_data);
    }

    void Print() {
        if (root != nullptr) {
            root->Print();
        }
    }
};

#endif // __KDTREE_HPP__