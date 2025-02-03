#ifndef __KDTREE_HPP__
#define __KDTREE_HPP__

#include "IntervalTree.hpp"

template <short DATA_SIZE>
class KDNode {
public:
    KDNode() : left(nullptr), right(nullptr) {} // Default constructor
    vertex point;
    double data[DATA_SIZE];
    KDNode* left;
    KDNode* right;
    unsigned int depth = -1;

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
                left->Insert(P, data, depth + 1);
            } else {
                left->Insert(P, data, depth + 1);
            }
        } else {
            if (right == nullptr) {
                right = new KDNode();
                right->Insert(P, data, depth + 1);
            } else {
                right->Insert(P, data, depth + 1);
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
    public :
        KDTree() : root(nullptr) {} // Default constructor
        KDNode<DATA_SIZE>* root;

        ~KDTree() { // Destructor
            if (root != nullptr) {
                delete root;
            }
        }

        void Insert(vertex P, double data[DATA_SIZE]) {
            if (root == nullptr) {
                root = new KDNode<DATA_SIZE>();
            }
            root->Insert(P, data, 0);
        }

        void Print() {
            if (root != nullptr) {
                root->Print();
            }
        }
};

#endif // __KDTREE_HPP__