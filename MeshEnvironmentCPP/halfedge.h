#pragma once

struct Vertex;
struct Edge;
struct Face;
struct HalfEdge;

#include "Eigen/Dense"
using namespace Eigen;

struct HalfEdge {
    int id=-1;
    HalfEdge* twin;
    HalfEdge* next;
    Vertex* vertex;
    Edge* edge;
    Face* face;

    HalfEdge() {}
    HalfEdge(int id, Vertex* vertex) : id(id), vertex(vertex) {}
    HalfEdge(int id, Vertex* vertex, Face* face) : id(id), vertex(vertex), face(face) {}
    HalfEdge(int id, Vertex* vertex, Edge* edge, Face* face) : id(id), vertex(vertex), edge(edge), face(face) {}

    void setFace(Face* f) {
        if (face == nullptr) {
            face = f;
        }
    }
};

struct Vertex {
    int id=-1;
    Vector3f vertex3f;
    HalfEdge* he = nullptr;

    // for quadric error simplification
    Matrix4f Q = Matrix4f::Zero(); // quadric

    // pre-computed qty for loop subdivision (this is filled for newly created vertices during the edgeSplit operation)
    Vector3f newVertexPositionForLoop = Vector3f(0.f, 0.f, 0.f);

    Vertex() {}
    Vertex(int id, Vector3f vertex3f) : id(id), vertex3f(vertex3f) {}

    int getInDegree() {
        int numConnectedEdges=0;
        HalfEdge* h = this->he;
        do {
            numConnectedEdges += 1;
            h = h->twin->next;
        } while (h != this->he);
        return numConnectedEdges;
    }

    void setHalfEdge(HalfEdge* halfedge) {
        // only set half edge if it is initially null
        if (he == nullptr) {
            he = halfedge;
        } // else, some other half edge has already claimed this vertex
    }
};

struct Edge {
    int id=-1;
    HalfEdge* he = nullptr;
    Matrix4f Q; // quadric
    float QEMCost = 0;

    Edge() {}
    Edge(int id) : id(id) {}
    Edge(int id, HalfEdge* he) : id(id), he(he) {}

    float getEdgeLength() {
        Vector3f v0 = this->he->vertex->vertex3f;
        Vector3f v1 = this->he->twin->vertex->vertex3f;
        return (v1 - v0).norm();
    }
};

struct Face {
    int id=-1;
    HalfEdge* he = nullptr;

    // for quadric error simplification
    bool hasQuadric = false; // if set to false, quadric is calculated for this face
    Matrix4f Q = Matrix4f::Zero(); // quadric

    Vector3i face3i;

    Face() {}
    Face(int id) : id(id) {}

    Vector3f getTriangleCentroid() {
        HalfEdge* hptr = this->he;
        Vector3f centroid(0.f, 0.f, 0.f);
        do {
            centroid += hptr->vertex->vertex3f;
            hptr = hptr->next;
        } while (hptr != this->he);
        return centroid / (float) 3.f;
    }

    Vector3f getFaceNormal() {
        // visualize:
        //         v1
        //      /  |
        //  v2  f0 |
        //    \    |
        //        v0
        HalfEdge* heV0V1 = this->he;
        Vertex* v0 = heV0V1->vertex;
        Vertex* v1 = heV0V1->next->vertex;
        Vertex* v2 = heV0V1->next->next->vertex;

        Vector3f v01 = (v1->vertex3f - v0->vertex3f);
        Vector3f v02 = (v2->vertex3f - v0->vertex3f);

        Vector3f faceNormal = v01.cross(v02);
        faceNormal.normalize();
        return faceNormal;
    }

    Matrix4f getFaceQuadric() {
        if (this->hasQuadric) return this->Q;
        Vector3f faceNormal = this->getFaceNormal();
        float a=faceNormal[0], b=faceNormal[1], c=faceNormal[2];
        // float d = -faceNormal.dot(this->getTriangleCentroid());
        float d = -faceNormal.dot(this->he->vertex->vertex3f);
        Matrix4f Quadric;
        Quadric << powf(a, 2.f), a*b, a*c, a*d,
             a*b, powf(b, 2.f), b*c, b*d,
             a*c, b*c, powf(c, 2.f), c*d,
             a*d, b*d, c*d, powf(d, 2.f);
        this->Q = Quadric;
        this->hasQuadric = true;
        return this->Q;
    }
};
