#pragma once

#include "halfedge.h"
#include <iostream>
#include <unordered_set>
#include <deque>
#include <set>

using namespace Eigen;
using namespace std;

struct CustomCompare {
    bool operator()(const pair<float, Edge*>& lhs, const pair<float, Edge*>& rhs) const {
        return lhs.first < rhs.first; // greater sign coz i am sorting priority by least edge cost
    }
};

class HalfEdgeMesh
{
public:
    HalfEdgeMesh();
    ~HalfEdgeMesh();

    bool runValidator=false;
    bool validateMesh();

    void buildHalfEdgeMesh(const vector<Vector3f> &vertices, const vector<Vector3i> &faces);
    void printMeshStats();
    void createObjFileVerticesFaces(vector<Vector3f> &vertices, vector<Vector3i> &faces);

    // utils
    Vector3i getVertexIdsFromFace(Face* f);
    vector<int> getFaceIdsFromVertex(Vertex* v);
    Vector3f getFaceNormal(Face* f);
    Vector3f getVertexNormal(Vertex* v);

    // atomic operations
    Vertex* edgeCollapse(Edge* edge, Vector3f collapsePoint); // this returns the vertex obj ptr to which the edge was collapsed to

    // geoprocessing operations
    void computeEdgeFeatures();     // used in meshenv v2
    void initQEMCosts(bool greedyQEMAgent = false); // simplify operation
    float updateEdgeQEMCost(Edge* edge);
    // Vector3f minimizeEdgeQuadric(Edge* edge);
    pair<int, float> removeEdge(int edgeId);

    unordered_map<int, Vertex*> vertexMap; // maps vertexID --> Vertex
    unordered_map<int, Face*> faceMap; // maps faceID --> Face
    unordered_map<int, Edge*> edgeMap; // maps edgeID --> Edge
    unordered_map<int, HalfEdge*> halfEdgeMap;

    float randomQEMStep();
    float greedyQEMStep();
    multiset<pair<float, Edge*>, CustomCompare> priorityQueue; // (edgeCost, edgeID, minCollapsePt)
private:
    // unique id counters for each halfedge mesh element
    unsigned long long halfEdgeIdCounter = 0;
    unsigned long long vertexIdCounter = 0;
    unsigned long long faceIdCounter = 0;
    unsigned long long edgeIdCounter = 0;

    int initialEdgeCount = 0;
    int initialVertexCount = 0;
    int initialFaceCount = 0;

    void cleanUpHalfEdgeMesh();

    // geo processing helpers
    bool isManifoldAfterEdgeCollapse(Edge* edge, Vector3f collapsePoint);
    Matrix4f getVertexQuadric(Vertex *v);
    void addEdgeToQueue(multiset<pair<float, Edge*>, CustomCompare> &priorityQueue, Edge* edge);
    void updateEdgeInQueue(multiset<pair<float, Edge*>, CustomCompare> &priorityQueue, Edge* edgeToUpdate);

    // utils
    void addHalfEdges(vector<HalfEdge*> hes);
    void addEdges(vector<Edge*> es);
    void addFaces(vector<Face*> fs);
    void addVertices(vector<Vertex*> vs);
    void deleteHalfEdges(vector<HalfEdge*> hes);
    void deleteEdges(vector<Edge*> es);
    void deleteFaces(vector<Face*> fs);
    void deleteVertices(vector<Vertex*> vs);

};

