#pragma once

#include "halfedgemesh.h"
#include <vector>
#include <unordered_map>
#include "Eigen/StdVector"
#include "Eigen/Dense"
#include "nlohmann/json.hpp"


EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3i);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;


class MeshEnv
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    MeshEnv(string meshFilePath = "", int maxFaceCount = 500, //200,
                                int maxVertexCount = 252, //102,
                                int maxEdgeCount = 750 // 300 // i.e. num of actions
                                );

    int maxSteps = 1023;
    bool printSteps = false; // prints the edge collapse operations out
    int envVersion = 0;
    bool isTraining = true;
    int finalFaceCount = 150; // final simplified mesh must contain atmost these many faces (used as terminal condition)
    float emptyVal = -1; // the value to fill empty rows with
    bool reachedRequiredFaces = false;


    void initMeshEnv();
    void initFromVectors(const std::vector<Eigen::Vector3f> &vertices,
                         const std::vector<Eigen::Vector3i> &faces);

    void loadFromFile();
    void saveToFile(const std::string &filePath);

    HalfEdgeMesh* originalMesh;
    HalfEdgeMesh* halfEdgeMesh; // this is the mesh the RL agent updates
    HalfEdgeMesh* halfEdgeMeshGreedy; // this is the mesh the default QEM algo updates (greedily collapses min QEM cost edges)
    HalfEdgeMesh* halfEdgeMeshRandom; // this is the mesh where the edges are collapsed randomly each step

    string getMeshFilePath() {return meshFilePath;}
    int getEdgeCount() { return halfEdgeMesh->edgeMap.size(); };
    int getFaceCount() { return halfEdgeMesh->faceMap.size(); };

    void setMeshFilePath(string s) { cout << "\n***** Set MESH FILE PATH to " + s + " *****\n"<< endl; meshFilePath = s;};
    void setIsTraining(bool b) {
        if (b) cout << "\n****** Set meshenv to TRAINING mode ******\n" << endl;
        else cout << "\n****** Set meshenv to TESTING mode ******\n" << endl;
        isTraining = b;
    };
    void setFinalFaceCount(int fc) { finalFaceCount = fc;};

    void printEpisodeStats();
    void saveEpisodeStats(json& j);
    void saveValidEdgeIds(json& j) {
        vector<int> edgeIds;
        edgeIds.reserve(halfEdgeMesh->edgeMap.size());
        for (const auto& em : halfEdgeMesh->edgeMap) {
            int edgeId = em.first;
            edgeIds.push_back(edgeId);
        }
        j["validEdgeIds"] = edgeIds;
    }

    // in V1 the action space is discrete --- the agent picks from 750 edges
    void reset();
    vector<vector<float>>& getState();
    pair<float, bool> step(int action);

    vector<vector<float>> meshState; // first half is vertices, second half is faces
    vector<vector<float>> meshStateV2; // first half is vertices, second half is faces

    // in V2 the action space is continous --- x, y, z coordinates of the edge to collapse are predicted
    void resetV2();
    vector<vector<float>>& getStateV2();
    pair<float, bool> stepV2(Vector3f xyz);
    void setVersion(int v);

    // utils
    vector<Vector3f> samplePoints(int numPoints, HalfEdgeMesh* mesh);
    float minDistance(Vector3f v, HalfEdgeMesh* mesh);
    float approximationError(HalfEdgeMesh* originalMesh, HalfEdgeMesh* simplifiedMesh);

private:
    vector<Vector3f> _vertices;
    vector<Vector3i> _faces;

    string meshFilePath = "";
    int initialEdgeCount = 0;
    int episodeCount = 0;

    // episode reward stats
    float episodeRewards = 0; // total episode rewards
    float episodeQEMErrorRewards = 0;
    float episodeApproxErrorRewards = 0;
    float maxQEMRewardGiven = 0;
    float maxApproximationError = 0;
    vector<float> agentQEMCosts;
    vector<float> greedyQEMCosts; // deterministic greedy QEM agent
    vector<float> randomQEMCosts; // random agent

    // episode edge collapse stats
    long long int totalEnvSteps = 0;
    int numCollapses = 0;
    int numNonManifoldCollapses = 0;
    int numDeletedEdgeCollapses = 0;
    int numDNEEdgeCollapses = 0;

    // must be set when env is initialized
    // any thing beyond these will break the RL agent
    // visit the constructor to update the defaults for this
    int maxFaceCount;
    int maxVertexCount;
    int maxEdgeCount; // i.e. num of actions

    void printVec2d(const std::vector<std::vector<float>>& vec) {
        for (const auto& row : vec) {
            for (const auto& elem : row) {
                cout << elem << ", ";
            }
            std::cout << std::endl;
        }
        cout << " ---- " << endl;
    };

    void printVec1d(const std::vector<float>& vec) {
        for (const auto& elem : vec) {
            cout << elem << ", ";
        }
        std::cout << std::endl;
    };
};
