#pragma once

#include "halfedgemesh.h"

#include <vector>
#include <unordered_map>
#include "Eigen/StdVector"
#include "Eigen/Dense"


EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3i);

using namespace Eigen;
using namespace std;


class MeshEnv
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    MeshEnv(string meshFilePath = "", int maxFaceCount = 200,
                                int maxVertexCount = 102,
                                int maxEdgeCount = 300 // i.e. num of actions
                                );
    void initMeshEnv();
    void initFromVectors(const std::vector<Eigen::Vector3f> &vertices,
                         const std::vector<Eigen::Vector3i> &faces);

    void loadFromFile();
    void saveToFile(const std::string &filePath);

    HalfEdgeMesh* halfEdgeMesh;

    bool isTraining = true;
    int finalFaceCount = 50; // final simplified mesh must contain atmost these many faces (used as terminal condition)

    string getMeshFilePath() {return meshFilePath;}
    int getEdgeCount() { return halfEdgeMesh->edgeMap.size(); };
    int getFaceCount() { return halfEdgeMesh->faceMap.size(); };

    void setMeshFilePath(string s) { cout << "Set mesh file path to " + s << endl; meshFilePath = s;};
    void setIsTraining(bool b) { isTraining = b; };
    void setFinalFaceCount(int fc) { finalFaceCount = fc;};

    void printEpisodeStats();

    void reset();
    vector<vector<float>>& getState();
    pair<float, bool> step(int action);

private:
    vector<Vector3f> _vertices;
    vector<Vector3i> _faces;

    float emptyVal = 0; // the value to fill empty rows with
    int maxSteps = 3000;
    bool printSteps = false; // prints the edge collapse operations out
    bool reachedRequiredFaces = false;

    string meshFilePath = "";
    int initialEdgeCount = 0;
    int episodeCount = 0;
    float maxRewardGiven = 0;

    // episode stats
    float episodeRewards = 0;
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

    vector<vector<float>> meshState; // first half is vertices, second half is faces

    void printVec(const std::vector<std::vector<float>>& vec) {
        for (const auto& row : vec) {
            for (const auto& elem : row) {
                cout << elem << ", ";
            }
            std::cout << std::endl;
        }
        cout << " ---- " << endl;
    };
};
