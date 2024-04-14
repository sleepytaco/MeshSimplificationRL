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

    MeshEnv(string meshFilePath, int maxEdgeCount = 750, // i.e. num of actions
                                int maxVertexCount = 250,
                                int maxFaceCount = 500);
    void initMeshEnv();
    void initFromVectors(const std::vector<Eigen::Vector3f> &vertices,
                         const std::vector<Eigen::Vector3i> &faces);

    void loadFromFile();
    void saveToFile(const std::string &filePath);

    HalfEdgeMesh* halfEdgeMesh;

    int getEdgeCount() { return halfEdgeMesh->edgeMap.size(); };

    void reset();
    vector<vector<float>>& getState();
    pair<float, bool> step(int action);
private:
    vector<Vector3f> _vertices;
    vector<Vector3i> _faces;

    string meshFilePath = "";

    // must be set when env is initialized
    // any thing beyond these will break the RL agent
    int maxEdgeCount = 750; // i.e. num of actions
    int maxVertexCount = 250;
    int maxFaceCount = 500;

    int initialEdgeCount = 0;
    int numFacesInResult = 100; // final simplified mesh must contain atmost these many faces (used as terminal condition)

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
