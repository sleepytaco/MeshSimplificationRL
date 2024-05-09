#include "meshenv.h"
#include <random>

// samples N points from mesh
vector<Vector3f> MeshEnv::samplePoints(int numPoints, HalfEdgeMesh* mesh) {

    if (numPoints > mesh->vertexMap.size()) numPoints = mesh->vertexMap.size();

    vector<Vector3f> samples(numPoints, Vector3f(0, 0, 0));

    // Seed the random number generator
    random_device rd;
    mt19937 g(rd());

    // Shuffle the elements of the map
    vector<pair<int, Vertex*>> vec(mesh->vertexMap.begin(), mesh->vertexMap.end());
    shuffle(vec.begin(), vec.end(), g);

    // Sample N elements from the shuffled vector
    for (int i = 0; i < numPoints && i < vec.size(); ++i) {
        samples[i] = vec[i].second->vertex3f;
    }

    return samples;
}

// from QEM paper: d(v, M) = min p ∈ M ‖v − p‖ is the minimum distance from v to the closest face of M
float MeshEnv::minDistance(Vector3f v, HalfEdgeMesh* mesh) {
    float minDist = numeric_limits<float>::max();
    for (auto it=mesh->vertexMap.begin(); it != mesh->vertexMap.end(); ++it) {
        Vector3f v_ = it->second->vertex3f;

        float eucledianDist = (v - v_).norm();

        minDist = min(minDist, eucledianDist);
    }
    return minDist;
}

float MeshEnv::approximationError(HalfEdgeMesh* originalMesh, HalfEdgeMesh* simplifiedMesh) {
    // notation from QEM paper https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf
    HalfEdgeMesh* Mn = originalMesh;
    HalfEdgeMesh* Mi = simplifiedMesh;

//    vector<Vector3f> Xn = samplePoints(150, Mn);
//    vector<Vector3f> Xi = samplePoints(150, Mi);

    float E = 0;

    for (auto& vm : Mn->vertexMap) {
        Vector3f v = vm.second->vertex3f;
    // for (Vector3f& v : Xn) {
        E += powf(minDistance(v, Mi), 2.f);
    }

    for (auto& vm : Mi->vertexMap) {
        Vector3f v = vm.second->vertex3f;
    // for (Vector3f& v : Xi) {
        E += powf(minDistance(v, Mn), 2.f);
    }

    E = E / (float) (Mn->vertexMap.size() + Mi->vertexMap.size());
    return E;
}
