#include "meshenv.h"

#include <iostream>
#include <fstream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "util/tiny_obj_loader.h"
#include <cassert>

//#include <QFileInfo>
//#include <QString>

using namespace Eigen;
using namespace std;

MeshEnv::MeshEnv(string configFilePath) {
    iniFilePath = configFilePath;
    halfEdgeMesh = new HalfEdgeMesh();
}


void MeshEnv::reset() {
    initMeshEnv();
}

void MeshEnv::initMeshEnv() {
    loadFromFile(iniFilePath); // sets up halfedge datastruct for the specified mesh file path
    halfEdgeMesh->initQEMCosts(); // calcs up QEM cost of each edge in the mesh

    initialEdgeCount = halfEdgeMesh->edgeMap.size();
    assert(halfEdgeMesh->edgeMap.size() <= maxEdgeCount && "num of edges in input mesh > maxEdgeCount");
    assert(halfEdgeMesh->faceMap.size() <= maxFaceCount && "num of faces in input mesh > maxFaceCount");
    assert(halfEdgeMesh->vertexMap.size() <= maxVertexCount && "num of vertices in input mesh > maxVertexCount");

    // meshState.clear();
    if (meshState.size() == 0) {
        int meshStateSize = maxVertexCount + maxFaceCount; // halfEdgeMesh->vertexMap.size() + halfEdgeMesh->faceMap.size();
        meshState.reserve(meshStateSize);
        for (int i=0; i<meshStateSize; ++i) {
            meshState.push_back({-1, -1, -1}); // add empty rows
        }
    }
}

vector<vector<float>>& MeshEnv::getState() {

    // cout << meshState.size() << endl; cout << initialVertexCount<< endl; cout << initialFaceCount<< endl;
    for (int j=0; j<maxFaceCount; ++j) {
        if (!halfEdgeMesh->faceMap.contains(j)) {
            meshState[maxVertexCount + j] = {-1, -1, -1};
            continue;
        }
        Face* f = halfEdgeMesh->faceMap[j];
        Vector3i f3i = halfEdgeMesh->getVertexIdsFromFace(f);
        meshState[maxVertexCount + f->id] = {(float)f3i[0], (float)f3i[1], (float)f3i[2]};
    }
    for (int j=0; j<maxVertexCount; ++j) {
        if (!halfEdgeMesh->vertexMap.contains(j)) {
            meshState[j] = {-1, -1, -1};
            continue;
        }
        Vertex* v = halfEdgeMesh->vertexMap[j];
        Vector3f v3f = v->vertex3f;
        meshState[v->id] = {v3f[0], v3f[1], v3f[2]};
    }

    // cout << "mesh faces: " << halfEdgeMesh->faceMap.size() << endl; cout << "mesh vertices: " << halfEdgeMesh->vertexMap.size() << endl; // cout << "mesh state size: " << i << endl;
    // printVec(meshState);
    return meshState;
}

pair<float, bool> MeshEnv::step(int action) {
    // here action == edgeId
    int edgeId = action;
    float reward = 0;
    bool isTerminal = false;

    /*
     * returns <error code, QEM cost for that edge collapse>
     * possible error codes:
     *  0 - no error, edge collapse successful
     *  1 - edge does never existed at all
     *  2 - edge does not exist (was deleted)
     *  3 - edge was not collapsed because it breaks manifoldness property of the mesh
     */
    if (edgeId >= initialEdgeCount) { // error code = 1
        reward = 0;
        cout << "--- edge id " << action << " never existed" << endl;
    } else {
        pair<int, float> res = halfEdgeMesh->removeEdge(edgeId);
        int errorCode = res.first;
        if (errorCode == 2) {
            reward = 0;
            cout << "--- edge id " << action << " does not exist (was deleted)" << endl;
        } else if (errorCode == 3) {
            reward = 100;
            cout << "--- edge id " << action << " was not collapsed due to breaking manifoldness" << endl;
        } else {
            reward = res.second;
            cout << "removed edge id " << action << endl;
        }
    }

    // terminal conditions
    // - the smallest possible manifold mesh is a tetrahedron, which has 6 edges and 4 faces
    // - the number of faces in the current state is less than equal to what the user wants in the simplified mesh result
    if (getEdgeCount() <= 6 || halfEdgeMesh->faceMap.size() <= numFacesInResult) {
        isTerminal = true;
    }

    return {reward, isTerminal};
}


void MeshEnv::initFromVectors(const vector<Vector3f> &vertices, const vector<Vector3i> &faces) {
    // Copy vertices and faces into internal vector
    _vertices = vertices;
    _faces    = faces;
    halfEdgeMesh = new HalfEdgeMesh();
}

void MeshEnv::loadFromFile(const string &iniFilePath) {
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;

    // QFileInfo info(QString(iniFilePath.c_str()));
    string err;
//    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
//                                info.absoluteFilePath().toStdString().c_str(), (info.absolutePath().toStdString() + "/").c_str(), true);
//    cout << info.absoluteFilePath().toStdString().c_str() << endl;
//    cout << (info.absolutePath().toStdString() + "/").c_str() << endl;
    string p = iniFilePath;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                                "/Users/mohammedk/Documents/Brown/CS2951F/Final Project/MeshSimplificationRL/meshenv/meshes/icosahedron.obj",
                                "/Users/mohammedk/Documents/Brown/CS2951F/Final Project/MeshSimplificationRL/meshenv/meshes/", true);

    if (!err.empty()) {
        cerr << err << endl;
    }

    if (!ret) {
        cerr << "Failed to load/parse .obj file" << endl;
        return;
    }

    _faces.clear();
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            unsigned int fv = shapes[s].mesh.num_face_vertices[f];

            Vector3i face;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                face[v] = idx.vertex_index;
            }
            _faces.push_back(face);

            index_offset += fv;
        }
    }
    _vertices.clear();
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        _vertices.emplace_back(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
    }
    cout << "Loaded " << _faces.size() << " faces and " << _vertices.size() << " vertices" << endl;

//    delete halfEdgeMesh;
//    halfEdgeMesh = new HalfEdgeMesh();
    halfEdgeMesh->buildHalfEdgeMesh(_vertices, _faces);
}

void MeshEnv::saveToFile(const string &filePath) {
    ofstream outfile;
    outfile.open(filePath);

    halfEdgeMesh->createObjFileVerticesFaces(_vertices, _faces);

    // Write vertices
    for (size_t i = 0; i < _vertices.size(); i++)
    {
        const Vector3f &v = _vertices[i];
        outfile << "v " << v[0] << " " << v[1] << " " << v[2] << endl;
    }

    // Write faces
    for (size_t i = 0; i < _faces.size(); i++)
    {
        const Vector3i &f = _faces[i];
        outfile << "f " << (f[0]+1) << " " << (f[1]+1) << " " << (f[2]+1) << endl;
    }

    outfile.close();
}


