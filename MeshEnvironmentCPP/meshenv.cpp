#include "meshenv.h"

#include <iostream>
#include <fstream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "util/tiny_obj_loader.h"
#include <cassert>


MeshEnv::MeshEnv(string meshFilePath, int maxFaceCount,  int maxVertexCount, int maxEdgeCount)
    : meshFilePath(meshFilePath), maxEdgeCount(maxEdgeCount),  maxVertexCount(maxVertexCount), maxFaceCount(maxFaceCount) {
    originalMesh = new HalfEdgeMesh();
    halfEdgeMesh = new HalfEdgeMesh();
    halfEdgeMeshGreedy = new HalfEdgeMesh();
    halfEdgeMeshRandom = new HalfEdgeMesh();
}

void MeshEnv::printEpisodeStats() {

    if (isTraining) cout << (reachedRequiredFaces ? "Reached the required number of faces!" : "Did NOT reach the required number of faces (episode TRUNCATED)") << endl;
    cout << "total episode steps: " << numCollapses + numNonManifoldCollapses + numDeletedEdgeCollapses + numDNEEdgeCollapses << endl;

    // print various rewards sum during episodes
    cout << "TOTAL episode rewards sum: " << episodeRewards << endl; // includes non-manifold cost + (one of QEM or approx errors)
    cout << "QEM rewards sum: " << episodeQEMErrorRewards << " | MAX QEM reward given: " << maxQEMRewardGiven << endl;
    if (episodeApproxErrorRewards > 0) cout << "APPROX ERROR rewards sum: " << episodeApproxErrorRewards << " | MAX APPROX ERROR: " << maxApproximationError << endl;


    // print edge collapse related stats
    cout << "total valid edge collapses: " << numCollapses << endl;
    cout << "total non-manifold edge collapses: " << numNonManifoldCollapses << endl;
    if (numDeletedEdgeCollapses > 0) cout << "total deleted edge collapses: " << numDeletedEdgeCollapses << endl;
    if (numDNEEdgeCollapses > 0) cout << "total does not exist edge collapses: " << numDNEEdgeCollapses << endl;

    // mostly for debugging to see if edge collapses are happening
//    cout << "face count before reset: " << halfEdgeMesh->faceMap.size() << endl;
//    cout << "vertices count before reset: " << halfEdgeMesh->vertexMap.size() << endl;
//    cout << "edge count before reset: " << halfEdgeMesh->edgeMap.size() << endl;
}

void MeshEnv::saveEpisodeStats(json& j) {

    cout << "Saving episode stats in info dict..." << endl;
    j["hasInfo"] = true;
    j["reachedRequiredFaces"] = reachedRequiredFaces;
    j["totalEpisodeSteps"] = numCollapses + numNonManifoldCollapses + numDeletedEdgeCollapses + numDNEEdgeCollapses;
    j["totalValidEdgeCollapses"] = numCollapses;
    j["totalNonManifoldEdgeCollapses"] = numNonManifoldCollapses;
    j["totalDeletedEdgeCollapses"] = numDeletedEdgeCollapses;
    j["totalDNEEdgeCollapses"] = numDNEEdgeCollapses;
    j["faceCountBeforeReset"] = halfEdgeMesh->faceMap.size();
    j["vertexCountBeforeReset"] = halfEdgeMesh->vertexMap.size();
    j["edgeCountBeforeReset"] = halfEdgeMesh->edgeMap.size();

    // reset episode reward stats
    j["episodeRewards"] = episodeRewards; // total episode rewards
    j["episodeQEMErrorRewards"] = episodeQEMErrorRewards;
    j["episodeApproxErrorRewards"] = episodeApproxErrorRewards;
    j["maxQEMRewardGiven"] = maxQEMRewardGiven;
    j["maxApproximationError"] = maxApproximationError;

    if (!isTraining) {
        j["randomQEMCostsList"] = randomQEMCosts;
        j["greedyQEMCostsList"] = greedyQEMCosts;
        j["agentQEMCostsList"] = agentQEMCosts;
    }
}

void MeshEnv::reset() {
    if (isTraining) {
        episodeCount++;
        cout << "-----MESH ENV RESET: EPISODE #" << episodeCount << "-------" << endl;
        printEpisodeStats();
    }
    cout << endl;

    initMeshEnv(); // reset mesh by loading it again

//    cout << "APPROX ERRORS BETWEEN MESHES" << endl;
//    cout << approximationError(originalMesh, originalMesh) << endl;
//    float maxQEM = -6969;
//    for (int i=0; i<150; ++i)  {
//        float QEM = halfEdgeMesh->randomQEMStep() * 10;
////        cout << "iteration " << i << ": ";
////        cout << "QEM loss - " << QEM << " | ";
////        cout << "approximation loss - " << approximationError(originalMesh, halfEdgeMesh) * 1000;
//        maxQEM = fmax(QEM, maxQEM);
//        // cout << endl;
//    }
//    cout << "MAX QEM: " << maxQEM << endl;
//    cout << "---------------------------" << endl;

    // reset episode edge collapse stats
    numCollapses = 0;
    numNonManifoldCollapses = 0;
    numDeletedEdgeCollapses = 0;
    numDNEEdgeCollapses = 0;
    reachedRequiredFaces = false;

    // reset episode reward stats
    episodeRewards = 0; // total episode rewards
    episodeQEMErrorRewards = 0;
    episodeApproxErrorRewards = 0;
    maxQEMRewardGiven = 0;
    maxApproximationError = 0;
    agentQEMCosts.clear();
    greedyQEMCosts.clear();
    randomQEMCosts.clear();

    cout << "------------------------" << endl;
}

void MeshEnv::initMeshEnv() {

    assert(meshFilePath != "" && "Mesh filepath not set!");
    loadFromFile(); // sets up halfedge datastruct for the specified mesh file path during meshenv

    // moved to loadFromFile() ---> inside buildHalfEdgeMesh()
//    halfEdgeMesh->initQEMCosts(); // calcs up QEM cost of each edge in the mesh
//    halfEdgeMeshGreedy->initQEMCosts(true); // calcs up QEM cost of each edge in the mesh
//    halfEdgeMeshRandom->initQEMCosts();

    // might not need this here, as this is called in getState everytime anyway
    // if (envVersion == 2) halfEdgeMesh->computeEdgeFeatures();

    initialEdgeCount = halfEdgeMesh->edgeMap.size();
    assert(halfEdgeMesh->edgeMap.size() <= maxEdgeCount && "num of edges in input mesh > maxEdgeCount");
    assert(halfEdgeMesh->faceMap.size() <= maxFaceCount && "num of faces in input mesh > maxFaceCount");
    assert(halfEdgeMesh->vertexMap.size() <= maxVertexCount && "num of vertices in input mesh > maxVertexCount");

    // meshState.clear();
    if (envVersion == 1 && meshState.size() == 0) { // only reserve meshState if env version is 1
        int meshStateSize = maxVertexCount + maxFaceCount; // halfEdgeMesh->vertexMap.size() + halfEdgeMesh->faceMap.size();
        meshState.reserve(meshStateSize);
        for (int i=0; i<meshStateSize; ++i) {
            meshState.push_back({emptyVal, emptyVal, emptyVal}); // add empty rows
        }
    }
}


// this is for env V1 where the state space is (numV + numF) x 3
vector<vector<float>>& MeshEnv::getState() {

    if (envVersion != 1) return meshState;

    // cout << meshState.size() << endl; cout << initialVertexCount<< endl; cout << initialFaceCount<< endl;
    for (int j=0; j<maxFaceCount; ++j) {
        if (!halfEdgeMesh->faceMap.contains(j)) {
            meshState[maxVertexCount + j] = {emptyVal, emptyVal, emptyVal};
            continue;
        }
        Face* f = halfEdgeMesh->faceMap[j];
        Vector3i f3i = halfEdgeMesh->getVertexIdsFromFace(f);
        meshState[maxVertexCount + f->id] = {(float)f3i[0], (float)f3i[1], (float)f3i[2]};
    }
    for (int j=0; j<maxVertexCount; ++j) {
        if (!halfEdgeMesh->vertexMap.contains(j)) {
            meshState[j] = {emptyVal, emptyVal, emptyVal};
            continue;
        }
        Vertex* v = halfEdgeMesh->vertexMap[j];
        Vector3f v3f = v->vertex3f;
        meshState[v->id] = {v3f[0], v3f[1], v3f[2]};
    }

    // cout << "mesh faces: " << halfEdgeMesh->faceMap.size() << endl; cout << "mesh vertices: " << halfEdgeMesh->vertexMap.size() << endl; // cout << "mesh state size: " << i << endl;
    // cout << "meshState.size() " << meshState.size() << endl;
    return meshState;
}

// this is for env V1 where the state space is (numV + numF) x 3 and action space is an integer id in range 0-749
pair<float, bool> MeshEnv::step(int action) {

    if (envVersion != 1) return {0, 0};

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
        reward += 0;
        numDNEEdgeCollapses ++;
        if (printSteps) cout << "--- edge id " << action << " never existed" << endl;
    } else {
        pair<int, float> res = halfEdgeMesh->removeEdge(edgeId);
        int errorCode = res.first;
        if (errorCode == 2) {
            // reward += -100; // turns out i NEED this, else the agent picks it
            reward += -21;
            numDeletedEdgeCollapses ++;
            if (printSteps)  cout << "--- edge id " << action << " does not exist (was deleted)" << endl;
        } else if (errorCode == 3) {
            // reward += -50;
            reward += -20;
            numNonManifoldCollapses ++;
            if (printSteps)  cout << "--- edge id " << action << " was not collapsed due to breaking manifoldness" << endl;
        } else {
            float scale = 1000.f;

            numCollapses ++;
            if (printSteps)  cout << "removed edge id " << action << endl;

            float QEMreward = res.second * 10;
            float approxError = approximationError(originalMesh, halfEdgeMesh) * 1000;

            // some stats
            episodeQEMErrorRewards += QEMreward;
            episodeApproxErrorRewards += approxError;
            maxQEMRewardGiven = fmax(QEMreward, maxQEMRewardGiven);
            maxApproximationError = fmax(approxError, maxApproximationError);

            reward = -QEMreward; // since RL tries to maximize the sum of rewards

            // store QEM costs collected
            if (!isTraining)
            {
                scale = 100.f;
                agentQEMCosts.push_back(res.second*scale);
                greedyQEMCosts.push_back(halfEdgeMeshGreedy->greedyQEMStep()*scale);
                randomQEMCosts.push_back(halfEdgeMeshRandom->randomQEMStep()*scale);

//                halfEdgeMeshGreedy->greedyQEMStep();
//                halfEdgeMeshRandom->randomQEMStep();
//                agentQEMCosts.push_back(approxError *scale);
//                greedyQEMCosts.push_back(approximationError(originalMesh, halfEdgeMeshGreedy) *scale);
//                randomQEMCosts.push_back(approximationError(originalMesh, halfEdgeMeshRandom) *scale);

//                cout << "Difference from original mesh vs RL mesh: " << approximationError(originalMesh, halfEdgeMesh) << endl;
//                cout << "Difference from original mesh vs Greedy mesh: " << approximationError(originalMesh, halfEdgeMeshGreedy) << endl;
//                cout << "Difference from original mesh vs Random mesh: " << approximationError(originalMesh, halfEdgeMeshRandom) << endl;
            }
        }
    }

    // terminal conditions
    // - the smallest possible manifold mesh is a tetrahedron, which has 6 edges and 4 faces
    // - the number of faces in the current state is less than equal to what the user wants in the simplified mesh result
    // - (only during training) truncate if the total number of actions/steps (i.e. edge collapses taken is greater than the maxSteps
    int totalCollapses = numCollapses + numNonManifoldCollapses + numDeletedEdgeCollapses + numDNEEdgeCollapses;
    if (isTraining && totalCollapses > maxSteps) {
        isTerminal = true;
    }
    if (getEdgeCount() <= 6 || halfEdgeMesh->faceMap.size() <= finalFaceCount) {
        // reward = 100;
        isTerminal = true;
        reachedRequiredFaces = true;
    }

    episodeRewards += reward;

    return {reward, isTerminal};
}


void MeshEnv::initFromVectors(const vector<Vector3f> &vertices, const vector<Vector3i> &faces) {
    // Copy vertices and faces into internal vector
    _vertices = vertices;
    _faces    = faces;
    halfEdgeMesh = new HalfEdgeMesh();
    halfEdgeMeshGreedy = new HalfEdgeMesh();
    halfEdgeMeshRandom = new HalfEdgeMesh();
}

void MeshEnv::loadFromFile() {
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;

    // QFileInfo info(QString(iniFilePath.c_str()));
    string err;
//    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
//                                info.absoluteFilePath().toStdString().c_str(), (info.absolutePath().toStdString() + "/").c_str(), true);
//    cout << info.absoluteFilePath().toStdString().c_str() << endl;
//    cout << (info.absolutePath().toStdString() + "/").c_str() << endl;
    // "/Users/mohammedk/Documents/Brown/CS2951F/Final Project/MeshSimplificationRL/meshenv/meshes/bunny.obj"
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                                meshFilePath.c_str(),
                                "/Users/mohammedk/Documents/Brown/CS2951F/Final Project/MeshSimplificationRL/AgentV1/meshes/centaur/test", true);

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

    // only build the original mesh once
    if (!isTraining && originalMesh->vertexMap.size() == 0) {
        cout << "Building original mesh" << endl;
        originalMesh->buildHalfEdgeMesh(_vertices, _faces);
    }

    halfEdgeMesh->buildHalfEdgeMesh(_vertices, _faces);
    if (!isTraining) halfEdgeMeshGreedy->buildHalfEdgeMesh(_vertices, _faces);
    if (!isTraining) halfEdgeMeshRandom->buildHalfEdgeMesh(_vertices, _faces);
}

void MeshEnv::saveToFile(const string &filePath) {

    cout << "********** Saving current mesh state as OBJ file... **********\n" << endl;
    string baseFilePath = filePath;

    vector<HalfEdgeMesh*> meshes = {halfEdgeMesh, halfEdgeMeshGreedy, halfEdgeMeshRandom};
    vector<string> names = {"_to_" + to_string(getFaceCount()) + "f_RL.obj",
                           "_to_" + to_string(getFaceCount()) + "f_GreedyQEM.obj",
                           "_to_" + to_string(getFaceCount()) + "f_RandomQEM.obj"};

    for (int i=0; i<meshes.size(); ++i) {
        ofstream outfile;
        outfile.open(baseFilePath + names[i]);

        meshes[i]->createObjFileVerticesFaces(_vertices, _faces);

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
}



