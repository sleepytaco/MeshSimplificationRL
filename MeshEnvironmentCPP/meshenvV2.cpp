#include "meshenv.h"

vector<vector<float>>& MeshEnv::getStateV2() {

    // ------------ This is with the edge features as state space (size num_edges x 5)
    if (envVersion == 2) halfEdgeMesh->computeEdgeFeatures();
    int row = 0;
    for (auto it=halfEdgeMesh->edgeMap.begin(); it != halfEdgeMesh->edgeMap.end(); ++it) {
        meshStateV2[row++] = it->second->edgeFeatures;
    }

    emptyVal = -1; // using -1 instead of 0 made LOT of difference
    while (row != maxEdgeCount) {
        meshStateV2[row++] = {emptyVal, emptyVal, emptyVal, emptyVal, emptyVal};
    }
    // ------------------------------------------------------------------------------


    // ------------- This is with the list of vertices +  faces as the state (752 x 3)

    // cout << meshState.size() << endl; cout << initialVertexCount<< endl; cout << initialFaceCount<< endl;
//    int stateSpaceSize = maxVertexCount + maxFaceCount;

//    int row = 0;
//    for (int j=0; j<maxVertexCount; ++j) {
//        if (!halfEdgeMesh->vertexMap.contains(j)) {
//            continue;
//        }
//        Vertex* v = halfEdgeMesh->vertexMap[j];
//        Vector3f v3f = v->vertex3f;
//        meshStateV2[row++] = {v3f[0], v3f[1], v3f[2]};
//    }

//    for (int j=0; j<maxFaceCount; ++j) {
//        if (!halfEdgeMesh->faceMap.contains(j)) {
//            continue;
//        }
//        Face* f = halfEdgeMesh->faceMap[j];
//        Vector3i f3i = halfEdgeMesh->getVertexIdsFromFace(f);
//        meshStateV2[row++] = {(float)f3i[0], (float)f3i[1], (float)f3i[2]};
//    }

//    while (row < stateSpaceSize) {
//        meshStateV2[row++] = {emptyVal, emptyVal, emptyVal};
//    }

    // -------------------------------------------------------------------------------

    return meshStateV2;
}

void MeshEnv::resetV2() {
    reset();
}

pair<float, bool> MeshEnv::stepV2(Vector3f xyz) {

    // if (!isTraining) cout << "Collapse #" << numCollapses << ": Model predicted edge collapse location: " << xyz.transpose() << endl;

    // here action == xyz coord of edge to collapse
    int edgeId = 0;
    float reward = 0;
    bool isTerminal = false;

    float minDist = numeric_limits<float>::max();
    for (auto em : halfEdgeMesh->edgeMap) {
        Edge* edge = em.second;
        Vector3f edgeMP = (edge->he->vertex->vertex3f + edge->he->twin->vertex->vertex3f) / (float) 2.f;

        float dist = (edgeMP - xyz).norm();
        if (dist < minDist) {
            minDist = dist;
            edgeId = edge->id;
        }
    }

    int action = edgeId;

    reward += -minDist; // penalize if the agent's "point" is far from the closest mesh vertex...?
    minAgentDistFromMesh = fmin(minAgentDistFromMesh, minDist);

    // cout << "minDist: " << minDist << endl;

    /*
     * halfEdgeMesh->removeEdge(edgeId) returns <error code, QEM cost for that edge collapse>
     * possible error codes:
     *  0 - no error, edge collapse successful
     *  1 - edge does never existed at all
     *  2 - edge does not exist (was deleted)
     *  3 - edge was not collapsed because it breaks manifoldness property of the mesh
     */
    pair<int, float> res = halfEdgeMesh->removeEdge(edgeId);
    int errorCode = res.first;
    if (errorCode == 1) {
        numDNEEdgeCollapses ++;
        if (printSteps)  cout << "--- edge id " << action << " does not exist (never existed)" << endl;
    } else if (errorCode == 2) {
        numDeletedEdgeCollapses ++;
        if (printSteps)  cout << "--- edge id " << action << " does not exist (was deleted)" << endl;
    } else if (errorCode == 3) {
        // reward += -50; // bad

        // giving small penality reward for breaking manifoldess helps a lOT as well, becoz i only train it for <10M max
        // the agent does not really leanrn it fully within that.
        // non-manifoldness related collapses could also have smallest QEM rewards for a state! it the agent picks it we can simply ignore, much as the original QEM paper
        // reward += -20;
        float nonManifoldReward = res.second;
        float penalty = -nonManifoldReward;
        float rewardGiven = -nonManifoldReward + penalty;

        // maxNonManifoldQEMReward = max(maxNonManifoldQEMReward, nonManifoldReward);
        reward += rewardGiven; // only smol non-manifold-ness?
        maxNonManifoldQEMReward = fmax(maxNonManifoldQEMReward, -rewardGiven);
        numNonManifoldCollapses ++;

        if (printSteps)  cout << "--- edge id " << action << " was not collapsed due to breaking manifoldness" << endl;
    } else {

        numCollapses ++;
        if (printSteps)  cout << "removed edge id " << action << endl;

        float QEMreward = res.second; // 1 * 10;

        // some stats
        episodeQEMErrorRewards += QEMreward;
        maxQEMRewardGiven = fmax(QEMreward, maxQEMRewardGiven);

        // halfEdgeMeshGreedy->greedyQEMStep();
        // float idealApproxError = approximationError(halfEdgeMeshGreedy, halfEdgeMesh); // compare RL agent's mesh and Greedy QEM agent's mesh energy
        float approxError = approximationError(originalMesh, halfEdgeMesh) * 100;
        episodeApproxErrorRewards += approxError;
        maxApproximationError = fmax(approxError, maxApproximationError);

        reward += -QEMreward -approxError; // -idealApproxError; // -approxError; // since RL tries to maximize the sum of rewards


        // store QEM costs collected
        if (!isTraining)
        {
            float scale = 1000.f;
            float approxError = approximationError(originalMesh, halfEdgeMesh) * scale;

//            agentQEMCosts.push_back(res.second*scale);
//            greedyQEMCosts.push_back(halfEdgeMeshGreedy->greedyQEMStep()*scale);
//            randomQEMCosts.push_back(halfEdgeMeshRandom->randomQEMStep()*scale);

            halfEdgeMeshGreedy->greedyQEMStep();
            halfEdgeMeshRandom->randomQEMStep();
            agentQEMCosts.push_back(approxError *scale);
            greedyQEMCosts.push_back(approximationError(originalMesh, halfEdgeMeshGreedy) *scale);
            randomQEMCosts.push_back(approximationError(originalMesh, halfEdgeMeshRandom) *scale);
        }
    }


    // terminal conditions
    // - the smallest possible manifold mesh is a tetrahedron, which has 6 edges and 4 faces
    // - the number of faces in the current state is less than equal to what the user wants in the simplified mesh result
    // - (only during training) truncate if the total number of actions/steps (i.e. edge collapses taken is greater than the maxSteps
    int totalCollapses = numCollapses + numNonManifoldCollapses + numDeletedEdgeCollapses + numDNEEdgeCollapses;
    maxSteps = 767; // (int) ((maxFaceCount - finalFaceCount) / 2) + 50;
    // if (totalCollapses > 150) reward += -10;
    if (totalCollapses > maxSteps) {
        // reward += -200;
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

