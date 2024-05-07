#include "meshenv.h"

vector<vector<float>>& MeshEnv::getStateV2() {
    if (envVersion == 2) halfEdgeMesh->computeEdgeFeatures();

    int row = 0;
    for (auto it=halfEdgeMesh->edgeMap.begin(); it != halfEdgeMesh->edgeMap.end(); ++it) {
        meshStateV2[row++] = it->second->edgeFeatures;
    }

    emptyVal = 0;
    while (row != maxEdgeCount) {
        meshStateV2[row++] = {emptyVal, emptyVal, emptyVal, emptyVal, emptyVal};
    }

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
        // reward += -50;
         reward += -100;
         // reward += -500;
         // isTerminal = true; // try this with TD3 too lol
        numNonManifoldCollapses ++;
        if (printSteps)  cout << "--- edge id " << action << " was not collapsed due to breaking manifoldness" << endl;
    } else {

        numCollapses ++;
        if (printSteps)  cout << "removed edge id " << action << endl;

        float QEMreward = res.second * 10;
        // float approxError = approximationError(originalMesh, halfEdgeMesh) * 1000;

        // some stats
        episodeQEMErrorRewards += QEMreward;
        maxQEMRewardGiven = fmax(QEMreward, maxQEMRewardGiven);

//        originalMesh->greedyQEMStep();
//        float idealApproxError = approximationError(originalMesh, halfEdgeMesh);
//        episodeApproxErrorRewards += idealApproxError;
//        maxApproximationError = fmax(idealApproxError, maxApproximationError);

        reward += -QEMreward; // -idealApproxError; // -approxError; // since RL tries to maximize the sum of rewards


        // store QEM costs collected
        if (!isTraining)
        {
            float scale = 10.f;
            agentQEMCosts.push_back(res.second*scale);
            greedyQEMCosts.push_back(halfEdgeMeshGreedy->greedyQEMStep()*scale);
            randomQEMCosts.push_back(halfEdgeMeshRandom->randomQEMStep()*scale);

//            halfEdgeMeshGreedy->greedyQEMStep();
//            halfEdgeMeshRandom->randomQEMStep();
//            agentQEMCosts.push_back(approxError *scale);
//            greedyQEMCosts.push_back(approximationError(originalMesh, halfEdgeMeshGreedy) *scale);
//            randomQEMCosts.push_back(approximationError(originalMesh, halfEdgeMeshRandom) *scale);
        }
    }


    // terminal conditions
    // - the smallest possible manifold mesh is a tetrahedron, which has 6 edges and 4 faces
    // - the number of faces in the current state is less than equal to what the user wants in the simplified mesh result
    // - (only during training) truncate if the total number of actions/steps (i.e. edge collapses taken is greater than the maxSteps
    int totalCollapses = numCollapses + numNonManifoldCollapses + numDeletedEdgeCollapses + numDNEEdgeCollapses;
    maxSteps = (int) ((maxFaceCount - finalFaceCount) / 2) + 50;
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

