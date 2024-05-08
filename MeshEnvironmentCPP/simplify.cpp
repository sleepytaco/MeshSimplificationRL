#include "halfedgemesh.h"

Matrix4f HalfEdgeMesh::getVertexQuadric(Vertex *v) {
    Matrix4f Q = Matrix4f::Zero();
    HalfEdge* hptr = v->he;
    do {
        Q += hptr->face->getFaceQuadric();
        hptr = hptr->twin->next;
    } while (hptr != v->he);
    return Q;
}

float HalfEdgeMesh::updateEdgeQEMCost(Edge* edge) {
    Vertex* i = edge->he->vertex;
    Vertex* j = edge->he->twin->vertex;
    Matrix4f Qij = i->Q + j->Q;
    edge->Q = Qij;

    // Vector3f x = minimizeEdgeQuadric(edge);
    Vector3f edgeMP = (i->vertex3f + j->vertex3f) / 2.f;

    auto calcEdgeCost = [Qij] (Vector3f vec) {
        Vector4f x_homo(vec.x(), vec.y(), vec.z(), 1.f); // homogeneous coord of pt x
        return (x_homo.transpose() * Qij * x_homo).norm();
    };

    // Vector4f x_homo(x.x(), x.y(), x.z(), 1.f); // homogeneous coord of pt x
    float edgeCost = calcEdgeCost(edgeMP);
    edge->QEMCost = edgeCost;
    return edgeCost;
}


void HalfEdgeMesh::addEdgeToQueue(multiset<pair<float, Edge*>, CustomCompare> &priorityQueue, Edge* edge) {
    Vertex* i = edge->he->vertex;
    Vertex* j = edge->he->twin->vertex;
    Matrix4f Qij = i->Q + j->Q;
    edge->Q = Qij;

    // Vector3f x = minimizeEdgeQuadric(edge);
    Vector3f edgeMP = (i->vertex3f + j->vertex3f) / 2.f;


    auto calcEdgeCost = [Qij] (Vector3f vec) {
        Vector4f x_homo(vec.x(), vec.y(), vec.z(), 1.f); // homogeneous coord of pt x
        return (x_homo.transpose() * Qij * x_homo).norm();
    };

    // Vector4f x_homo(x.x(), x.y(), x.z(), 1.f); // homogeneous coord of pt x
    float edgeCost = calcEdgeCost(edgeMP); //x_homo.transpose() * Qij * x_homo;
    edge->QEMCost = edgeCost;
    priorityQueue.insert({calcEdgeCost(edgeMP), edge});
}

void HalfEdgeMesh::updateEdgeInQueue(multiset<pair<float, Edge*>, CustomCompare> &priorityQueue, Edge* edgeToUpdate) {
    // remove edge from pq
    auto it = priorityQueue.begin();
    bool removedFromQueue = false;
    while (it != priorityQueue.end()) {
        if (it->second->id == edgeToUpdate->id) {
            priorityQueue.erase(it);
            removedFromQueue = true;
            break;
        }
        ++it;
    }

    if (!removedFromQueue) { // actually this is fine, as greedy QEM can ignore edges that lead to manifoldness, but its ok if it adds it back
        // cout << edgeToUpdate->id << endl;
    }
    // assert(removedFromQueue);

    // add edge back to queue
    addEdgeToQueue(priorityQueue, edgeToUpdate);
}

// assigns a QEM cost to each edge in the mesh
void HalfEdgeMesh::initQEMCosts(bool greedyQEMAgent) {
    // step 1. compute Q for each triangle/face
//    for (auto& fm : faceMap) {
//        Face* face = fm.second;
//        face->getFaceQuadric(); // computes and stores quadric for face within the face object
//    }

    // step 2. compute Q for each vertex to the sum of Qs of incident triangles
    for (auto& vm : vertexMap) {
        Vertex* vertex = vm.second;
        vertex->Q = getVertexQuadric(vertex); // compute and stores quadric for vertex within the vertex object
    }

    // step 3. for each edge find pt x minimizing error, set cost to Q(x)
    // custom comparator for multiset priority queue
    for (auto& em : edgeMap) {
        Edge* edge = em.second;
        updateEdgeQEMCost(edge);
        addEdgeToQueue(priorityQueue, edge);
        // cout << "added edge with id: " << edge->id << endl;
    }

}

// RL agent's step function
pair<int, float> HalfEdgeMesh::removeEdge(int edgeId) {
    /*
     * returns <error code, QEM cost for that edge collapse>
     * possible error codes:
     *  0 - no error, edge collapse successful
     *  1 - edge does never existed at all
     *  2 - edge does not exist (was deleted)
     *  3 - edge was not collapsed because it breaks manifoldness property of the mesh
     */
    if (edgeMap.find(edgeId) == edgeMap.end()) // NEED THIS check as edgeCollapse operation deletes some surrounding edges from the edgeMap
        return {2, -1};

    Edge* edge = edgeMap[edgeId];
    float edgeQEMCost = edge->QEMCost;
    Matrix4f edgeQij = edge->Q;

    Vertex* v0 = edge->he->vertex;
    Vertex* v1 = edge->he->twin->vertex;
    Vector3f edgeMP = (v0->vertex3f + v1->vertex3f) / 2.f;
    Vertex* collapsedVertex = edgeCollapse(edge, edgeMP); // returns vertex to which the edge was collapsed to
    if (collapsedVertex == nullptr) { // if null, then the edge was not collapsed by edgeCollapse func due to violating manifoldness
        return {3, edgeQEMCost};
    }

    // set quadric at new vertex to Qij
    Matrix4f Qij = edgeQij;
    collapsedVertex->Q = Qij;

    // update cost of edges touching new vertex
    HalfEdge* hptr = collapsedVertex->he;
    do {
        Edge* incidentEdge = hptr->edge;
        updateEdgeQEMCost(incidentEdge);
        hptr = hptr->twin->next;
    } while (hptr != collapsedVertex->he);

    QEMCostsPerStep.push_back(edgeQEMCost*10);

    return {0, edgeQEMCost};

}

float HalfEdgeMesh::greedyQEMStep() {

    int nonManifoldCollapses = 0;

    // run until we reach target num of faces
    float QEMCost = 666;
    int numCollapses = 0;
    while (numCollapses<1 && priorityQueue.size() > 0) {
        // auto minEdge = *priorityQueue.begin();
        QEMCost = priorityQueue.begin()->first;
        Edge* edge = priorityQueue.begin()->second;
        Matrix4f edgeQij = edge->Q;
        priorityQueue.erase(priorityQueue.begin());

        if (edgeMap.find(edge->id) == edgeMap.end()) {// NEED THIS check as edgeCollapse operation deletes some surrounding edges from the edgeMap
            nonManifoldCollapses++;
            continue;
        }

        Vertex* i = edge->he->vertex;
        Vertex* j = edge->he->twin->vertex;
        Vector3f edgeMP = (i->vertex3f + j->vertex3f) / 2.f;

        Vertex* collapsedVertex = edgeCollapse(edge, edgeMP); // returns vertex to which the edge was collapsed to
        if (collapsedVertex == nullptr) // if null, then the edge was not collapsed by edgeCollapse func due to violating manifoldness
            continue;

        // set quadric at new vertex to Qij
        Matrix4f Qij = edgeQij;
        collapsedVertex->Q = Qij;

        // update cost of edges touching new vertex
        HalfEdge* hptr = collapsedVertex->he;
        do {
            Edge* incidentEdge = hptr->edge;
            updateEdgeInQueue(priorityQueue, incidentEdge);
            hptr = hptr->twin->next;
        } while (hptr != collapsedVertex->he);

        numCollapses ++;
    }

    QEMCostsPerStep.push_back(QEMCost*10);
    int prev = (nonManifoldCollapsesPerStep.size() == 0) ? 0 : nonManifoldCollapsesPerStep[nonManifoldCollapsesPerStep.size()-1];
    nonManifoldCollapsesPerStep.push_back(prev + nonManifoldCollapses);

    return QEMCost;
}

#include <random>
float HalfEdgeMesh::randomQEMStep() {

    int nonManifoldCollapses = 0;

    // pick random edge ID
//     int rand_num=rand()%99+1; // produces numbers from 1-99
//    int edgeItemNumber = arc4random()%750;
//    while (edgeMap.find(edgeItemNumber) == edgeMap.end()) {
//        edgeItemNumber = arc4random()%750;
//    }

//        int edgeItemNumber = arc4random()%edgeMap.size();
//        auto it = edgeMap.begin();
//        for (int i=0; i<edgeItemNumber; ++i) {
//            ++it; // move the edgeMap iterator forward
//            edgeItemNumber--;
//        }

    // Seed the random number generator
    random_device rd;
    mt19937 g(rd());
    // Shuffle the elements of the map
    vector<pair<int, Edge*>> vec(edgeMap.begin(), edgeMap.end());
    shuffle(vec.begin(), vec.end(), g);


    // run until we reach target num of faces
    float QEMCost = 666;
    int numCollapses = 0;
    int i=0;
    while (numCollapses<1) {

        int edgeItemNumber = vec[i++].first;

//        if (edgeMap.find(edge->id) == edgeMap.end()) // NEED THIS check as edgeCollapse operation deletes some surrounding edges from the edgeMap
//            continue;

        Edge* edge = edgeMap[edgeItemNumber]; // it->second; //
        Matrix4f edgeQij = edge->Q;
        QEMCost = edge->QEMCost;

        Vertex* i = edge->he->vertex;
        Vertex* j = edge->he->twin->vertex;
        Vector3f edgeMP = (i->vertex3f + j->vertex3f) / 2.f;

        Vertex* collapsedVertex = edgeCollapse(edge, edgeMP); // returns vertex to which the edge was collapsed to
        if (collapsedVertex == nullptr) { // if null, then the edge was not collapsed by edgeCollapse func due to violating manifoldness
            nonManifoldCollapses++;
            continue;
        }

        // set quadric at new vertex to Qij
        Matrix4f Qij = edgeQij;
        collapsedVertex->Q = Qij;

        // update cost of edges touching new vertex
        HalfEdge* hptr = collapsedVertex->he;
        do {
            Edge* incidentEdge = hptr->edge;
            updateEdgeQEMCost(incidentEdge);
            hptr = hptr->twin->next;
        } while (hptr != collapsedVertex->he);

        numCollapses ++;
    }

    QEMCostsPerStep.push_back(QEMCost*10);
    int prev = (nonManifoldCollapsesPerStep.size() == 0) ? 0 : nonManifoldCollapsesPerStep[nonManifoldCollapsesPerStep.size()-1];
    nonManifoldCollapsesPerStep.push_back(prev + nonManifoldCollapses);
    return QEMCost;
}
