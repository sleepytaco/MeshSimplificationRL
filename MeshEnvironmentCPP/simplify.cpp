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

Vector3f HalfEdgeMesh::minimizeEdgeQuadric(Edge* edge) {

    Matrix4f& Qij = edge->Q;

    Vertex* v0 = edge->he->vertex;
    Vertex* v1 = edge->he->twin->vertex;
    Vector3f edgeMP = (v0->vertex3f + v1->vertex3f) / 2.f;

    //if (Qij == Matrix4f::Zero()) return edgeMP;

    Matrix4f K;
    K << Qij(0, 0), Qij(0, 1), Qij(0, 2), Qij(0, 3),
         Qij(1, 0), Qij(1, 1), Qij(1, 2), Qij(1, 3),
         Qij(2, 0), Qij(2, 1), Qij(2, 2), Qij(2, 3),
         0.f, 0.f, 0.f, 1.f;

    // inverse of K doesn not exist
    // then, return the point that has minimum edge cost - between v1, v2, vMP

    auto calcEdgeCost = [Qij] (Vector3f vec) {
        Vector4f x_homo(vec.x(), vec.y(), vec.z(), 1.f); // homogeneous coord of pt x
        return (x_homo.transpose() * Qij * x_homo).norm();
    };

    if (abs(K.determinant()) == 0)  {
        // return edgeMP;
        // Vector4f x_homo(edgeMP.x(), edgeMP.y(), edgeMP.z(), 1.f); // homogeneous coord of pt x
        auto minEdgeCost = calcEdgeCost(edgeMP);
        Vector3f minEdgeCostPoint = edgeMP;

        for (Vector3f x : {v0->vertex3f, v1->vertex3f}) {
            // Vector4f x_homo(x.x(), x.y(), x.z(), 1.f); // homogeneous coord of pt x
            auto edgeCost = calcEdgeCost(x); // x_homo.transpose() * Qij * x_homo;
            if (edgeCost < minEdgeCost) {
                minEdgeCostPoint = x;
            }
        }
        return minEdgeCostPoint; // return the mid point between the edge endpoints
    }

    Matrix4f KInverse = K.inverse();
    Vector4f x = KInverse * Vector4f(0.f, 0.f, 0.f, 1.f);

    return Vector3f(x.x(), x.y(), x.z());
}


float HalfEdgeMesh::updateEdgeQEMCost(Edge* edge) {
    Vertex* i = edge->he->vertex;
    Vertex* j = edge->he->twin->vertex;
    Matrix4f Qij = i->Q + j->Q;
    edge->Q = Qij;

    Vector3f x = minimizeEdgeQuadric(edge);

    auto calcEdgeCost = [Qij] (Vector3f vec) {
        Vector4f x_homo(vec.x(), vec.y(), vec.z(), 1.f); // homogeneous coord of pt x
        return (x_homo.transpose() * Qij * x_homo).norm();
    };

    // Vector4f x_homo(x.x(), x.y(), x.z(), 1.f); // homogeneous coord of pt x
    float edgeCost = calcEdgeCost(x);
    edge->QEMCost = edgeCost;
    return edgeCost;
}

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
    Vertex* v0 = edge->he->vertex;
    Vertex* v1 = edge->he->twin->vertex;
    Vector3f edgeMP = (v0->vertex3f + v1->vertex3f) / 2.f;
    Vertex* collapsedVertex = edgeCollapse(edge, edgeMP); // returns vertex to which the edge was collapsed to
    if (collapsedVertex == nullptr) // if null, then the edge was not collapsed by edgeCollapse func due to violating manifoldness
        return {3, -1};

    // set quadric at new vertex to Qij
    Matrix4f Qij = edge->Q;
    collapsedVertex->Q = Qij;

    // update cost of edges touching new vertex
    HalfEdge* hptr = collapsedVertex->he;
    do {
        Edge* incidentEdge = hptr->edge;
        updateEdgeQEMCost(incidentEdge);
        hptr = hptr->twin->next;
    } while (hptr != collapsedVertex->he);

    return {0, edgeQEMCost};

}

// assigns a QEM cost to each edge in the mesh
void HalfEdgeMesh::initQEMCosts() {
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
    for (auto& em : edgeMap) {
        Edge* edge = em.second;
        updateEdgeQEMCost(edge);
    }
}
