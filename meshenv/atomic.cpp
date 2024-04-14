// Contains atomic operations for the HalfEdgeMesh DS: edge split, edge flip, edge collapse

#include "halfedgemesh.h"

bool HalfEdgeMesh::isManifoldAfterEdgeCollapse(Edge* edge, Vector3f collapsePoint) {
    // check if collapsing this edge will lead to non-manifoldness in the mesh
    Vertex* v0 = edge->he->vertex;
    Vertex* v1 = edge->he->twin->vertex;

    // CONDITION 1 - Edge endpoints have more than two shared neighbors
    unordered_set<Vertex*> v0neighborhoorVertices;
    HalfEdge* hptr = v0->he;
    do {
        Vertex* neighborVertex = hptr->twin->vertex;
        v0neighborhoorVertices.insert(neighborVertex);
        hptr = hptr->twin->next;
    } while (hptr != v0->he);
    unordered_set<Vertex*> v1neighborhoorVertices;
    hptr = v1->he;
    do {
        Vertex* neighborVertex = hptr->twin->vertex;
        v1neighborhoorVertices.insert(neighborVertex);
        hptr = hptr->twin->next;
    } while (hptr != v1->he);

    vector<Vertex*> sharedNeighbors;
    for (Vertex* v : v1neighborhoorVertices) {
        if (v0neighborhoorVertices.find(v) !=  v0neighborhoorVertices.end())
            sharedNeighbors.push_back(v);
        if (sharedNeighbors.size() > 2)
            return false;
    }

    // CONDITION 2 - Edge endpoints have a shared neighbor w/ degree 3
    for (Vertex* v : sharedNeighbors) {
        if (v->getInDegree() == 3)
            return false;
    }

    // CONDITION 3- Check if edge collapse leads to triangle flip
    auto calculateNormal = [](vector<Vector3f> vertices) {
        Vector3f v0 = vertices[0]; Vector3f v1 = vertices[1]; Vector3f v2 = vertices[2];
        Vector3f v01 = (v1 - v0);
        Vector3f v02 = (v2 - v0);
        Vector3f faceNormal = v01.cross(v02);
        faceNormal.normalize();
        return faceNormal;
    };
    Vector3f vMp = collapsePoint; // (v0->vertex3f + v1->vertex3f) / (float) 2.f; // TODO: this can be anything in general
    Face* f0 = edge->he->face;
    Face* f1 = edge->he->twin->face;
    vector<int> faceIDsAroundV0 = getFaceIdsFromVertex(v0);
    for (int faceID : faceIDsAroundV0) {
        if (faceMap[faceID] == f0 || faceMap[faceID] == f1) continue; // ignore these faces as they will get deleted anyway after edge collapse
        Face* f = faceMap[faceID];
        Vector3f normalBeforeFlip = getFaceNormal(f);

        vector<Vector3f> vertexPositions;
        for (int vID : getVertexIdsFromFace(f)) {
            if (vID == v0->id) vertexPositions.push_back(vMp);
            else vertexPositions.push_back(vertexMap[vID]->vertex3f);
        }

        Vector3f normalAfterFlip = calculateNormal(vertexPositions);

        if (normalBeforeFlip.dot(normalAfterFlip) < 0) {
            // cout << "triangle normal flips" << endl;
            return false;
        }

    }

    vector<int> faceIDsAroundV1 = getFaceIdsFromVertex(v1);
    for (int faceID : faceIDsAroundV1) {
        if (faceMap[faceID] == f0 || faceMap[faceID] == f1) continue; // ignore these faces as they will get deleted anyway after edge collapse
        Face* f = faceMap[faceID];
        Vector3f normalBeforeFlip = getFaceNormal(f);

        vector<Vector3f> vertexPositions;
        for (int vID : getVertexIdsFromFace(f)) {
            if (vID == v0->id) vertexPositions.push_back(vMp);
            else vertexPositions.push_back(vertexMap[vID]->vertex3f);
        }

        Vector3f normalAfterFlip = calculateNormal(vertexPositions);

        if (normalBeforeFlip.dot(normalAfterFlip) < 0) {
            // cout << "triangle normal flips" << endl;
            return false;
        }

    }
    return true;
}

Vertex* HalfEdgeMesh::edgeCollapse(Edge* edge, Vector3f collapsePoint) {
    // visualize:
    //  v5 -------   v2  -------- v7
    //    \  f2  /  f0    \  f4 /
    //       v0  --------- v1
    //    / f3  \   f1   / f5  \
    //  v6 -----   v3 -------- v8
    // vMP is midpoint/collapsePoint of edge from v0 to v1
    if (!isManifoldAfterEdgeCollapse(edge, collapsePoint)) {
        // cout << "edge collapse results in non-manifoldness" << endl;
        return nullptr;
    }

    HalfEdge* heV0V1 = edge->he; HalfEdge* heV1V0 = heV0V1->twin;
    HalfEdge* heV1V2 = heV0V1->next; HalfEdge* heV2V1 = heV1V2->twin;
    HalfEdge* heV2V0 = heV1V2->next; HalfEdge* heV0V2 = heV2V0->twin;
    HalfEdge* heV0V3 = heV1V0->next; HalfEdge* heV3V0 = heV0V3->twin;
    HalfEdge* heV3V1 = heV0V3->next; HalfEdge* heV1V3 = heV3V1->twin;

    Vertex* v0 = heV0V1->vertex; // this vertex will be deleted
    Vertex* v1 = heV1V0->vertex; // this vertex will be deleted
    Vertex* v2 = heV2V1->vertex;
    Vertex* v3 = heV3V0->vertex;

    Edge* eV0V1 = edge; // this edge will deleted
    Edge* eV2V0 = heV2V0->edge;
    Edge* eV0V3 = heV0V3->edge;
    Edge* eV1V2 = heV1V2->edge; // this edge will be deleted
    Edge* eV3V1 = heV3V1->edge; // this edge will be deleted

    heV2V1->edge = eV2V0;
    heV1V3->edge = eV0V3;
    eV2V0->he = heV0V2;
    eV0V3->he = heV3V0;

    Face* f0 = heV0V1->face;
    Face* f1 = heV1V0->face;

    // Vector3f mp = (v0->vertex3f + v1->vertex3f)/2.f; // edge mid point - TODO: this can be anything in general, not necessarily the midpoint
    v0->vertex3f = collapsePoint; // update v0 to be the MIDPOINT of the edge that is going to be collapsed
    // Vertex* vMP = new Vertex(vertexIdCounter++, collapsePoint);
    // Vertex* vMP = new Vertex(v0->id, collapsePoint);
    // addVertices({vMP});

    HalfEdge* hptr = v1->he;
    do { // set all of the edges connected to v1 to refer to v0 as their start vertex
        hptr->vertex = v0; // vMP;
        hptr = hptr->twin->next;
    } while(hptr != v1->he);
    hptr = v0->he;
    do { // set all of the edges connected to v1 to refer to v0 as their start vertex
        hptr->vertex = v0; //vMP;
        hptr = hptr->twin->next;
    } while(hptr != v0->he);

    heV0V2->twin = heV2V1;
    heV2V1->twin = heV0V2;
    heV1V3->twin = heV3V0;
    heV3V0->twin = heV1V3;

    // vMP->he = heV0V2;
    v0->he = heV0V2;
    v1->he = heV1V3; // will be deleted
    v2->he = heV2V1;
    v3->he = heV3V0;

    deleteHalfEdges({heV0V1, heV1V0, heV1V2, heV2V0, heV0V3, heV3V1});
    deleteFaces({f0, f1});
    deleteEdges({eV0V1, eV1V2, eV3V1});
    // deleteVertices({v0, v1});
    deleteVertices({v1});
    // addVertices({vMP});
    return v0; //vMP; // return the Vertex* to which the edge was collapsed to
}
