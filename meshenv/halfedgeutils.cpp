#include "halfedgemesh.h"

// travel around the Face and get the three vertex IDs belonging to this face
Vector3i HalfEdgeMesh::getVertexIdsFromFace(Face* f) {
    Vector3i vertexIDList(-1, -1, -1);
    HalfEdge* h = f->he;
    int i=0;
    do {
        vertexIDList[i++] = h->vertex->id;
        h = h->next;
    } while (h != f->he);
    return vertexIDList;
}

// travel around the halfedges connected to this vertex and get the face IDs that this vertex is a part of
vector<int> HalfEdgeMesh::getFaceIdsFromVertex(Vertex* v) {
    vector<int> faceIDList;
    HalfEdge* h = v->he;
    int i=0;
    do {
        faceIDList.push_back(h->face->id);
        h = h->twin->next;
    } while (h != v->he);
    return faceIDList;
}

Vector3f HalfEdgeMesh::getFaceNormal(Face* face) {
    // visualize:
    //         v1
    //      /  |
    //  v2  f0 |
    //    \    |
    //        v0
    HalfEdge* heV0V1 = face->he;
    Vertex* v0 = heV0V1->vertex;
    Vertex* v1 = heV0V1->next->vertex;
    Vertex* v2 = heV0V1->next->next->vertex;

    Vector3f v01 = (v1->vertex3f - v0->vertex3f);
    Vector3f v02 = (v2->vertex3f - v0->vertex3f);

    Vector3f faceNormal = v01.cross(v02);
    faceNormal.normalize();
    return faceNormal;
}

Vector3f HalfEdgeMesh::getVertexNormal(Vertex* vertex) {
    int n=0;
    HalfEdge* h = vertex->he;
    Vector3f faceNormalSum = Vector3f(0.f, 0.f, 0.f);
    do {
        n += 1;
        faceNormalSum += getFaceNormal(h->face);
        h = h->twin->next;
    } while (h != vertex->he);
    return faceNormalSum/(float)n;
}

void HalfEdgeMesh::addHalfEdges(vector<HalfEdge*> hes) {
    for (HalfEdge* he : hes) {
        halfEdgeMap[he->id] = he;
    }
}

void HalfEdgeMesh::addEdges(vector<Edge*> es) {
    for (Edge* e : es) {
        edgeMap[e->id] = e;
    }
}

void HalfEdgeMesh::addFaces(vector<Face*> fs) {
    for (Face* f : fs) {
        faceMap[f->id] = f;
    }
}

void HalfEdgeMesh::addVertices(vector<Vertex*> vs) {
    for (Vertex* v : vs) {
        vertexMap[v->id] = v;
    }
}

void HalfEdgeMesh::deleteHalfEdges(vector<HalfEdge*> hes) {
    for (HalfEdge* he : hes) {
        auto it = halfEdgeMap.find(he->id);
        delete halfEdgeMap[he->id]; // delete the ptr
        halfEdgeMap.erase(it); // erase the entry from the map
    }
}

void HalfEdgeMesh::deleteEdges(vector<Edge*> es) {
    for (Edge* e : es) {
        auto it = edgeMap.find(e->id);
        delete edgeMap[e->id]; // delete the ptr
        edgeMap.erase(it); // erase the entry from the map
    }
}

void HalfEdgeMesh::deleteFaces(vector<Face*> fs) {
    for (Face* f : fs) {
        auto it = faceMap.find(f->id);
        delete faceMap[f->id]; // delete the ptr
        faceMap.erase(it); // erase the entry from the map
    }
}

void HalfEdgeMesh::deleteVertices(vector<Vertex*> vs) {
    for (Vertex* v : vs) {
        auto it = vertexMap.find(v->id);
        delete vertexMap[v->id]; // delete the ptr
        vertexMap.erase(it); // erase the entry from the map
    }
}
