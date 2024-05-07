#include "halfedgemesh.h"


float getDihedralAngle(Edge* edge) {
    //             vj
    //          /   |  \
    //        vk    |   vl
    //          \   |   /
    //             vi
    Vertex* vi = edge->he->vertex;
    Vertex* vj = edge->he->twin->vertex;
    Vertex* vk = edge->he->next->next->vertex;
    Vertex* vl = edge->he->twin->next->next->vertex;

    // normal of left face adj to edge
    Vector3f vij = (vj->vertex3f - vi->vertex3f);
    Vector3f vik = (vk->vertex3f - vi->vertex3f);
    Vector3f faceNormal1 = vij.cross(vik);
    faceNormal1.normalize();

    // normal of right face adj to edge
    Vector3f vil = (vl->vertex3f - vi->vertex3f);
    Vector3f faceNormal2 = vil.cross(vij);
    faceNormal2.normalize();

    // angle between the two face normals is the dihedral angle
    float dot = faceNormal1.dot(faceNormal2);
    float dihedralAngle = std::acos(std::fmin(std::fmax(dot, -1.0), 1.0));

    assert(!std::isnan(dihedralAngle));
    return dihedralAngle;
}

vector<float> getInnerAngles(Edge* edge) {
    //             vj
    //          /   |  \
    //        vk    |   vl
    //          \   |   /
    //             vi
    Vertex* vi = edge->he->vertex;
    Vertex* vj = edge->he->twin->vertex;
    Vertex* vk = edge->he->next->next->vertex;
    Vertex* vl = edge->he->twin->next->next->vertex;

    Vector3f vkj = vj->vertex3f - vk->vertex3f; vkj.normalize();
    Vector3f vki = vi->vertex3f - vk->vertex3f; vki.normalize();
    float alphadot = acos(vkj.dot(vki)); // acos expects input in range [-1, 1], otherwise it produces a nan value
    float alphaij = std::acos(std::fmin(std::fmax(alphadot, -1.0), 1.0));

    Vector3f vlj = vj->vertex3f - vl->vertex3f; vlj.normalize();
    Vector3f vli = vi->vertex3f - vl->vertex3f; vli.normalize();
    float betadot = acos(vlj.dot(vli));
    float betaij = std::acos(std::fmin(std::fmax(betadot, -1.0), 1.0));

    assert(!std::isnan(alphaij));
    assert(!std::isnan(betaij));
    return {alphaij, betaij};
}

vector<float> getEdgeLengthRatios(Edge* edge) {
    //             vj
    //          /   |  \
    //        vk    |   vl
    //          \   |   /
    //             vi
    Vertex* vi = edge->he->vertex;
    Vertex* vj = edge->he->twin->vertex;
    Vertex* vk = edge->he->next->next->vertex;
    Vertex* vl = edge->he->twin->next->next->vertex;

    // ij edge length
    float edgeLength = (vi->vertex3f - vj->vertex3f).norm();

    vector<float> edgeLengthRatios;


    // edge length ratio for traingle ijk
    Vector3f vkj = vj->vertex3f - vk->vertex3f;
    Vector3f vki = vi->vertex3f - vk->vertex3f;
    Vector3f cross = vki.cross(vkj);
    float area = 0.5f * cross.norm(); // area of triangle
    float height = 2.f * area / (float) edgeLength;
    edgeLengthRatios.push_back(edgeLength / (float) height);

    // edge length ratio for traingle ilj
    Vector3f vlj = vj->vertex3f - vl->vertex3f;
    Vector3f vli = vi->vertex3f - vl->vertex3f;
    cross = vlj.cross(vli);
    area = 0.5f * cross.norm(); // area of triangle
    height = 2.f * area / (float) edgeLength;
    edgeLengthRatios.push_back(edgeLength / (float) height);

    assert(!std::isnan(edgeLengthRatios[0]));
    assert(!std::isnan(edgeLengthRatios[1]));

    return edgeLengthRatios;
}

// computes and stores the 5-d feature vector for each edge
// these are the features used by MeshCNN
void HalfEdgeMesh::computeEdgeFeatures() {
    for (auto em : edgeMap) {
        Edge* e = em.second;
        // e->edgeFeatures.clear();

        e->edgeFeatures[0] = getDihedralAngle(e);

        auto innerAngles = getInnerAngles(e);
        e->edgeFeatures[1] = innerAngles[0];
        e->edgeFeatures[2] = innerAngles[1];

        auto edgeLengthRatios = getEdgeLengthRatios(e);
        e->edgeFeatures[3] = edgeLengthRatios[0];
        e->edgeFeatures[4] = edgeLengthRatios[1];
    }
}

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
