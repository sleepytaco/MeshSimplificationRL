#include "halfedgemesh.h"
#include <cassert>

HalfEdgeMesh::HalfEdgeMesh() {}

HalfEdgeMesh::~HalfEdgeMesh() {
    cleanUpHalfEdgeMesh();
}

void HalfEdgeMesh::buildHalfEdgeMesh(const vector<Vector3f> &vertices, const vector<Vector3i> &faces) {

    cleanUpHalfEdgeMesh();

    unordered_map<int, unordered_map<int, HalfEdge*>> halfEdgeMap2D; // maps (vertex1ID, vertex2ID) --> HalfEdge from Vertex1ID to Vertex2ID
    unordered_map<int, unordered_map<int, Edge*>> edgeMap2D; // maps (vertex1ID, vertex2ID) --> HalfEdge from Vertex1ID to Vertex2ID

    for (int id=0; id<vertices.size(); ++id) {
        int vertexID = id; // vertexIdCounter++;
        vertexMap[vertexID] = new Vertex(vertexID, vertices[id]);
    }

    for (int id=0; id<faces.size(); ++id) {
        int faceID = id; //faceIdCounter++;
        faceMap[faceID] = new Face(faceID);
        Vector3i face = faces[faceID];
        faceMap[faceID]->face3i = face;

        vector<Vertex*> vertexBuffer(3, nullptr);
        for (int i=0; i<3; ++i) {
            vertexBuffer[i] = vertexMap[face[i]];
        }

        vector<HalfEdge*> heCCWBuffer(3, nullptr); // 3 half edges going in CCW direction
        vector<Edge*> edgeCCWBuffer(3, nullptr); // 3 full edges going in CCW direction
        for (int i=0; i<3; ++i) {
            Vertex* v1 = vertexMap[face[i]];
            Vertex* v2 = vertexMap[face[(i+1)%3]];

            // check if half edge going from v1 to v2 exists
            if (halfEdgeMap2D.find(v1->id) != halfEdgeMap2D.end() && halfEdgeMap2D[v1->id].find(v2->id) != halfEdgeMap2D[v1->id].end()) {
                // then don't create it
                halfEdgeMap2D[v1->id][v2->id]->setFace(faceMap[id]);
            } else {
                // else, create the he and set it to originate from v1
                int halfEdgeID = halfEdgeIdCounter++;
                halfEdgeMap2D[v1->id][v2->id] = new HalfEdge(halfEdgeID, v1, faceMap[id]);
                halfEdgeMap[halfEdgeID] = halfEdgeMap2D[v1->id][v2->id];
                // v1->he = halfEdgeMap2D[v1->id][v2->id];
                v1->setHalfEdge(halfEdgeMap2D[v1->id][v2->id]);
            }

            if (edgeMap2D.find(v1->id) != edgeMap2D.end() && edgeMap2D[v1->id].find(v2->id) != edgeMap2D[v1->id].end()) {
                // edge going from v1->v2
                edgeCCWBuffer[i] = edgeMap2D[v1->id][v2->id];
            } else if (edgeMap2D.find(v2->id) != edgeMap2D.end() && edgeMap2D[v2->id].find(v1->id) != edgeMap2D[v2->id].end()) {
                // edge going from v2->v1
                edgeCCWBuffer[i] = edgeMap2D[v2->id][v1->id];
            } else {
                // edge does not exist between v1 and v2 (in any direction), so create a new one
                int edgeID = edgeIdCounter++;
                edgeMap2D[v1->id][v2->id] = new Edge(edgeID);
                edgeMap[edgeID] = edgeMap2D[v1->id][v2->id];
                edgeMap[edgeID]->he = halfEdgeMap2D[v1->id][v2->id];
                edgeCCWBuffer[i] = edgeMap2D[v1->id][v2->id];
            }

            heCCWBuffer[i] = halfEdgeMap2D[v1->id][v2->id];
            heCCWBuffer[i]->edge = edgeCCWBuffer[i];
        }

        vector<HalfEdge*> heCWBuffer(3, nullptr); // 3 half edges going in CW direction
        Vector3i faceCW = face;
        faceCW[1] = face[2]; faceCW[2] = face[1];
        for (int i=0; i<3; ++i) {
            Vertex* v1 = vertexMap[faceCW[i]];
            Vertex* v2 = vertexMap[faceCW[(i+1)%3]];

            // check if half edge going from v1 to v2 exists
            if (halfEdgeMap2D.find(v1->id) != halfEdgeMap2D.end() && halfEdgeMap2D[v1->id].find(v2->id) != halfEdgeMap2D[v1->id].end()) {
                // then don't create it
            } else {
                // else, create the he and set it to originate from v1
                int halfEdgeID = halfEdgeIdCounter++;
                halfEdgeMap2D[v1->id][v2->id] = new HalfEdge(halfEdgeID, v1);
                halfEdgeMap[halfEdgeID] = halfEdgeMap2D[v1->id][v2->id];
                // v1->he = halfEdgeMap2D[v1->id][v2->id];
                v1->setHalfEdge(halfEdgeMap2D[v1->id][v2->id]);
            }

            heCWBuffer[i] = halfEdgeMap2D[v1->id][v2->id];
        }

        // set the next pointers for the half edges
        heCCWBuffer[0]->next = heCCWBuffer[1];
        heCCWBuffer[1]->next = heCCWBuffer[2];
        heCCWBuffer[2]->next = heCCWBuffer[0];

        // set the twin pointers for the half edges
        heCCWBuffer[0]->twin = heCWBuffer[2];
        heCCWBuffer[1]->twin = heCWBuffer[1];
        heCCWBuffer[2]->twin = heCWBuffer[0];
        heCWBuffer[0]->twin = heCCWBuffer[2];
        heCWBuffer[1]->twin = heCCWBuffer[1];
        heCWBuffer[2]->twin = heCCWBuffer[0];

        // set the Edges of the twin ptrs to be the same
        heCCWBuffer[0]->twin->edge = heCCWBuffer[0]->edge;
        heCCWBuffer[1]->twin->edge = heCCWBuffer[1]->edge;
        heCCWBuffer[2]->twin->edge = heCCWBuffer[2]->edge;

        faceMap[faceID]->he = heCCWBuffer[0];

    }

    // calc the size of halfEdgeMap2D
//    int numHalfEdges = 0;
//    for (const auto& he : halfEdgeMap2D) numHalfEdges += he.second.size();

    cout << "Built half-edge mesh with " << faceMap.size() << " faces, " << vertexMap.size() << " vertices, "
         << edgeMap.size() << " edges, and " << halfEdgeMap.size() << " halfedges." << endl;
    initialEdgeCount = edgeMap.size();
    initialFaceCount = faceMap.size();
    initialVertexCount = vertexMap.size();

    // im only working with manifold meshes for this proj
//    bool temp = runValidator;
//    runValidator = true; // force run validation when halfedge mesh created for the first time
//    validateMesh();
//    runValidator = temp; // reset to old value
//    cout << "Successfully validated initially built half-edge mesh." << endl;
}

void HalfEdgeMesh::cleanUpHalfEdgeMesh() {
    for (auto& f : faceMap) {
        delete f.second;
    }
    for (auto& v : vertexMap) {
        delete v.second;
    }
    for (auto& e : edgeMap) {
        delete e.second;
    }
    for (auto& he : halfEdgeMap) {
        delete he.second;
    }
    vertexIdCounter = 0;
    faceIdCounter = 0;
    halfEdgeIdCounter = 0;
    edgeIdCounter = 0;
}

void HalfEdgeMesh::createObjFileVerticesFaces(vector<Vector3f> &vertices, vector<Vector3i> &faces) {
    vertices.clear(); faces.clear();

    cout << "Creating result OBJ file with " << faceMap.size() << " faces, " << vertexMap.size() << " vertices, "
         << edgeMap.size() << " edges, and " << halfEdgeMap.size() << " halfedges..." << " ";

    unordered_map<int, int> heVIDToObjVID; // maps vertexIDs from halfedgeDS to vertexIDs for the obj file vertices
    int id=0;
    // for (auto& v : vertexMap) {
    for (int i=0; i<initialVertexCount; ++i) {
        if (!vertexMap.contains(i)) continue;
        Vertex* vertex = vertexMap[i]; //v.second;
        heVIDToObjVID[vertex->id] = id++;
        vertices.push_back(vertex->vertex3f);
    }

    // for (auto& f : faceMap) {
    for (int i=0; i<initialFaceCount; ++i) {
        if (!faceMap.contains(i)) continue;
        Face* face = faceMap[i]; //f.second;
        Vector3i vertexIDs = getVertexIdsFromFace(face);
        vertexIDs[0] = heVIDToObjVID[vertexIDs[0]];
        vertexIDs[1] = heVIDToObjVID[vertexIDs[1]];
        vertexIDs[2] = heVIDToObjVID[vertexIDs[2]];
        faces.push_back(vertexIDs);
    }

//    printMeshStats();
//    cout << "Created result obj file with " << faceMap.size() << " faces, " << vertexMap.size() << " vertices, "
//         << edgeMap.size() << " edges, and " << halfEdgeMap.size() << " halfedges." << endl;
    cout << "Done." << endl;
}

bool HalfEdgeMesh::validateMesh() {

    if (!runValidator) return true;

    unordered_set<int> allHalfEdgeIDs(halfEdgeMap.size());
    for (auto& hem : halfEdgeMap) {
        int heID = hem.first;
        allHalfEdgeIDs.insert(heID);
    }

    for (auto& hem : halfEdgeMap) {
        HalfEdge* he = hem.second;

        // check if all half edge properties are not null
        assert(he != nullptr && "Assert Failed: Found a nullptr Halfedge in HalfedgeMap");
        assert(he->twin != nullptr && "Assert Failed: Halfedge twin does not exist");
        assert(he->next != nullptr && "Assert Failed: Halfedge next does not exist");
        assert(he->vertex != nullptr && "Assert Failed: Halfedge does not emerge from any vertex");
        assert(he->edge != nullptr && "Assert Failed: Halfedge's edge does not exist");
        assert(he->face != nullptr && "Assert Failed: Halfedge does not belong to any face");

        allHalfEdgeIDs.erase(he->next->id);

        // check if this halfedge is part of a valid loop of 3 vertices
        HalfEdge* hptr = he;
        int numVertices = 0;
        do {
            numVertices++;
            hptr = hptr->next;
        } while (hptr != he && numVertices <= 3);
        assert(numVertices == 3 && "Assert Failed: A halfedge is not part of a valid loop of exactly 3 vertices");

        // manifold assumption
        assert(he->twin->twin == he && "Assert Failed: Twin of my twin is NOT myself");
        assert(he->twin != he && "Assert Failed: Half edge's twin is itself");
    }

    assert(allHalfEdgeIDs.size() == 0 && "Assert failed: Every halfedge is someone's next");

    for (auto& fm : faceMap) {
        Face* f = fm.second;
        assert(f != nullptr && "Assert Failed: Found a nullptr Face in FaceMap");
        assert(f->he != nullptr && "Assert Failed: Face halfedge does not exist");

        // check if this face is made up of exactly 3 vertices
        HalfEdge* hptr = f->he;
        int numVertices = 0;
        do {
            numVertices++;
            hptr = hptr->next;
        } while (hptr != f->he && numVertices <= 3);
        if (numVertices != 3) return false;
        assert(numVertices == 3 && "Assert Failed: A face has less/more than 3 vertices");
    }

    for (auto& vm : vertexMap) {
        Vertex* v = vm.second;
        assert(v != nullptr && "Assert Failed: Found a nullptr Vertex in VertexMap");
        assert(v->he != nullptr && "Assert Failed: Vertex halfedge does not exist");
    }

    for (auto& em : edgeMap) {
        Edge* e = em.second;
        assert(e != nullptr && "Assert Failed: Found a nullptr Edge in EdgeMap");
        assert(e->he != nullptr && "Assert Failed: Edge halfedge does not exist");
    }

    return true;
}
