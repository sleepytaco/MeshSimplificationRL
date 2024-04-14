#include <QCoreApplication>
#include <QCommandLineParser>
#include <QtCore>

#include <iostream>
#include <chrono>

#include "meshenv.h"

#include <boost/asio.hpp>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using namespace boost::asio;
using namespace boost::asio::ip;
using namespace std;
using json = nlohmann::json;

string serializeVec2D(auto& data) {
    json j(data);
    return j.dump();
}

string serializeStep(pair<float, int>& p) {
    json j;
    j["reward"] = p.first;
    j["isTerminal"] = p.second;
    return j.dump();
}

int main(int argc, char *argv[]) {
    cout << "Started server..." << endl;

    MeshEnv env("/Users/mohammedk/Documents/Brown/CS2951F/Final Project/MeshSimplificationRL/meshenv/meshes/cow.obj");

    io_service service;
    tcp::acceptor acceptor(service, tcp::endpoint(tcp::v4(), 12345));


    while (true) {
        tcp::socket socket(service);
        acceptor.accept(socket);

        boost::system::error_code ignored_error;
        std::string request;
        read_until(socket, dynamic_buffer(request), "\n", ignored_error);

        string response;
        json j;
        if (request.find("GET /hello") != std::string::npos) {
            j["message"]  = "Server is running :D";
        } else if (request.find("GET /reset") != std::string::npos) {
            env.reset();
            j["state"] = env.getState();
            j["message"]  = "MeshEnv has been reset to initial mesh state!";
        } else if (request.find("GET /get-state") != std::string::npos) {
            j["state"] = env.getState();
            j["message"] = "Returned current mesh state.";
        } else if (request.find("GET /step") != std::string::npos) { // endpoint /step?action={edgeIdToRemove}
            // prse action from the request
            string action_str;
            size_t pos = request.find("action=");
            if (pos != string::npos) {
                pos += 7; // len of "action="
                size_t end_pos = request.find('&', pos);
                if (end_pos != string::npos) {
                    action_str = request.substr(pos, end_pos - pos);
                } else {
                    action_str = request.substr(pos);
                }
            }

            int action = std::stoi(action_str);

            pair<float, bool> actionResponse = env.step(action);
            j["reward"] = actionResponse.first;
            j["isTerminal"] = actionResponse.second;
            j["state"] = env.getState();
            j["message"] = "Took action of removing edge with ID";
        } else {
            j["message"] = "Invalid request.";
        }
        response = j.dump();

        response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(response.size()) + "\r\n\r\n" + response;
        write(socket, buffer(response), ignored_error);
    }

    return 0;
}
//int main(int argc, char *argv[])
//{
//    QCoreApplication a(argc, argv);
//    QCommandLineParser parser;
//    parser.addHelpOption();
//    parser.addPositionalArgument("config",  "Path of the config (.ini) file.");
//    parser.process(a);

//    // Check for invalid argument count
//    const QStringList args = parser.positionalArguments();
//    if (args.size() < 1) {
//        std::cerr << "Not enough arguments. Please provide a path to a config file (.ini) as a command-line argument." << std::endl;
//        a.exit(1);
//        return 1;
//    }

//    // Parse common inputs
//    QSettings settings( args[0], QSettings::IniFormat );
//    QString infile  = settings.value("IO/infile").toString();
//    QString outfile = settings.value("IO/outfile").toString();
//    QString method  = settings.value("Method/method").toString();

//    // Load
//    MeshEnv env(infile.toStdString());
//    // m.initMeshEnv(infile.toStdString());

//    // Start timing
//    auto t0 = std::chrono::high_resolution_clock::now();

//    // Switch on method
//    if (method == "simplify") {

//        // args1:
//        // Simplify:  number of faces to remove

//        // int numFacesToRemove = settings.value("Parameters/args1").toInt(); // number of faces to remove
//        bool terminal = false;
//        int edgeId = 0;
//        env.reset();
//        env.getState();
//        while (!terminal) {
//            auto res = env.step(edgeId++);
//            float reward = res.first;
//            terminal = res.second;
//            cout << reward << ", " << terminal << endl;
//            env.getState();
//        }
////        env.reset();
//        env.getState();

//    } else if (method == "edgeCollapse") {

//        // args1:
//        // edgeCollapse:   number of edges to collapse

////        int numEdgesToCollapse = settings.value("Parameters/args1").toInt();

////        unordered_set<int> originalEdgeIDs(m.halfEdgeMesh.edgeMap.size()); // store edge IDs of original mesh (coz edgeSplit will add new items to the edgeMap)
////        for (auto& em : m.halfEdgeMesh.edgeMap) originalEdgeIDs.insert(em.first);
////        int i=0, numCollapsed=0;
////        for (int edgeID : originalEdgeIDs) {
////            if (m.halfEdgeMesh.edgeMap.find(edgeID) == m.halfEdgeMesh.edgeMap.end()) // NEED THIS check as edgeCollapse operation deletes some surrounding edges from the edgeMap
////            { i ++;continue;}

////            Vertex* v0 = m.halfEdgeMesh.edgeMap[edgeID]->he->vertex;
////            Vertex* v1 = m.halfEdgeMesh.edgeMap[edgeID]->he->next->vertex;
////            Vector3f collapsePt = (v0->vertex3f + v1->vertex3f) / 2.f; // collapse edge to midpoint
////            m.halfEdgeMesh.edgeCollapse(m.halfEdgeMesh.edgeMap[edgeID], collapsePt);

////            i ++;
////            numCollapsed ++;
////            if (numCollapsed == numEdgesToCollapse) break;
////        }

//    } else {

//        std::cerr << "Error: Unknown method \"" << method.toUtf8().constData() << "\"" << std::endl;

//    }

//    env.halfEdgeMesh->validateMesh();

//    // Finish timing
//    auto t1 = std::chrono::high_resolution_clock::now();
//    auto duration = duration_cast<std::chrono::milliseconds>(t1 - t0).count();
//    std::cout << "Execution took " << duration << " milliseconds." << std::endl;

//    // Save
//    env.saveToFile(outfile.toStdString());

//    a.exit();
//}
