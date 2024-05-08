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

string parseParam(string& request, string param) {
    string param_str;
    size_t pos = request.find(param);
    if (pos != string::npos) {
        pos += param.size();
        size_t end_pos = request.find('&', pos);
        if (end_pos != string::npos) {
            param_str = request.substr(pos, end_pos - pos);
        } else {
            param_str = request.substr(pos);
        }
    }
    return param_str;
}

int main(int argc, char *argv[]) {

    QCoreApplication a(argc, argv);

    cout << "Started server..." << endl;

//    string baseDir = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/MeshCNN/datasets/shrec_16/";
//    string meshName = "armadillo/train/T54.obj";
//    MeshEnv env(baseDir + meshName);
//    env.initMeshEnv();
    // env.setFinalFaceCount(75);

    int version = 1;
    MeshEnv env;
    env.setVersion(version);

    io_service service;
    tcp::acceptor acceptor(service, tcp::endpoint(tcp::v4(), 12345));

    bool running = true;
    while (running) {
        tcp::socket socket(service);
        acceptor.accept(socket);

        boost::system::error_code ignored_error;
        string request;
        read_until(socket, dynamic_buffer(request), "\n", ignored_error);

        string response;
        json j;

        j["info"]["hasInfo"] = false;
        if (request.find("GET /hello") != std::string::npos) {
            j["message"]  = "Server is running :D";
        } else if (request.find("GET /get-info") != std::string::npos) {
            env.saveEpisodeStats(j["info"]);
        } else if (request.find("GET /reset") != std::string::npos) {
            env.saveEpisodeStats(j["info"]);
            env.reset();
            // j["state"] = env.getState();
            j["message"]  = "MeshEnv has been reset to initial mesh state!";
        } else if (request.find("GET /get-state") != std::string::npos) {
            // j["state"] = env.getState();
            j["message"] = "Returned current mesh state.";
        } else if (request.find("GET /step") != std::string::npos) { // endpoint /step?action={edgeIdToRemove}
            pair<float, bool> actionResponse;

            if (env.envVersion == 1) {
                int action = std::stoi(parseParam(request, "action="));
                actionResponse = env.step(action);
            } else if (env.envVersion == 2) {
                float x = std::stof(parseParam(request, "x="));
                float y = std::stof(parseParam(request, "y="));
                float z = std::stof(parseParam(request, "z="));
                Vector3f action(x, y, z);
                actionResponse = env.stepV2(action);
            }

            j["reward"] = actionResponse.first;
            j["isTerminal"] = actionResponse.second;
        } else if (request.find("GET /update-env") != std::string::npos) { // endpoint /step?action={edgeIdToRemove}&meshFilePath={pathtomeshfile}&
            // prse action from the request
            string action_str = parseParam(request, "action=");
            if (action_str != "") {
                j["message"] = "Performed action=" + action_str;
                if (action_str == "train") env.setIsTraining(true);
                else if (action_str == "test") env.setIsTraining(false);
            }

            string meshFilePath_str = parseParam(request, "meshFilePath=");
            if (meshFilePath_str != "") {
                j["message"] = "\nUpdated mesh env with new mesh file.";
                env.setMeshFilePath(meshFilePath_str);
                env.saveEpisodeStats(j["info"]);
                env.reset();
                // j["state"] = env.getState();
            }

            string faceCount_str = parseParam(request, "faceCount=");
            if (faceCount_str != "") {
                int faceCount = std::stoi(faceCount_str);
                env.setFinalFaceCount(faceCount);
            }

            string version_str = parseParam(request, "version=");
            if (version_str != "") {
                version = std::stoi(version_str);
                cout << "ENV VERSION NIMBER " << version_str << endl;
                // env.setVersion(version);
                cout << "Set environment to version " << version << endl;
            }

        } else if (request.find("GET /save-mesh") != std::string::npos) {
            if (!env.isTraining) {env.printEpisodeStats(); cout << endl;};

            string savePath = env.getMeshFilePath(); // + "_to_" + to_string(env.getFaceCount()) + "f_RL.obj";
            j["message"]  = "Saved current state of mesh to " + savePath;
            env.saveToFile(savePath);
            // running = false; // shut down server after training / testing done
        } else if (request.find("GET /bye") != std::string::npos) {
            if (!env.isTraining) {env.saveEpisodeStats(j["info"]); env.printEpisodeStats(); cout << endl;};

            string savePath = env.getMeshFilePath(); // + "_to_" + to_string(env.getFaceCount()) + "f_RL.obj";
            j["message"]  = "Server is shutting down... saving the current state of the mesh at " + savePath;
            if (!env.isTraining) env.saveToFile(savePath);
            // running = false; // shut down server after training / testing done
        } else {
            j["message"] = "Invalid request.";
        }

        if (env.envVersion == 1) j["state"] = env.getState();
        else if (env.envVersion == 2) j["state"] = env.getStateV2();
        else j["state"] = {{0, 0, 0}};

        if (!env.isTraining) env.saveValidEdgeIds(j["info"]);

        j["currFaceCount"] = env.getFaceCount();

        response = j.dump();

        response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(response.size()) + "\r\n\r\n" + response;
        write(socket, buffer(response), ignored_error);
    }

    a.exit();
    return 0;
}
