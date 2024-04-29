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

int main(int argc, char *argv[]) {

    QCoreApplication a(argc, argv);

    cout << "Started server..." << endl;

//    string baseDir = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/MeshCNN/datasets/shrec_16/";
//    string meshName = "armadillo/train/T54.obj";
//    MeshEnv env(baseDir + meshName);
//    env.initMeshEnv();
    // env.setFinalFaceCount(75);

    MeshEnv env;

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
            j["message"] = "Took action of removing edge with ID" + action_str;
        } else if (request.find("GET /update-env") != std::string::npos) { // endpoint /step?action={edgeIdToRemove}&meshFilePath={pathtomeshfile}&
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

            if (action_str != "") {
                j["message"] = "Performed action=" + action_str;
                if (action_str == "train") env.setIsTraining(true);
                else if (action_str == "test") env.setIsTraining(false);
            }

            string meshFilePath_str;
            pos = request.find("meshFilePath=");
            if (pos != string::npos) {
                pos += 13; // len of "meshFilePath="
                size_t end_pos = request.find('&', pos);
                if (end_pos != string::npos) {
                    meshFilePath_str = request.substr(pos, end_pos - pos);
                } else {
                    meshFilePath_str = request.substr(pos);
                }
            }
            if (meshFilePath_str != "") {
                j["message"] = "\nUpdated mesh env with new mesh file.";
                env.setMeshFilePath(meshFilePath_str);
                env.reset();
                j["state"] = env.getState();
            }

            string faceCount_str;
            pos = request.find("faceCount=");
            if (pos != string::npos) {
                pos += 10; // len of "faceCount="
                size_t end_pos = request.find('&', pos);
                if (end_pos != string::npos) {
                    faceCount_str = request.substr(pos, end_pos - pos);
                } else {
                    faceCount_str = request.substr(pos);
                }
            }

            if (faceCount_str != "") {
                int faceCount = std::stoi(faceCount_str);
                env.setFinalFaceCount(faceCount);
            }
        } else if (request.find("GET /save-mesh") != std::string::npos) {
            if (!env.isTraining) {env.printEpisodeStats(); cout << endl;};

            string savePath = env.getMeshFilePath() + "_to_" + to_string(env.getFaceCount()) + "f_RL.obj";
            j["message"]  = "Saved current state of mesh to " + savePath;
            env.saveToFile(savePath);
            // running = false; // shut down server after training / testing done
        } else if (request.find("GET /bye") != std::string::npos) {
            if (!env.isTraining) {env.printEpisodeStats(); cout << endl;};

            string savePath = env.getMeshFilePath() + "_to_" + to_string(env.getFaceCount()) + "f_RL.obj";
            j["message"]  = "Server is shutting down... saving the current state of the mesh at " + savePath;
            if (!env.isTraining) env.saveToFile(savePath);
            running = false; // shut down server after training / testing done
        } else {
            j["message"] = "Invalid request.";
        }
        j["currFaceCount"] = env.getFaceCount();
        response = j.dump();

        response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(response.size()) + "\r\n\r\n" + response;
        write(socket, buffer(response), ignored_error);
    }

    a.exit();
    return 0;
}