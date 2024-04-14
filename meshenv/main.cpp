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

    string baseDir = "/Users/mohammedk/Documents/Brown/CS2951F/Final Project/MeshSimplificationRL/meshenv/meshes/";
    string meshName = "cow_500f.obj";
    MeshEnv env(baseDir + meshName);
    env.initMeshEnv();

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
            j["message"] = "Took action of removing edge with ID";
        } else if (request.find("GET /bye") != std::string::npos) {
            j["message"]  = "Server is shutting down... saving the current state of the mesh at: " + baseDir + "RLsimplified_" + meshName;
            env.saveToFile(baseDir + "RLsimplified_" + meshName);
            running = false;
        } else {
            j["message"] = "Invalid request.";
        }
        response = j.dump();

        response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(response.size()) + "\r\n\r\n" + response;
        write(socket, buffer(response), ignored_error);
    }

    a.exit();
    return 0;
}
