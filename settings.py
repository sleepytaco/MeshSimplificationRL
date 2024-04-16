MAX_FACE_COUNT = 200
MAX_VERTEX_COUNT = 102
MAX_EDGE_COUNT = 300

FINAL_FACE_COUNT = 50

STATE_SPACE_SIZE = MAX_VERTEX_COUNT + MAX_FACE_COUNT
ACTION_SPACE_SIZE = MAX_EDGE_COUNT  # num of edges that the agent can choose from


# c++ server that hosts my mesh simplification code
MESH_SERVER_HOST = '127.0.0.1'  # server IP address (local host)
MESH_SERVER_PORT = 12345  # server port