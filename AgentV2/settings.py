# mesh cnn meshes config (med sized meshes)
ROOT_DIR = ""
MAX_FACE_COUNT = 500 # 200
MAX_VERTEX_COUNT = 252 # 102
MAX_EDGE_COUNT = 750 # 300

STATE_SPACE_SIZE = MAX_VERTEX_COUNT + MAX_FACE_COUNT
ACTION_SPACE_SIZE = 3  # for agent V2, the action space is continuous

# c++ server that hosts my mesh simplification code
MESH_SERVER_HOST = '127.0.0.1'  # server IP address (local host)
MESH_SERVER_PORT = 12345  # server port