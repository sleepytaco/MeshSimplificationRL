# mesh cnn meshes config (med sized meshes)
ROOT_DIR = ""
BASE_MESH_DIR = "../MeshCNN/datasets/shrec_16/"
MESH_NAME = "armadillo/train/T54.obj"
MAX_FACE_COUNT = 500 # 200
MAX_VERTEX_COUNT = 252 # 102
MAX_EDGE_COUNT = 750 # 300

MESH_FILE = BASE_MESH_DIR + MESH_NAME

FINAL_FACE_COUNT = 50

STATE_SPACE_SIZE = MAX_VERTEX_COUNT + MAX_FACE_COUNT
ACTION_SPACE_SIZE = MAX_EDGE_COUNT  # num of edges that the agent can choose from


# c++ server that hosts my mesh simplification code
MESH_SERVER_HOST = '127.0.0.1'  # server IP address (local host)
MESH_SERVER_PORT = 12345  # server port