import socket

# Maze H and W in cm (to modify depending on the physical setup)
MAP_H_IRL = 110
MAP_W_IRL = 110

# Local search area dimension
SEARCH_H = 1850
SEARCH_W = 1850

# Camera resolution
MAX_CAM_H = 1920
MAX_CAM_W = 1080
CAM_W = MAX_CAM_H / 1.5
CAM_H = MAX_CAM_W / 1.5

# Communication specs
SENDING_PORT = 1030
ADDR_FAMILY = socket.AF_INET
SOCKET_TYPE = socket.SOCK_STREAM

# Video source (camera)
SOURCE = "/dev/video0"

# Id of output stream
SOURCE_ID = "neuromod_cozmo_tracking"
