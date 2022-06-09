import socket

# Maze H and W (to modify depending on the physical setup)
MAP_H_IRL = 72.0  
MAP_W_IRL = 110.4 

# Local search area dimension
SEARCH_H = 100
SEARCH_W = 100

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