import socket

# Maze H and W (to modify depending on the physical setup)
MAP_H_IRL = 72.0  
MAP_W_IRL = 110.4 

# Local search area dimension
SEARCH_H = 100
SEARCH_W = 100

# Camera resolution
CAM_W = 1920 
CAM_H = 1080 

# Communication specs
SENDING_PORT = 1030
ADDR_FAMILY = socket.AF_INET
SOCKET_TYPE = socket.SOCK_STREAM

# Video source (camera)
SOURCE = "/dev/video0"