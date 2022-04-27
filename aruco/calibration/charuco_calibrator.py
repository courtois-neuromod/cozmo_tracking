import cv2.aruco as aruco
from charuco_board_generator import make_board

# Camera resolution
CAM_W = 1920
CAM_H = 1080
imageSize = (CAM_W, CAM_H)  # input image size

board = make_board(
    squaresX=5,
    squaresY=7,
    squareLength=0.04,
    markerLength=0.03,
    dictionary=aruco.Dictionary_get(aruco.DICT_5X5_50),
)

charucoCorners = None   # vector of detected charuco corners per frame
charucoIds = None   # list of identifiers for each corner in charucoCorners per frame

# Detect charuco board from several viewpoints and fill charucoCorners and charucoIds
# ...
# After capturing in several viewpoints, start calibration

cameraMatrix = None # Output 3x3 floating-point camera matrix 
distCoeffs = None   # Output vector of distortion coefficients
rvecs = None    # Output vector of rotation vectors
tvecs = None    # Output vector of translation vectors estimated for each pattern view
flags = None    # Different flags for the calibration process (see #calibrateCamera for details)
repError = aruco.calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria)