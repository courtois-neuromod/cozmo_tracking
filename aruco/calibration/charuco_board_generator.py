import cv2
import numpy as np


def make_board(squaresX, squaresY, squareLength, markerLength, dictionary):

    board = cv2.aruco.CharucoBoard_create(
        squaresX, squaresY, squareLength, markerLength, dictionary
    )
    img = 0
    img = board.draw((500, 700), marginSize=20)  

    cv2.imwrite("charuco_board.jpg", img)
    # cv2.imshow("board", img)
    # cv2.waitKey(0)

    return board
