from gettext import npgettext
import cv2.aruco as aruco
import cv2
import numpy as np

squaresX = 5
squaresY = 7
squareLength = 0.04 # meters
markerLength = 0.03 # meters
dictionary = aruco.Dictionary_get(aruco.DICT_5X5_50)

board = aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)
img = 0
img = board.draw((500, 700), marginSize=20)#, img, 1, 1)

cv2.imwrite('charuco_board.jpg', img)
cv2.imshow("board", img)
cv2.waitKey(0)
