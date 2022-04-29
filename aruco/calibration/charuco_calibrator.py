import numpy as np
import cv2
import time, sys
from charuco_board_generator import make_board
import matplotlib.pyplot as plt

CAM_W = 1920
CAM_H = 1080

ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


def store(img, calib_images):
    calib_images.append(img)
    return calib_images


def quit(*args, **kwargs):
    cv2.destroyAllWindows()
    sys.exit()


cap_actions = {
    "c": store,
    "q": quit,
}


def set_cap_prop(cap):
    """Camera setting function"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    return cap


def capture_diff_pov(cam):
    calib_images = []

    cap = cv2.VideoCapture(cam)
    cap = set_cap_prop(cap)
    time.sleep(2.0)
    while True:
        _, img = cap.read()
        cv2.imshow("Cam feed", img)
        key = cv2.waitKey(1)
        if key != -1:
            try:
                calib_images = cap_actions[chr(key)](img, calib_images)
            except KeyError:
                if key == ord("f"):
                    break

    return calib_images


def detect_board_pov(board):
    charucoCorners = []
    charucoIds = []
    for img in calib_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(img, ARUCO_DICT)
        
        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],img,board)        
            print(res2[1])
            print(type(res2[1]))
            print(res2[2])
            print(type(res2[2]))
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
                charucoCorners.append(res2[1])
                charucoIds.append(res2[2])  

    return charucoCorners, charucoIds


if __name__ == "__main__":

    # build Charuco board
    board = make_board(
        squaresX=5,
        squaresY=7,
        squareLength=0.04,
        markerLength=0.03,
        dictionary=ARUCO_DICT,
    )

    # capture the board from different viewpoints
    cam = "/dev/video2"
    calib_images = capture_diff_pov(cam)

    # save calibration captures
    for i, img in enumerate(calib_images):
        cv2.imwrite(f"calib_img_{i}.png", img)

    # fill corners and ids from captured images
    charucoCorners, charucoIds = detect_board_pov(board)

    # Camera resolution
    imsize = (CAM_W, CAM_H)  # input image size

    cameraMatrixInit = np.array(
        [
            [2000.0, 0.0, imsize[0] / 2.0],
            [0.0, 2000.0, imsize[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    distCoeffsInit = np.zeros((5, 1))
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL
    (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    ) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=charucoCorners,
        charucoIds=charucoIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
    )

    np.savetxt("calib_mat_logitech_brio.csv", camera_matrix)
    frame = calib_images[0]
    img_undist = cv2.undistort(frame,camera_matrix,distortion_coefficients0,None)
    cv2.imshow("ert", img_undist)
    cv2.waitKey(0)