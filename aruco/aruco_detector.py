"""
Adapted from https://pyimagesearch.com/
"""

from datetime import datetime
from ssl import ALERT_DESCRIPTION_UNEXPECTED_MESSAGE
import cv2
import numpy as np
import sys
import time
from random import randint
import argparse
import socket
import threading

# Maze H and W (to modify depending on the physical setup)
MAP_H_IRL = 72.0  # 23.0
MAP_W_IRL = 110.4  # 15.2

# Local search area dimension
SEARCH_H = 75
SEARCH_W = 75

# Camera resolution
CAM_W = 1920
CAM_H = 1080

# Communication specs
SENDING_PORT = 1030
ADDR_FAMILY = socket.AF_INET
SOCKET_TYPE = socket.SOCK_STREAM

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

SOURCE = "/dev/video2"


class ArUcoDecoder:
    """ArUco markers decoder class"""

    def __init__(self, traj):
        self.arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.cap = cv2.VideoCapture(SOURCE)
        if self.cap is None or not self.cap.isOpened():
            print("Warning: unable to open video source: ", SOURCE)
            sys.exit(0)
        self.set_cap_prop()
        time.sleep(2.0)

        self.img = None
        self.traj = traj
        self.traj_img = None
        self.P = None
        self.ref_centers = {}
        self.robot_pos_raw = None

        """ 
        self.sock_send = socket.socket(ADDR_FAMILY, SOCKET_TYPE)
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_send.bind(("", self.tcp_port_send))
        self.sock_send.listen(10)
        self.sock_send.settimeout(1.5)

        self.thread_send = threading.Thread(target=self.send_loop)
        self.thread_send.start()
        self.lock_send = threading.Lock() 
        """

    """ def send_loop(self):
        # connect to the task listener
        # fetch last position (and timestamp)
        # send last position (and timestamp) """

    def set_cap_prop(self):
        """Camera setting function"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

    def draw_corners(self, corners, ids):
        """Drawing function, adding corners and ids of detected markers onto the displayed feed.

        :param corners: list of detected markers' coordinates
        :type corners: list
        :param ids: list of detected markers' IDs
        :type ids: list
        """
        bot_center = None
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(self.img, topLeft, topRight, (0, 255, 0), 1)
                cv2.line(self.img, topRight, bottomRight, (0, 255, 0), 1)
                cv2.line(self.img, bottomRight, bottomLeft, (0, 255, 0), 1)
                cv2.line(self.img, bottomLeft, topLeft, (0, 255, 0), 1)

                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                center = (cX, cY)
                self.ref_centers[str(markerID)] = center
                cv2.circle(self.img, center, 2, (0, 0, 255), -1)

                if self.traj and markerID == 5:
                    bot_center = center
                # draw the ArUco marker ID on the image
                cv2.putText(
                    self.img,
                    str(markerID),
                    (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return bot_center

    def resize(self, source, scale_percent=70):
        width = int(source.shape[1] * scale_percent / 100)
        height = int(source.shape[0] * scale_percent / 100)
        dim = (width, height)
        rs_img = cv2.resize(source, dim)
        return rs_img

    def calibration(self):
        """Initialization function, detecting the 4 ArUco markers located in the corners, and deriving the homography matrix between the camera's and the floor's planes."""

        # detect the four corner markers
        ids = []
        while ids is None or not all(x in ids for x in [1, 2, 3, 4]):
            _, self.img = self.cap.read()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            (corners, ids, _) = cv2.aruco.detectMarkers(
                self.img, self.arucoDict, parameters=self.arucoParams
            )
            _ = self.draw_corners(corners, ids)

            cv2.imshow("Calibration", self.resize(source=self.img))
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                sys.exit()

        cv2.destroyAllWindows()
        cv2.imshow("Detected ref", self.resize(source=self.img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # get perpective matrix
        top_left = self.ref_centers["1"]
        top_right = self.ref_centers["2"]
        bottom_left = self.ref_centers["3"]
        bottom_right = self.ref_centers["4"]

        w_a = np.sqrt(
            (top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2
        )
        w_b = np.sqrt(
            (top_right[0] - bottom_right[0]) ** 2
            + (top_right[1] - bottom_right[1]) ** 2
        )
        h_a = np.sqrt(
            (top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2
        )
        h_b = np.sqrt(
            (bottom_left[0] - bottom_right[0]) ** 2
            + (bottom_left[1] - bottom_right[1]) ** 2
        )

        zoom = 5

        self.max_h = zoom * max(int(w_a), int(w_b))
        self.max_w = zoom * max(int(h_a), int(h_b))

        dstPoints = np.array(
            [
                [0, 0],
                [self.max_w - 1, 0],
                [self.max_w - 1, self.max_h - 1],
                [0, self.max_h - 1],
            ],
            dtype="float32",
        )

        srcPoints = np.array(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            ],
            dtype="float32",
        )

        self.P = cv2.getPerspectiveTransform(srcPoints, dstPoints)

        # warp image
        img_warp = cv2.warpPerspective(self.img, self.P, (self.max_w, self.max_h))

        cv2.imshow("warp", self.resize(source=img_warp, scale_percent=40))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # if the robot has been detected, store its first position for local tracking
        if "5" in self.ref_centers:
            self.robot_pos_raw = self.ref_centers["5"]

    def decode(self):
        """Decoding function, detecting markers present in the received image, and computing the robot's position in the maze's reference frame."""
        # local search
        srch_area = self.img
        win_origin = (0, 0)
        if self.robot_pos_raw:

            w_min = max(self.robot_pos_raw[0] - SEARCH_W, 0)
            w_max = min(self.robot_pos_raw[0] + SEARCH_W, np.shape(self.img)[1])
            h_min = max(self.robot_pos_raw[1] - SEARCH_H, 0)
            h_max = min(self.robot_pos_raw[1] + SEARCH_H, np.shape(self.img)[0])
            win_origin = (w_min, h_min)
            srch_area = self.img[h_min:h_max, w_min:w_max]

        corners, ids, _ = cv2.aruco.detectMarkers(
            srch_area, self.arucoDict, parameters=self.arucoParams
        )
        
        corners = np.asarray(corners)  # TODO: account for multiple detections
        if corners.size != 0:
            corners += win_origin
        corners = tuple(corners)
        bot_pos = self.draw_corners(corners, ids)
           
        # update robot position for local search
        #self.robot_pos_raw = bot_pos
        self.ref_centers["5"]

        # draw traj if needed
        if self.traj:
            # create trajectory image
            if self.traj_img is None:
                shape = np.shape(self.img)
                self.traj_img = np.zeros(shape, np.uint8)
            # update trajectory image
            #cv2.circle(self.traj_img, bot_pos, 1, (0, 0, 255), -1)
            cv2.circle(self.traj_img, self.ref_centers["5"], 1, (0, 0, 255), -1)
            
            # blend images
            alpha = 0.6
            beta = 1 - 0.6
            self.img = cv2.addWeighted(self.img, alpha, self.traj_img, beta, 0.0)

        #cv2.imshow("Frame", self.resize(source=self.img))
        #self.img = cv2.warpPerspective(self.img, self.P, (self.max_w, self.max_h))
        #self.img = cv2.copyMakeBorder(self.img, 100, 100, 100, 100, cv2.BORDER_CONSTANT)

        if ids is not None and 5 in ids:
            robot = np.asarray(self.ref_centers["5"])
            robot = np.append(robot, 1)

            robot = np.matmul(self.P, np.transpose(robot))
            robot /= robot[2]
            robot[1] *= MAP_H_IRL / self.max_h
            robot[0] *= MAP_W_IRL / self.max_w

            cv2.putText(
                self.img,
                "Robot's position: ({:.2f}, {:.2f})".format(robot[0], robot[1]),
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2,
            )
        
        cv2.imshow("Frame", self.resize(source=self.img))
        #cv2.imshow("warp", self.resize(source=self.img, scale_percent=30))
        if cv2.waitKey(1) == ord("q"):
            if self.traj:
                cv2.imwrite(
                    f"trajectory_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png",
                    self.img,
                )
            cv2.destroyAllWindows()
            sys.exit()

        return

    def tracking(self):
        """Tracking function, reading a frame from the camera and decoding it."""
        while True:
            start_loop = time.time()
            _, self.img = self.cap.read()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            start = time.time()
            self.decode()
            end = time.time()

            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()

                sys.exit()
            end_loop = time.time()


def main(traj):
    decoder = ArUcoDecoder(traj)
    decoder.calibration()
    decoder.calib = True
    time.sleep(1)
    decoder.tracking()


def parser():
    parser = argparse.ArgumentParser(description="ArUco detector and tracker.")
    parser.add_argument(
        "-t",
        "--traj",
        action="store_true",
        default=False,
        help="trajectory drawing boolean",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    main(args.traj)
