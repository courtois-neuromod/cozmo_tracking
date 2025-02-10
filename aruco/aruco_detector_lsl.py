"""
Adapted from https://pyimagesearch.com/
"""

from datetime import datetime
import cv2
import numpy as np
import sys
import time
import argparse
import threading
import subprocess

from pylsl import StreamInfo, StreamOutlet


from config import (
    MAP_H_IRL,
    MAP_W_IRL,
    SEARCH_H,
    SEARCH_W,
    CAM_W,
    CAM_H,
    SOURCE,
    SOURCE_ID,
)


class ArUcoDecoder:
    """ArUco markers decoder class"""

    def __init__(self, traj):
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

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

        self.robot_position = None
        self.new_position = False

        self.thread_send = threading.Thread(target=self.send_loop)
        self.lock_send = threading.Lock()

        self.done = False

    def send_loop(self):
        info = StreamInfo(
            name="robot_tracking",
            type="aruco_detector",
            channel_count=2,
            channel_format="float32",
            source_id=SOURCE_ID,
        )
        outlet = StreamOutlet(info)
        while not self.done:
            time.sleep(0.01)
            self.lock_send.acquire()
            new_position = self.new_position
            self.new_position = False
            last_pos = self.robot_position
            self.lock_send.release()

            if new_position:
                x_pos, y_pos = last_pos
                if last_pos != (None, None):
                    data = [x_pos, y_pos]
                    outlet.push_sample(np.array(data, dtype=np.float32))

    def set_cap_prop(self):
        """Camera setting function"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        # lower exposure time of camera to increase detection rate
        # if too long the motion blur makes it harder to detect while moving fast
        subprocess.check_call(f"v4l2-ctl -d {SOURCE} -c auto_exposure=1", shell=True)
        subprocess.check_call(
            f"v4l2-ctl -d {SOURCE} -c exposure_time_absolute=200", shell=True
        )

    def draw_corners(self, corners, ids):
        """Drawing function, adding corners and ids of detected markers onto the displayed feed.

        :param corners: list of detected markers' coordinates
        :type corners: list
        :param ids: list of detected markers' IDs
        :type ids: list
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for markerCorner, markerID in zip(corners, ids):
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
                if markerID == 5:
                    self.ref_centers[str(markerID)] = center
                else:
                    self.ref_centers[str(markerID)] = center

                cv2.circle(self.img, center, 2, (0, 0, 255), -1)

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

    def resize(self, source, scale_percent=70):
        width = int(source.shape[1] * scale_percent / 100)
        height = int(source.shape[0] * scale_percent / 100)
        dim = (width, height)
        rs_img = cv2.resize(source, dim)
        return rs_img

    def calibration(self):
        """Initialization function, detecting the 4 ArUco markers located in the corners,
        and deriving the homography matrix between the camera's and the floor's planes.
        """

        # detect the four corner markers
        ids = []
        while ids is None or not all(x in ids for x in [1, 2, 3, 4]):
            _, self.img = self.cap.read()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            (corners, ids, _) = self.dectector.detectMarkers(self.img)
            self.draw_corners(corners, ids)
            cv2.imshow("Calibration", self.resize(source=self.img, scale_percent=100))
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
        """Decoding function, detecting markers present in the received image,
        and computing the robot's position in the maze's reference frame."""
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

        corners, ids, _ = self.detector.detectMarkers(srch_area)

        corners = np.asarray(corners)
        if corners.size != 0:
            corners += win_origin
        corners = tuple(corners)

        self.draw_corners(corners, ids)

        self.robot_pos_raw = None  # if robot not detected
        if ids is not None and 5 in ids:
            # update robot position for local search
            self.robot_pos_raw = self.ref_centers["5"]

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

        else:
            robot = (None, None)
            cv2.putText(
                self.img,
                "Robot's position: ({}, {})".format(robot[0], robot[1]),
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2,
            )

        self.lock_send.acquire()
        self.new_position = True
        self.robot_position = (robot[0], robot[1])
        self.lock_send.release()

        # draw traj if needed
        if self.traj:
            # create trajectory image
            if self.traj_img is None:
                shape = np.shape(self.img)
                self.traj_img = np.zeros(shape, np.uint8)
            # update trajectory image
            if "5" in self.ref_centers:
                cv2.circle(self.traj_img, self.ref_centers["5"], 1, (0, 0, 255), -1)

            # blend images
            alpha = 1
            beta = 1 - 0.6
            self.img = cv2.addWeighted(self.img, alpha, self.traj_img, beta, 0.0)

        cv2.imshow("Frame", self.resize(source=self.img))

        return

    def tracking(self):
        """Tracking function, reading a frame from the camera and decoding it."""
        while True:
            _, self.img = self.cap.read()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.decode()

            if cv2.waitKey(1) == ord("q"):
                print("Exiting...")
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                if self.traj:
                    cv2.imwrite(
                        "trajectory_"
                        + datetime.now().strftime("%Y%m%d-%H%M%S")
                        + ".png",
                        self.img,
                    )
                self.done = True
                self.thread_send.join()
                print("Done.")
                sys.exit()


def main(traj):
    decoder = ArUcoDecoder(traj)
    decoder.calibration()
    decoder.thread_send.start()
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
