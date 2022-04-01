import cv2
import numpy as np
import sys
import time

MAP_H_IRL = 23.0
MAP_W_IRL = 15.2

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
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def display(im, decodedObjects, message="Results"):
    # Loop over all decoded objects

    for decodedObject in decodedObjects:

        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(
                np.array([point for point in points], dtype=np.float32)
            )
            hull = list(map(tuple, np.squeeze(hull)))

        else:
            hull = points

        # Number of points in the convex hull
        n = len(hull)
        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 2)

    # Display results
    cv2.imshow(message, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class ArUcoDecoder:
    def __init__(self):
        self.arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.set_cap_prop()
        time.sleep(2.0)

        self.img = None
        self.P = None
        self.ref_centers = {}

    def set_cap_prop(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("resolution (w x h): {} x {}".format(w, h))

    def draw_corners(self, corners, ids):
        
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
                
                # draw the ArUco marker ID on the image
                cv2.putText(self.img, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            
    def calibration(self):
        ids = []
        while ids is None or len(ids) != 4:
            _, self.img = self.cap.read()
            (corners, ids, _) = cv2.aruco.detectMarkers(self.img, self.arucoDict, parameters=self.arucoParams)            
            cv2.imshow("", self.img)
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                sys.exit()
        cv2.destroyAllWindows()
        self.draw_corners(corners, ids)
        cv2.imshow("Detected ref", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        top_left = self.ref_centers["1"]
        top_right = self.ref_centers["2"]
        bottom_left = self.ref_centers["3"]
        bottom_right = self.ref_centers["4"]

        w_a = np.sqrt( (top_left[0] - bottom_left[0])**2 + (top_left[1] - bottom_left[1])**2 ) 
        w_b = np.sqrt( (top_right[0] - bottom_right[0])**2 + (top_right[1] - bottom_right[1])**2 ) 
        h_a = np.sqrt( (top_left[0] - top_right[0])**2 + (top_left[1] - top_right[1])**2 ) 
        h_b = np.sqrt( (bottom_left[0] - bottom_right[0])**2 + (bottom_left[1] - bottom_right[1])**2 ) 

        zoom = 5

        self.max_h = zoom*max(int(w_a), int(w_b))
        self.max_w = zoom*max(int(h_a), int(h_b))
        
        dstPoints = np.array([
                [0, 0],
                [self.max_w - 1, 0],
                [self.max_w - 1, self.max_h - 1],
                [0, self.max_h - 1]], dtype = "float32")

        srcPoints = np.array([top_left, top_right, bottom_right, bottom_left,], dtype = "float32")

        self.P = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        img_warp = cv2.warpPerspective(self.img, self.P, (self.max_w, self.max_h))

        cv2.imshow("warp", img_warp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def decode(self):
        (corners, ids, _) = cv2.aruco.detectMarkers(self.img, self.arucoDict, parameters=self.arucoParams)            
        self.draw_corners(corners, ids)
        cv2.imshow("Frame", self.img)

        robot = np.asarray(self.ref_centers["5"])
        robot = np.append(robot, 1)
        robot = np.reshape(robot, (3,1))
        robot = np.matmul(self.P, robot)
        print(robot)
        robot = np.reshape(robot, (1, 3))
        exit(robot)
        robot[0][1] *= MAP_H_IRL/self.max_h
        robot[0][0] *= MAP_W_IRL/self.max_w
        cv2.putText(self.img, "Cozmo - ({})".format(robot),
                    (10.0, 10.0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        self.img = cv2.warpPerspective(self.img, self.P, (self.max_w, self.max_h))
        cv2.imshow("warp", self.img)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            sys.exit() 

        return

    def tracking(self):
        while True:
            _, self.img = self.cap.read()
            self.decode()
            
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()

                sys.exit()


def main():
    decoder = ArUcoDecoder()   
    decoder.calibration()
    key = input("Once Cozmo is set, press any litteral key to continue.")
    if key:
        decoder.tracking()
    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
