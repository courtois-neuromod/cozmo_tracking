"using getperspective instead of homography"

import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
import sys

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
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
 
class QRDecoder:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        focus = 0  # min: 0, max: 255, increment:5
        self.cap.set(28, focus)
      
        self.h_corr = 1.0
        self.w_corr = 1.0
        self.P = None
        self.dstPoints = None
        self.srcPoints = None
        self.max_w = None
        self.max_h = None

    def calibration(self):
        decodedObjects = []

        while len(decodedObjects) != 4:
            _, img = self.cap.read()
            decodedObjects = pyzbar.decode(img)
            display(img, decodedObjects, message="Detected beacons")
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                sys.exit()
        
        display(img, decodedObjects, message="Detected beacons")
        cv2.destroyAllWindows()
        
        centers = {}
        for obj in decodedObjects:
            data = obj.data
            points = obj.polygon
            points = np.array([point for point in points])
            center = np.sum(points, axis=0) / 4
            center = center.astype(int)
            centers[data] = center
            cv2.circle(img, tuple(center), 1, (0, 0, 255), 5)

        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        top_left = centers[b"top_left_ref"]
        top_right = centers[b"top_right_ref"]
        bottom_left = centers[b"http://bottom_left_ref"]
        bottom_right = centers[b"bottom_right_ref"]

        w_a = np.sqrt( (top_left[0] - bottom_left[0])**2 + (top_left[1] - bottom_left[1])**2 ) 
        w_b = np.sqrt( (top_right[0] - bottom_right[0])**2 + (top_right[1] - bottom_right[1])**2 ) 
        h_a = np.sqrt( (top_left[0] - top_right[0])**2 + (top_left[1] - top_right[1])**2 ) 
        h_b = np.sqrt( (bottom_left[0] - bottom_right[0])**2 + (bottom_left[1] - bottom_right[1])**2 ) 

        self.max_w = max(int(w_a), int(w_b))
        self.max_h = max(int(h_a), int(h_b))

        self.dstPoints = np.array([
                [0, 0],
                [self.max_w - 1, 0],
                [self.max_w - 1, self.max_h - 1],
                [0, self.max_h - 1]], dtype = "float32")

        self.srcPoints = np.array([top_left, top_right, bottom_right, bottom_left,], dtype = "float32")

        self.P = cv2.getPerspectiveTransform(self.srcPoints, self.dstPoints)
        img_warp = cv2.warpPerspective(img, self.P, (self.max_w, self.max_h))

        cv2.imshow("warp", img_warp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def decode(self, im):
        decodedObjects = pyzbar.decode(im)
        robot = None
        if not decodedObjects:
            return None, None
        for obj in decodedObjects:
            if obj.data == "robot":
                robot = obj
                break
        if not robot:
            print("miss")
            return None, None
        points = robot.polygon
        points = np.array([point for point in points])
        center = np.sum(points, axis=0) / 4
        center = center.astype(int)
        cv2.circle(im, tuple(center), 1, (0, 0, 255), 5)

        """ im_warp = cv2.warpPerspective(im, self.P, (self.max_w, self.max_h))

        cv2.imshow("warp", im_warp) """
        cv2.imshow("found", im)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
        return
        #exit(0)


        left, top, width, height = robot.rect

        top_left = np.array([top, left])
        top_right = np.array([top, left + width])
        bottom_left = np.array([top - height, left])
        bottom_right = np.array([top - height, left + width])

        center = (top_left + top_right + bottom_left + bottom_right) / 4
        center = [center[0] * self.h_corr, center[1] * self.w_corr]

        return decodedObjects, center

    def tracking(self):
        while True:
            _, img = self.cap.read()
            decodedObjects, center = self.decode(img)
            print("CENTER : ", center)
            if decodedObjects:
                display(img, decodedObjects)
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()

                sys.exit()


def main():
    decoder = QRDecoder()
    decoder.calibration()
    key = input("Once Cozmo is set, press any litteral key to continue.")
    if key:
        decoder.tracking()
    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
