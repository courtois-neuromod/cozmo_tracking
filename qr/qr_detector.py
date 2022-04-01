from turtle import circle
import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
import sys

W_REF = 0.0
H_REF = 0.0

dstPoints = 15*np.array([np.array((6.0,6.0)),
                    np.array((11.6+6.0,6.0)),
                    np.array((0.0+6.0,18.4+6.0)),
                    np.array((11.6+6.0,18.4+6.0)),])

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

class QRDecoder:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3,640)
        self.cap.set(4,480)
        self.h_corr = 1.0
        self.w_corr = 1.0

    def calibration(self):
        decodedObjects = []

        while len(decodedObjects) != 4:
            _, img = self.cap.read()
            decodedObjects = pyzbar.decode(img)
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                sys.exit()
        
        display(img, decodedObjects, message="Detected beacons")
        
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
        print(top_left)
        top_right = centers[b"top_right_ref"]
        bottom_left = centers[b"http://bottom_left_ref"]
        bottom_right = centers[b"bottom_right_ref"]
        srcPoints = np.array([top_left, top_right, bottom_left, bottom_right])
        print(srcPoints)
        print(dstPoints)
        H, _ = cv2.findHomography(srcPoints, dstPoints)
        img_warp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        cv2.imshow("warp", img_warp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(H)
        h_obs = bottom_left[0] - top_left[0]
        w_obs = top_left[1] - top_left[1]

        self.h_corr = 1 - (h_obs / H_REF)
        self.w_corr = 1 - (w_obs / W_REF)

    def decode(self, im):
        decodedObjects = pyzbar.decode(im)

        if not decodedObjects:
            return None, None

        for obj in decodedObjects:
            if obj.data == "robot":
                robot = obj
                break

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
