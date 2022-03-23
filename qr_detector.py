import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
import sys

W_REF = 0.0
H_REF = 0.0


def display(im, decodedObjects):
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
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    # Display results
    cv2.imshow("Results", im)


class QRDecoder:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.h_corr = 1.0
        self.w_corr = 1.0

    def calibration(self):
        decodedObjects = []

        while len(decodedObjects) != 4:
            _, img = self.cap.read()
            decodedObjects = pyzbar.decode(img)
            cv2.imshow("Results", img)
            if cv2.waitKey(1) == ord("q"):
                cv2.waitKey(1)
                cv2.destroyAllWindows()

                sys.exit()

        centers = {}
        for obj in decodedObjects:
            data = obj.data
            left, top, width, height = obj.rect

            top_left = np.array([top, left])
            top_right = np.array([top, left + width])
            bottom_left = np.array([top - height, left])
            bottom_right = np.array([top - height, left + width])

            center = (top_left + top_right + bottom_left + bottom_right) / 4
            centers[data] = center

        top_left = centers["top_left_ref"]
        top_right = centers["top_right_ref"]
        bottom_left = centers["bottom_left_ref"]

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
