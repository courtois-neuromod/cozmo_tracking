"""
Adapted from https://github.com/Dr-Noob/qr_reader
"""

import cv2
import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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


def processing(img):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    (cont, _) = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return cont, img

def cont_inside_polygon(cnt, polygon):  
  for pt in cnt:
      point = Point(pt[0][0], pt[0][1])  
      if not (polygon.contains(point)):
          return False
      
  return True

def pt_inside_img(pt, img):
    h, w, _ = np.shape(img)
    return (pt[0] > 0 and pt[0] < w) and \
           (pt[1] > 0 and pt[1] < h)

def find_qrcont(frame, edge_candidates):
    edg1 = edge_candidates[0][0]
    edg2 = edge_candidates[1][0]
    edg3 = edge_candidates[2][0]

    dist1 = math.hypot(edg1[0] - edg2[0], edg1[1] - edg2[1])    
    dist2 = math.hypot(edg1[0] - edg3[0], edg1[1] - edg3[1]) 
    dist3 = math.hypot(edg2[0] - edg3[0], edg2[1] - edg3[1]) 
    
    if dist1 > dist2 and dist1 > dist3:
        edge1 = edge_candidates[0]
        edge2 = edge_candidates[1]
        corner = edge_candidates[2]
    elif dist2 > dist1 and dist2 > dist3:
        edge1 = edge_candidates[0]
        edge2 = edge_candidates[2]
        corner = edge_candidates[1]
    else:
        edge1 = edge_candidates[1]
        edge2 = edge_candidates[2]
        corner = edge_candidates[0]
    
    max_ed1 = (0, 0)
    max_ed2 = (0, 0)
    max_dst = -1
    tmpi = -1
    tmpj = -1
    for i in range(0,4):
        for j in range(0,4):
            dst = math.hypot(edge1[i][0] - edge2[j][0], edge1[i][1] - edge2[j][1])
            if dst > max_dst:
                max_dst = dst
                max_ed1 = edge1[i]
                max_ed2 = edge2[j]
                tmpi = i
                tmpj = j

    # Detect the third one (corner)  
    d_arr = np.array([0, 0, 0, 0])
    d_arr[0] = math.hypot(max_ed1[0] - corner[0][0], max_ed1[1] - corner[0][1]) + \
               math.hypot(max_ed2[0] - corner[0][0], max_ed2[1] - corner[0][1]) 
    d_arr[1] = math.hypot(max_ed1[0] - corner[1][0], max_ed1[1] - corner[1][1]) + \
               math.hypot(max_ed2[0] - corner[1][0], max_ed2[1] - corner[1][1]) 
    d_arr[2] = math.hypot(max_ed1[0] - corner[2][0], max_ed1[1] - corner[2][1]) + \
               math.hypot(max_ed2[0] - corner[2][0], max_ed2[1] - corner[2][1]) 
    d_arr[3] = math.hypot(max_ed1[0] - corner[3][0], max_ed1[1] - corner[3][1]) + \
               math.hypot(max_ed2[0] - corner[3][0], max_ed2[1] - corner[3][1])             
    max_idx = np.argmax(d_arr)
    max_ed3 = corner[max_idx]                                          
    
    # Detect the last one 
    max_area = -1
    partial_cnt = np.array([max_ed1, max_ed3, max_ed2])
    pt1 = np.append(max_ed1, 1)          
    pt3 = np.append(max_ed2, 1)    
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            pt2 = np.append(edge1[(tmpi+i)%4], 1)
            pt4 = np.append(edge2[(tmpj+j)%4], 1)
            
            l1 = np.cross(pt1, pt2)
            l2 = np.cross(pt3, pt4)
            
            ptfuga = np.cross(l1, l2) # float 
            
            if ptfuga[1] != 0:
              # Remove last coord
              if ptfuga[2] != 0:
                  ptfuga[0] = ptfuga[0]/ptfuga[2]
                  ptfuga[1] = ptfuga[1]/ptfuga[2] 
              ptfuga = ptfuga[:-1]
              
              # Check if inside img!
              if(pt_inside_img(ptfuga, frame)):
                  # If it is, keep the one with max area
                  tmp_ed4 = np.array((ptfuga[0], ptfuga[1])).astype('int32')                  
                  area = cv2.contourArea(np.append(partial_cnt, tmp_ed4).reshape(4,2))               
                  if area > max_area:
                      coord4x = ptfuga[0]
                      coord4y = ptfuga[1]  
                      max_area = area
                                                 
    max_ed4 = (coord4x, coord4y)
    outer_corners = [max_ed1, max_ed2, max_ed3, max_ed4]
    return outer_corners

def detect_qr(frame, cont):

    if len(cont) == 0:
        return

    edge_candidates = []
    
    for i in range(len(cont)):              
      cont_inside = 0    
      t_cnt = cont[i].reshape(4,2)
      polygon = Polygon([tuple(t_cnt[0]), tuple(t_cnt[1]), tuple(t_cnt[2]), tuple(t_cnt[3])])
      for c in cont:
          if(cont_inside_polygon(c, polygon)):
              cont_inside = cont_inside+1
              
      # Check if current contour is a edge candidate
      if cont_inside == 1 or cont_inside == 2:
          edge_candidates.append(t_cnt)          
          
    # We suppose that if we can find the three edges, we have a QR
    if len(edge_candidates) == 3:
        out = find_qrcont(frame, edge_candidates)
        if out is None:
            return None
        else:
            return np.array(out)
                        
    return None

def qr_wrap_perspective(img, H, qr_outside_cnt):
    s = 150
    h, w, c = img.shape    
    mymat = np.array([[s,   0,    0],
                      [0,  -s,  s],
                      [0,     0,    1]])
    
    perspect = np.zeros_like(img)
    cv2.warpPerspective(img, mymat @ np.linalg.inv(H), (w ,h), dst=perspect,  borderMode=cv2.BORDER_TRANSPARENT)
    return perspect[0:s,0:s]

def get_permutations(pts):
    perm = np.array(pts)
    rows, cols = pts.shape
    
    for i in range(0, rows-1):
        tmp = pts[i+1:rows]
        tmp = np.vstack([tmp, pts[0:i+1]])
        perm = np.append(perm, tmp)
    
    return perm.reshape(rows, rows, 2)

def qr_search_homography(imgout, qr_outside_cnt):
    pts = np.array([
               [0, 1],
               [1, 0],
               [0, 0],
               [1, 1],
              ])
    
    qr_outside_cnt = qr_outside_cnt.reshape(4,2)
    
    perms = get_permutations(qr_outside_cnt)
    print(perms)
    harr = [0, 0, 0, 0]
    # Search for the best H
    for i, perm in enumerate(perms):    
        tmpH, _ = cv2.findHomography(pts, perm, method=cv2.RANSAC, ransacReprojThreshold=5)        
        harr[i] = tmpH
    
    return harr[0]

# https://stackoverflow.com/questions/5228383/how-do-i-find-the-distance-between-two-points
def zoom(z, cont, img):
    h, w, c = img.shape
    z = -z
    
    max_dst = -1
    min_dst = 100000
    for i,pt in enumerate(cont):
        dst = math.hypot(pt[0], pt[1]) # distancia con punto 0,0
        if dst > max_dst:
            pt3 = i
            max_dst = dst
        if dst < min_dst:
            pt1 = i
            min_dst = dst
       
    max_dst = -1
    min_dst = 100000
    for i,pt in enumerate(cont):
        dst = math.hypot(pt[0]-w, pt[1]) # distancia con punto 0,w
        if dst > max_dst:
            pt4 = i
            max_dst = dst
        if dst < min_dst:
            pt2 = i   
            min_dst = dst
            
    cont[pt1] = cont[pt1] - z
    cont[pt2][0] = cont[pt2][0] + z
    cont[pt2][1] = cont[pt2][1] - z
    cont[pt3] = cont[pt3] + z
    cont[pt4][0] = cont[pt4][0] - z
    cont[pt4][1] = cont[pt4][1] + z
    
def qr(frame, cont, z_zoom):
    origimg = frame.copy()
    valid_cont = []
    
    for c in cont:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)            
            area = cv2.contourArea(c)
            if area > 300:
                a_r = w / float(h)
                if (a_r > .85 and a_r < 1.3):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 1)            
                    valid_cont.append(approx)
                    
    qr_outside_cnt = detect_qr(frame, valid_cont)                    
    
    if qr_outside_cnt is not None:
        zoom(-z_zoom, qr_outside_cnt,origimg)
        x,y,w,h = cv2.boundingRect(qr_outside_cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 3)                                      
        H = qr_search_homography(origimg, qr_outside_cnt) 
        return frame, qr_wrap_perspective(origimg, H, qr_outside_cnt), H
    
    return None

def main():
    cv2.namedWindow('Feed') 
    cv2.namedWindow('Warp') 
       
    cap = cv2.VideoCapture(0)             
    
    while True:
        _, frame = cap.read()
        
        z_zoom = 0
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cont, imgout = processing(imgin)
        cv2.imshow("Processed frame", imgout)
        cv2.waitKey(1)

        out = qr(frame, cont, z_zoom)
                        
        if out is not None:
            frame, qrimg, H = out               
            cv2.imshow('Warp', qrimg)
            cv2.waitKey(0)
            break
            
        cv2.imshow('Feed', frame)           
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    return H

def draw_corners(img, corners, ids):    
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
            cv2.line(img, topLeft, topRight, (0, 255, 0), 1)
            cv2.line(img, topRight, bottomRight, (0, 255, 0), 1)
            cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 1)
            cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 1)
            
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            center = (cX, cY)
            cv2.circle(img, center, 2, (0, 0, 255), -1)
            
            # draw the ArUco marker ID on the image
            cv2.putText(img, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            return img, center


def decode(img, H, arucoDict, arucoParams):
    ret = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)    #TODO: if None !  
    if ret[1] is None:
        return
    (corners, ids, _) =  ret        
    img, center = draw_corners(img, corners, ids)
    cv2.imshow("Frame", img)

    robot = np.asarray(center)
    robot = np.append(robot, 1)
    robot = np.reshape(robot, (3,1))
    robot = np.matmul(H, robot)
    robot = np.reshape(robot, (1, 3))
    robot[0][1] *= 20.4/150
    robot[0][0] *= 15.2/150
    print("pos robot:", robot)
    if cv2.waitKey(1) == ord("q"):
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    img = cv2.warpPerspective(img, H, (150, 150))
    cv2.imshow("warp", img)

    return

def second(H, *args):
    cv2.namedWindow('Feed') 
    cv2.namedWindow('Warp') 
       
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        decode(img, H, *args)
        
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == '__main__':
    H = main()
    key = input("Once Cozmo is set, press any litteral key to continue.")
    if key:
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
        arucoParams = cv2.aruco.DetectorParameters_create()
        second(H, arucoDict, arucoParams)
    cv2.waitKey(1)
    cv2.destroyAllWindows()