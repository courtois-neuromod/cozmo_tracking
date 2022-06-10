# cozmo_tracking
This repository implements various vision algorithms to detect and track [ArUco markers](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html). The final goal is to keep track of an Anki [Cozmo](https://www.digitaldreamlabs.com/pages/cozmo) robot's location in a maze using a top-view camera.

The most usable Python script is [`aruco_detector.py`](aruco/aruco_detector.py): it simply detects the position of an Aruco marker placed on top of the robot's head and compute its relative location with respect to the other four "boundary" ArUco markers. It does *not* perform 3D pose estimation like [here](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)(which requires to calibrate the camera, far enough from the maze and centered). Such a calibration-based alternative might be considered as further improvement. 

The ArUco detector is inspired from [this](https://pyimagesearch.com/).

# Usage 
## [`aruco_detector.py`](aruco/aruco_detector.py)
Launch the tracking script with: `python aruco_detector.py`

Possible options: `--traj` (`-t`), to pass if one wants to draw the trajectory of the robot on the video feed.

Steps:
- The script will first try to detect the 4 corner markers, and display the detected markers and wait for a key press from the user when it's done. 
- The script will then display the area of search, *i.e.* the perspective-corrected maze area. It will again wait for a key press from the user.
- The script will then display the camera feed, with on top of it the robot's position, the ArUco markers' borders an centers, and the trajectory of the robot if asked.

## [`aruco_generator.py`](aruco/aruco_generator.py)
Script from https://pyimagesearch.com/, see parser for options.
