import numpy as np
import cv2 as cv
import os
import glob
import threading
import math
import matplotlib.pyplot as plt
from object_loader import *
from collections import deque

global capture,  ret, matrix, distortion, r_vecs, t_vecs

class CameraCapture:
    def __init__(self, name, res=(640, 820)):
        self.capture = cv.VideoCapture(name, cv.CAP_DSHOW)
        self.capture.set(3, res[0])
        self.capture.set(4, res[1])
        self.q = deque()
        self.status = "init"

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        while self.status == "init":
            pass

        assert self.status == "capture", "Failed to open capture"

    # Read the latest frame and save them to be used
    def _reader(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("[error] ret")
                break

            self.q.append(frame)

            self.status = "capture"

            while len(self.q) > 1:
                self.q.popleft()

        self.status = "failed"

    # Read the latest frame
    def read(self):
        return self.q[-1]

    # Release the camera
    def release(self):
        self.capture.release()

def cameraCalibration(capture):
    # Adjust this according to what calibration pattern you are using!
    checkerBoardSize = (7, 9)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    points3D = []
    points2D = []

    objectp3d = np.zeros(
        (1, checkerBoardSize[0] * checkerBoardSize[1], 3), np.float32)

    objectp3d[0, :, :2] = np.mgrid[0:checkerBoardSize[0],
                                   0:checkerBoardSize[1]].T.reshape(-1, 2)

    while True:
        frame = capture.read()
        grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(
            grayImage, checkerBoardSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
       
        

        if ret and cv.waitKey(33) == 97:
            print("picture taken")
            points3D.append(objectp3d)
            # https://docs.opencv.org/master/dd/d92/tutorial_corner_subpixels.html
            corners2 = cv.cornerSubPix(
                grayImage, corners, (11, 11), (-1, -1), criteria)

            points2D.append(corners2)

            frame = cv.drawChessboardCorners(
                frame, checkerBoardSize, corners2, ret)
        cv.imshow("Calibrating", frame)
        if (cv.waitKey(1) & 0xFF == ord("q")) or len(points3D) > 10:
            break
    cv.destroyAllWindows()
    if len(points3D) > 1:
        return cv.calibrateCamera(
            points3D, points2D, grayImage.shape[::-1], None, None)
    else:
        return False, None, None, None, None



def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # Normalize vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # Compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)

    # Compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)


def render(frame, obj, projection, referenceImage, scale3d, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale3d
    h, w = referenceImage.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        cv.fillConvexPoly(frame, framePts, (137, 27, 211))

    return frame

# https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
# https://bitesofcode.wordpress.com/2018/09/16/augmented-reality-with-python-and-opencv-part-2/

def menu():
    global capture, ret, matrix, distortion, r_vecs, t_vecs

    capture = CameraCapture(0)

    ans=True
    while ans:
        print ("""
        1.Preparation Program
        2.Augmentation Program Normal Mode
        3.Augmentation Program Tutorial Mode
        4.Exit
        """)
        ans=input("What would you like to do?")

        if ans=="1":
            ret, matrix, distortion, r_vecs, t_vecs = cameraCalibration(capture)
            print(matrix)
        elif ans=="2":
            augmentation_program(ret, matrix, distortion, r_vecs, t_vecs, False)
        elif ans=="3":
            augmentation_program(ret, matrix, distortion, r_vecs, t_vecs, True)
        elif ans=="4":
            exit()

        elif ans !="":
            print("\n Not Valid Choice Try again") 


def found_marker(frame, sourceImagePts, referenceImage, referenceImagePts, matches, good_matches, obj, text, tutorial):
    scale3d = 80

    # Get the good key points positions
    sourcePoints = np.float32(
        [referenceImagePts[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    destinationPoints = np.float32(
        [sourceImagePts[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Obtain the homography matrix
    homography, _ = cv.findHomography(
        sourcePoints, destinationPoints, cv.RANSAC, 5.0
    )

    # Apply the perspective transformation to the source image corners
    h, w = referenceImage.shape
    corners = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
    ).reshape(-1, 1, 2)
    try:
        transformedCorners = cv.perspectiveTransform(
            corners, homography)
    except:
        return

    if(tutorial):
        matchesMask = [[0,0] for i in range(len(matches))]

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv.DrawMatchesFlags_DEFAULT)

        img3 = cv.drawMatchesKnn(
            referenceImage, referenceImagePts, frame, sourceImagePts, matches, None, **draw_params)
        plt.imshow(img3,), plt.show()

    # Draw a polygon on the second image joining the transformed corners

    # ================= Pose Estimation ================

    # obtain 3D projection matrix from homography matrix and camera parameters
    projection = projection_matrix(matrix, homography)

    # project cube or model
    frame = render(frame, obj, projection,
                    referenceImage, scale3d, False)
    
    # TODO move text to be on image
    cv.putText(frame, text, (0,200), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)

    # Draw a polygon on the second image joining the transformed corners
    frame = cv.polylines(
        frame, [np.int32(transformedCorners)
                ], True, (255, 0, 0), 3, cv.LINE_AA,
    )
    


def augmentation_program(ret, matrix, distortion, r_vecs, t_vecs, tutorial):
    if ret:

        # ============== Read data ==============

        # Load 3D model from OBJ file
        fox_obj = OBJ("./fox.obj", swapyz=True)

        one_cube_obj = OBJ("./one_cube.obj", swapyz=True)
        two_cubes_obj = OBJ("./two_cubes.obj", swapyz=True)
        three_cubes_obj = OBJ("./three_cubes.obj", swapyz=True)
        four_cubes_obj = OBJ("./four_cubes.obj", swapyz=True)
        five_cubes_obj = OBJ("./five_cubes.obj", swapyz=True)


        starwars_image = cv.imread("./starwars_poster.jpg", 0)
        avengers_image = cv.imread("./endgame_poster.jpg", 0)
        scarface_image = cv.imread("./scarface_poster.jpg", 0)
        batman_image = cv.imread("./batman_poster.jpg", 0)


        # Scale 3D model
        scale3d = 1

        sift = cv.SIFT_create()

        # Compute model keypoints and its descriptors
        starwarsImagePts, referenceImageDsc = sift.detectAndCompute(
            starwars_image, None)
        avengersImagePts, referenceImageDsc2 = sift.detectAndCompute(
            avengers_image, None)
        scarfaceImagePts, referenceImageDsc3 = sift.detectAndCompute(
            scarface_image, None)
        batmanImagePts, referenceImageDsc4 = sift.detectAndCompute(
            batman_image, None)
        

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        # TODO change this values or try explore dictionaries
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)

        MIN_MATCHES = len(starwarsImagePts) / 30
        MIN_MATCHES2 = len(avengersImagePts) / 30
        MIN_MATCHES3 = len(scarfaceImagePts) / 30
        MIN_MATCHES4 = len(batmanImagePts) / 30

        while True:
            frame = capture.read()

            # ============== Recognize =============

            # Compute scene keypoints and its descriptors
            starwarsSourceImagePts, starwarsSourceImageDsc = sift.detectAndCompute(frame, None)
            avengersSourceImagePts, avengersSourceImageDsc = sift.detectAndCompute(frame, None)
            scarfaceSourceImagePts, scarfaceSourceImageDsc = sift.detectAndCompute(frame, None)
            batmanSourceImagePts, batmanSourceImageDsc = sift.detectAndCompute(frame, None)


            # ============== Matching =============

            # Match frame descriptors with model descriptors
            try:
                starwarsMatches = flann.knnMatch(
                    referenceImageDsc, starwarsSourceImageDsc, k=2)
                avengersMatches = flann.knnMatch(
                    referenceImageDsc2, avengersSourceImageDsc, k=2)
                scarfaceMatches = flann.knnMatch(
                    referenceImageDsc3, scarfaceSourceImageDsc, k=2)
                batmanMatches = flann.knnMatch(
                    referenceImageDsc4, batmanSourceImageDsc, k=2)
            except:
                continue

            # -- Filter matches using the Lowe's ratio test
            ratio_thresh = 0.7
            starwars_good_matches = []
            avengers_good_matches = []
            scarface_good_matches = []
            batman_good_matches = []

            for m, n in starwarsMatches:
                if m.distance < ratio_thresh*n.distance:
                    starwars_good_matches.append(m)
            for m, n in avengersMatches:
                if m.distance < ratio_thresh*n.distance:
                    avengers_good_matches.append(m)
            for m, n in scarfaceMatches:
                if m.distance < ratio_thresh*n.distance:
                    scarface_good_matches.append(m)
            for m, n in batmanMatches:
                if m.distance < ratio_thresh*n.distance:
                    batman_good_matches.append(m)

            # ============== Homography =============
            # Apply the homography transformation if we have enough good matches
            if len(starwars_good_matches) > MIN_MATCHES:
                found_marker(frame, starwarsSourceImagePts, starwars_image, starwarsImagePts, starwars_good_matches, starwars_good_matches, two_cubes_obj, "Star wars TLY movie", False)  
            elif len(avengers_good_matches) > MIN_MATCHES2:
                found_marker(frame, avengersSourceImagePts, avengers_image, avengersImagePts, avengers_good_matches, avengers_good_matches, four_cubes_obj, "Avengers Endgame movie", False)                
            elif len(scarface_good_matches) > MIN_MATCHES3:
                found_marker(frame, scarfaceSourceImagePts, scarface_image, scarfaceImagePts, scarface_good_matches, scarface_good_matches, three_cubes_obj, "Scarface Movie", False)                
            elif len(batman_good_matches) > MIN_MATCHES4:
                found_marker(frame, batmanSourceImagePts, batman_image, batmanImagePts, batman_good_matches, batman_good_matches, five_cubes_obj, "Batman Movie", False)     

            
            if(tutorial):
                matchesMask = [[0,0] for i in range(len(matches))]

                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]

                draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)

                img3 = cv.drawMatchesKnn(
                    referenceImage, referenceImagePts, frame, sourceImagePts, matches, None, **draw_params)
                plt.imshow(img3,), plt.show()           

            # show result
            cv.imshow("frame", frame)
            
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        print("Camera calibration failed")
    capture.release()
    cv.destroyAllWindows()
    return 0

def main():
    menu()
    
if __name__ == "__main__":
    main()
