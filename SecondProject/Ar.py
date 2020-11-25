import numpy as np
import cv2 as cv
import os
import glob
import threading
from collections import deque


class CameraCapture:
    def __init__(self, name, res=(320, 240)):
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
    checkerBoardSize = (6, 6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
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
        if ret == True:
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


def main():
    capture = CameraCapture(0)
    status = "calibrating"
    ret, matrix, distortion, r_vecs, t_vecs = cameraCalibration(capture)
    if ret:
        print(matrix)
        while True:
            frame = capture.read()
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        print("Camera calibration failed")
    capture.release()
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
