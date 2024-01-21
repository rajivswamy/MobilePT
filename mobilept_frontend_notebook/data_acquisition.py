import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from datetime import datetime, date
import time
from threading import Thread


# Adapted from: https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def test_fps_webcam(webcam = 1):
    cap = cv2.VideoCapture(webcam)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    num_frames = 300

    print(f"Capturing {num_frames} frames")

    start = time.time()

    for i in range(num_frames):
        ret, frame = cap.read()
        cv2.imshow("frame", frame)

    end = time.time()

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    cap.release()

    seconds = end - start

    fps = num_frames/seconds

    print(f"Estimated frame rate: {fps:.2f}")


def capture_session(name, cnt_down, session_length, base_dir = './', webcam = 1):
    cap = cv2.VideoCapture(webcam)

    # SET THE COUNTDOWN TIMER
    # for simplicity we set it to 3
    # We can also take this as input
    COUNTDOWN = cnt_down
    SESSION_LENGTH = session_length

    # Directory to store the files
    dir = base_dir
    ending = ".m4v"
    image_ending = '.jpg'

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    cv2.startWindowThread()


    # Record image with countdown
    while cap.isOpened():

        # Read and display each frame
        ret, img = cap.read()
        cv2.imshow("MobilePT", img)

        # check for the key pressed
        k = cv2.waitKey(10)

        # set the key for the countdown
        # to begin. Here we set q
        # if key pressed is q
        if k == ord("r"):

            prev = time.time()

            TIMER = COUNTDOWN

            while TIMER >= 0:
                ret, img = cap.read()

                # Display countdown on each frame
                # specify the font and draw the
                # countdown using puttext
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img, str(TIMER), (200, 250), font, 7, (0, 255, 255), 4, cv2.LINE_AA
                )
                cv2.imshow("MobilePT", img)
                cv2.waitKey(1)

                # current time
                cur = time.time()

                # Update and keep track of Countdown
                # if time elapsed is one second
                # than decrease the counter
                if cur - prev >= 1:
                    prev = cur
                    TIMER = TIMER - 1

            
            ret, img = cap.read()

            # Display the clicked frame for 2
            # sec.You can increase time in
            # waitKey also
            cv2.imshow("MobilePT", img)

            # time for which image displayed, shows the captured image to the user
            cv2.waitKey(2000)

            # Save the frame
            cv2.imwrite(os.path.join(dir,name+image_ending), img)
            
            # HERE we can reset the Countdown timer
            # if we want more Capture without closing
            # the camera
            TIMER = COUNTDOWN
            break

        # Press Esc to exit
        elif k == ord("q"):
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return None, None

    print("Entering Video Loop")

    # Record video portion 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow("MobilePT", frame)

        # check for the key pressed
        k = cv2.waitKey(1)

        # Start countdown and subsequent recording
        TIMER = COUNTDOWN

        file_path = os.path.join(dir, name + ending)

        prev = time.time()
        

        # Execute Countdown period
        while TIMER >= 0:
            ret, frame = cap.read()

            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(
                frame, str(TIMER), (200, 250), font, 7, (0, 255, 255), 4, cv2.LINE_AA
            )
            cv2.imshow("MobilePT", frame)
            cv2.waitKey(1)

            # current time
            cur = time.time()

            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrease the counter
            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        out = cv2.VideoWriter(file_path, fourcc, 30.0, (1280, 720))

        # Execute recording
        # Reset timer to 20 seconds
        TIMER = SESSION_LENGTH

        prev = time.time()

        while TIMER >= 0:
            ret, frame = cap.read()

            # write the flipped frame
            out.write(frame)

            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(
                frame, str(TIMER), (200, 250), font, 7, (0, 255, 0), 4, cv2.LINE_AA
            )
            cv2.waitKey(1)
            cv2.imshow("MobilePT", frame)

            # current time
            cur = time.time()

            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrease the counter
            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        # Release writer object
        out.release()

        # Reset timer to 10 seconds
        TIMER = COUNTDOWN
        break


    # Release everything if job is finished
    cap.release()


    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return os.path.join(dir,name+image_ending), os.path.join(dir, name + ending)


def display_video(path):
    cap = cv2.VideoCapture(path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
    
    # Read until video is completed
    while(cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            cv2.imshow('Frame', frame)
            
        # Press Q on keyboard to exit
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
    
    # Break the loop
        else:
            break
    
    # When everything done, release
    # the video capture object
    cap.release()
    
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
