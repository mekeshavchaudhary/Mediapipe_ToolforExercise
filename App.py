#	Author: Keshav chaudhary
# 	contact: k9971162104@gmail.com
# 
# 	Five Classes each has it's own thread.
	# 1. Main
	# 2. GUI
	# 3. Tracker
	# 4. DB
	# 5. Analytics
# 
#	Exercises
	# 0. Bicep Curls (Left and Right)
	# 1. Bar Curls
	# 2. Pull Ups.
	# 3. Squats.

# imports start here

import threading
import time
import datetime
import math
import numpy as np
import cv2
import mediapipe as mp
import imutils
import os
import tkinter as tk
from tkinter import ttk
from threading import Thread
import subprocess
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
# imports end   here

# Global variables....

logo = """
 ____   ___  _   _____   _ 
| __ ) / _ \\| | |_   _| | |
|  _ \\| | | | |   | |   | |
| |_) | |_| | |___| |   |_|
|____/ \\___/|_____|_|   (_)
"""
raw_dimensions = [480,640] 	# height & width for laptop camera
# raw_dimensions = [600,339] 	# height & width for potrait mode mobile
# raw_dimensions = [480,846] 	# height & width for landscape mode mobile

app_root = 0

source = 0

# source = 'http://192.168.29.104:8080/video'

want_menu = False
inmenu = False
thread_already_started = False
results_pose = 0
# Global variables end here....

# Classes START....

# subprocess.Popen(['vlc-ctrl',  'volume',  '+10%'])

class FPS:

    #This class reads FPS            
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

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
            # self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True




# Classes END....

# Funcitons start here.....
def callback():
	startTracking()

def does_point_stay(trackpoint = 0, region = [[0,0],[10,10]], seconds=1, callbackFn=None):
	"""trackpoint is the number o the joint that will be tracked"""
	"""region is a 2d list [[x1,y1],[x2,y2]] i.e. area covered is (x2-x1)*(y2-y1)"""
	"""seconds is the # of seconds to wait & then see if the trackpoint is still the region."""
	"""callbackFn is the funciton that will be executed if the trackpoint stays in the region else no!"""
	cordinates = (int(results_pose.pose_landmarks.landmark[trackpoint].x*raw_dimensions[1]),int(results_pose.pose_landmarks.landmark[trackpoint].y*raw_dimensions[0]))
	print(cordinates)



def distanceCalc(a,b):
    distance = (a[0]*a[0] - b[0]*b[0]) + (a[1]*a[1] - b[1]*b[1])
    if distance<0:
        distance=distance*(-1)
    if distance==0:
        return 0
    distance = math.sqrt(distance)
    return distance

def angleCalc(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angleoutput = np.abs(radians*180.0/np.pi)
    if angleoutput > 180.0:
        angleoutput = 360.0 - angleoutput
    # print("Angle = "+str(angleoutput))
    return angleoutput

def startTracking():
	print('\n[i]\t\tTracking intiated.')
	vs = WebcamVideoStream(src=source).start()
	bicep_curls = 0
	bar_curls = 0
	bicep_down = 0
	bicep_up = 0

	bicep_curls2 = 0
	bar_curls2 = 0
	bicep_down2 = 0
	bicep_up2 = 0

	# vs = cv2.VideoCapture(0)
	fps = FPS().start()

	# size = 10
	# x_vec = np.linspace(0,1,size+1)[0:-1]
	# y_vec = np.random.randn(len(x_vec)
	size = 100
	x_vec = np.linspace(0,1,size+1)[0:-1]
	y_vec = np.random.randn(len(x_vec))
	line1 = []

	chosen_exercise = 0   #defaulting to bicep curls
	inmenu = False
	want_menu = False
	selected_exercise_menu = False
	_=10
	die = 0
	
	track_pose = True
	track_hands = False

	global results_pose
	global thread_already_started

	# COlours...
	creamy_white  = (238, 245, 219)
	wierd_red = (254, 95, 85)
	sea_green = (79, 99, 103)
	# Colours end here
	# SSS
	# with mp_pose.Pose(
	# min_detection_confidence=0.5,
	# min_tracking_confidence=0.5) as pose:
	pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
	hand = mp_hands.Hands(max_num_hands=1)
	god_loop_run = True

	mainmenu = False
	excercisemenu = False
	musicmenu = False

	while god_loop_run:
		image = vs.read()
		image = cv2.flip(image, 1)

		
		# image = imutils.resize(image, width=640)
		# image = imutils.resize(image, height=raw_dimensions[0])
		image = imutils.resize(image, height=400)

		raw_dimensions = (image.shape[0],image.shape[1])
		raw_dimensions_half = (int(raw_dimensions[0]/2),int(raw_dimensions[1]/2))

		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if track_pose:
			results_pose = pose.process(image)
		if track_hands:
			results_hands = hand.process(image)

		# Draw the pose annotation on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if track_pose:
			# mp_drawing.draw_landmarks(image,results_pose.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
			pass
		if track_hands:
			# mp_drawing.draw_landmarks(image,results_hands.pose_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
			if results_hands.multi_hand_landmarks:
				for handLms in results_hands.multi_hand_landmarks:
					# mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
					pass

		# Estimating if the user if asking for the menu...
		try:
			right_elbow = [results_pose.pose_landmarks.landmark[14].x,results_pose.pose_landmarks.landmark[14].y]
			right_wrist = [results_pose.pose_landmarks.landmark[16].x,results_pose.pose_landmarks.landmark[16].y]

			
			right_shoulder = [results_pose.pose_landmarks.landmark[12].x,results_pose.pose_landmarks.landmark[12].y]
			left_wrist = [results_pose.pose_landmarks.landmark[15].x,results_pose.pose_landmarks.landmark[15].y]
			d2 = distanceCalc(right_shoulder, left_wrist)

			# print("D2 = "+str(d2))

			if not inmenu and d2 < 0.2 and angleCalc(right_shoulder, right_elbow, right_wrist) < 180 and angleCalc(right_shoulder, right_elbow, right_wrist) > 130 :
				# We want it user to be in the menu positon for atleast 2 seconds before showing the menu, so we will 
				# make a Threaded function that will wait for 2 seconds and then chech the value of wants_menu if the value is still true
				# we will show the menu box else nothing 
				want_menu = True
				_+=1
				if _ > 20:
					inmenu = True
					mainmenu = True
					_=0

					
				cv2.rectangle(image,(0, raw_dimensions_half[0] - 50), (raw_dimensions[1], raw_dimensions_half[0] + 50), sea_green,3)

			else:
				want_menu = False

			if not want_menu :
				_=0
			selected_option = 0
			fist_open = True
			if inmenu:
				# Make the recangle Solid

				track_pose = False
				track_hands = True
				# menu dummy recangle
				

				#close menu rectangle
				cv2.rectangle(image,(raw_dimensions_half[1] - 100, raw_dimensions[0] - 60), (raw_dimensions_half[1] + 100, raw_dimensions[0] - 10), creamy_white,-1)
				cv2.putText(image,"close menu",(raw_dimensions_half[1] - 90,  raw_dimensions[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
				# Now checking that if hand comes to the menu
				ratio_cordinates1 = (0,0)
				if results_hands.multi_hand_landmarks:
					for handLms in results_hands.multi_hand_landmarks:
						ratio_cordinates1 = (int(handLms.landmark[0].x*raw_dimensions[1]),int(handLms.landmark[0].y*raw_dimensions[0]))
						ratio_cordinatesX = (int(handLms.landmark[5].x*raw_dimensions[1]),int(handLms.landmark[5].y*raw_dimensions[0]))
						ratio_cordinatesY = (int(handLms.landmark[17].x*raw_dimensions[1]),int(handLms.landmark[17].y*raw_dimensions[0]))
						circleCentre = (int((ratio_cordinates1[0]+ratio_cordinatesX[0]+ratio_cordinatesY[0])/3), int((ratio_cordinates1[1]+ratio_cordinatesX[1]+ratio_cordinatesY[1])/3) )

				# Numbere list for menu items
				# 1 - Select Exercise
				# 2 - Exit Application
				# 3 - Select Music
				# 4 - Close Menu
				# 5 - Select Bicep Curls
				# 6 - Select Bar Curls
				# 7 - Previuos Music
				# 8 - toggle Play/Pause
				# 9 - Next Song
				if mainmenu and circleCentre[1] > raw_dimensions_half[0] - 50 and circleCentre[1]< raw_dimensions_half[0] + 50:
					cv2.rectangle(image,(0, raw_dimensions_half[0] - 50), (raw_dimensions[1], raw_dimensions_half[0] + 50), sea_green,10)
					alpha_1 = int(raw_dimensions[0]/3)
					alpha_2 = int(raw_dimensions[1]/3)
					if circleCentre[0] > 0 and circleCentre[0] < alpha_2:
						cv2.rectangle(image,(0, raw_dimensions_half[0]-50), (alpha_2, raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Exercises",(10,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
						selected_option = 1
					if circleCentre[0] > alpha_2 and circleCentre[0] < alpha_2*2:
						cv2.rectangle(image,(alpha_2,raw_dimensions_half[0]-50), (alpha_2*2, raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Exit",(alpha_2+10,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)    
						selected_option = 2
					if circleCentre[0] > alpha_2*2 and circleCentre[0] < raw_dimensions[1]:
						cv2.rectangle(image,(alpha_2*2, raw_dimensions_half[0]-50), (raw_dimensions[1], raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Music",((alpha_2*2) + 10,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)    
						selected_option = 3

				if musicmenu and circleCentre[1] > raw_dimensions_half[0] - 50 and circleCentre[1]< raw_dimensions_half[0] + 50:
					
					cv2.rectangle(image,(0, raw_dimensions_half[0] - 50), (raw_dimensions[1], raw_dimensions_half[0] + 50), wierd_red,10)
					alpha_1 = int(raw_dimensions[0]/3)
					alpha_2 = int(raw_dimensions[1]/3)
					if circleCentre[0] > 0 and circleCentre[0] < alpha_2:
						cv2.rectangle(image,(0, raw_dimensions_half[0]-50), (alpha_2, raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Previuos",(10,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
						selected_option = 7
					if circleCentre[0] > alpha_2 and circleCentre[0] < alpha_2*2:
						cv2.rectangle(image,(alpha_2,raw_dimensions_half[0]-50), (alpha_2*2, raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Play/Pause",(alpha_2+10,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)    
						selected_option = 8
					if circleCentre[0] > alpha_2*2 and circleCentre[0] < raw_dimensions[1]:
						cv2.rectangle(image,(alpha_2*2, raw_dimensions_half[0]-50), (raw_dimensions[1], raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Next",(alpha_2*2 + 10,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)    
						selected_option = 9
				
				if excercisemenu and circleCentre[1] > raw_dimensions_half[0] - 50 and circleCentre[1]< raw_dimensions_half[0] + 50:
					cv2.rectangle(image,(0, raw_dimensions_half[0] - 50), (raw_dimensions[1], raw_dimensions_half[0] + 50), wierd_red,10)
					alpha_1 = int(raw_dimensions[0]/3)
					alpha_2 = int(raw_dimensions[1]/3)
					if circleCentre[0] > 0 and circleCentre[0] < raw_dimensions_half[1]:
						cv2.rectangle(image,(0, raw_dimensions_half[0]-50), (raw_dimensions_half[1], raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Bicep Curls",(20,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
						selected_option = 5
					if circleCentre[0] > 270 and circleCentre[0] < 370:
						cv2.rectangle(image,(raw_dimensions_half[1],raw_dimensions_half[0]-50), (raw_dimensions[1], raw_dimensions_half[0]+50), creamy_white,-1)
						cv2.putText(image,"Bar Curls",(raw_dimensions_half[1]+50,raw_dimensions_half[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)    
						selected_option = 6

				if circleCentre[0] > (raw_dimensions_half[1] - 100) and circleCentre[0] < (raw_dimensions_half[1] + 100)  and circleCentre[1] > (raw_dimensions[0] - 60) and circleCentre[1] < (raw_dimensions[0] - 10):
					selected_option = 4

				if results_hands.multi_hand_landmarks:
					for handLms in results_hands.multi_hand_landmarks:
						# mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

						index_A = [handLms.landmark[8].x,handLms.landmark[8].y]
						index_B = [handLms.landmark[6].x,handLms.landmark[6].y]
						index_C = [handLms.landmark[5].x,handLms.landmark[5].y]
						if fist_open and angleCalc([handLms.landmark[8].x,handLms.landmark[8].y], [handLms.landmark[6].x,handLms.landmark[6].y], [handLms.landmark[5].x,handLms.landmark[5].y]) < 20 or angleCalc([handLms.landmark[12].x,handLms.landmark[12].y], [handLms.landmark[10].x,handLms.landmark[10].y], [handLms.landmark[9].x,handLms.landmark[9].y]) < 20 :
							fist_open = False
							# Now we will decide according to the selected_option variable which option is selected , 0 is for none.
							if  selected_option == 0:
								pass
							if selected_option == 1:
								print("Exercise Selected")
								mainmenu = False
								excercisemenu = True
								musicmenu  = False
							if selected_option == 2:
								print("Exit!")
								god_loop_run = False
							if selected_option == 3:
								print("Music Selected")
								mainmenu = False
								musicmenu  = True
								excercisemenu = False


							if selected_option == 4:
								print("Closing Menu")
								inmenu = False
								musicmenu = False
								excercisemenu = False
								mainmenu = True

							if selected_option == 7:
								print("Previous Song")
								subprocess.Popen(['vlc-ctrl',  'prev'])
							if selected_option == 8:
								print("Toggle Pause/Play Music")
								subprocess.Popen(['vlc-ctrl',  'toggle'])
							if selected_option == 9:
								print("next song")
								subprocess.Popen(['vlc-ctrl',  'next'])
						if not angleCalc([handLms.landmark[8].x,handLms.landmark[8].y], [handLms.landmark[6].x,handLms.landmark[6].y], [handLms.landmark[5].x,handLms.landmark[5].y]) < 20 or not angleCalc([handLms.landmark[12].x,handLms.landmark[12].y], [handLms.landmark[10].x,handLms.landmark[10].y], [handLms.landmark[9].x,handLms.landmark[9].y]) < 20:
							fist_open = True

				cv2.circle(image, circleCentre, 15, sea_green, 3)			#tracker circle
			# Select Exercise Menu...
			
			else:
				track_hands = False
				track_pose = True
		except Exception as err:
			print("Error in Menu Pose Calculation..."+str(err))
		# We will use if else to know which excercise user choice
		if chosen_exercise == 0 and not inmenu:
			# Bicep Curls
			# print("Bicep Curl chosen")
			try:
				ratio_cordinates1 = (int(results_pose.pose_landmarks.landmark[16].x*raw_dimensions[1]),int(results_pose.pose_landmarks.landmark[16].y*raw_dimensions[0]))
				cv2.circle(image, ratio_cordinates1, 15, (255,0,255), 3)
				cv2.circle(image, ratio_cordinates1, 3, (255,255,255), 3)
				ratio_cordinates2 = (int(results_pose.pose_landmarks.landmark[14].x*raw_dimensions[1]),int(results_pose.pose_landmarks.landmark[14].y*raw_dimensions[0]))
				cv2.circle(image, ratio_cordinates2, 15, (255,255,0), 3)
				cv2.circle(image, ratio_cordinates2, 3, (255,255,255), 3)

				ratio_cordinates3 = (int(results_pose.pose_landmarks.landmark[12].x*raw_dimensions[1]),int(results_pose.pose_landmarks.landmark[12].y*raw_dimensions[0]))
				cv2.circle(image, ratio_cordinates3, 3, (255,255,0), 3)
				cv2.circle(image, ratio_cordinates3, 15, (255,255,0), 3)
				cv2.circle(image, ratio_cordinates3, 3, (255,255,255), 3)

				left_shoulder = [results_pose.pose_landmarks.landmark[12].x,results_pose.pose_landmarks.landmark[12].y]
				left_elbow = [results_pose.pose_landmarks.landmark[14].x,results_pose.pose_landmarks.landmark[14].y]
				left_wrist = [results_pose.pose_landmarks.landmark[16].x,results_pose.pose_landmarks.landmark[16].y]
				ang = angleCalc(left_shoulder,left_elbow,left_wrist)
				
				if ang < 180 and ang > 130:
					# print("Positon 1")
					if bicep_up==0:
						bicep_down=1
					if bicep_up==1:
						bicep_down=2
						bicep_up=0

				if ang > 0 and ang < 50:
					# print("Positon 2")
					bicep_up = 1

				if bicep_down==2:
					bicep_curls = bicep_curls + 1
					bicep_down=0

					# print("Bicep= "+str(bicep_curls))

				# Uncomment below for live plotting but remember that it results_pose in framerate drop!
				# rand_val = ang
				# y_vec[-1] = rand_val
				# line1 = live_plotter(x_vec,y_vec,line1)
				# y_vec = np.append(y_vec[1:],0.0)# x_verc = x_verc + 2

			except:
				print("Error")
				pass
			cv2.putText(image,"Curls =>"+str(bicep_curls),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)    
			# Flip the image horizontally for a selfie-view display.
		elif chosen_exercise == 1 and not inmenu:
			print("WWWWW")
		elif chosen_exercise == 2:
			# Bar Curls
			print("Pull Ups")
		elif chosen_exercise == 3:
			# Bar Curls
			print("Squats")
		elif not inmenu:
			print("Invalid Value")

		cv2.imshow('MediaPipe Pose',image)
		# cv2.imshow('MediaPipe Pose', image)
		
		if cv2.waitKey(5) & 0xFF == 27:
			print("Bicep= "+str(bicep_curls))
			break  

		fps.update()
			# print()
        # stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()


# Funcitons end here....
os.system('cls' if os.name == 'nt' else 'clear')	
print(logo)
print("Welcome to Bolt INC.")
print("Starting Virtual Trainer...")
print('\n\n\n\n')
print("[i]\t\tGUI intiated.")
# app_root = tk.Tk()
# # Adding tracking start button
# btn1 = ttk.Button(app_root, text="Start tracking.", command = callback)
# btn1.pack()

# app_root.mainloop()
callback()