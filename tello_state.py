#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
						print_function, unicode_literals)
from builtins import *
import os
import sys
import threading
import time

import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import tello_base as tello
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def mypause(interval):
	backend = plt.rcParams['backend']
	if backend in matplotlib.rcsetup.interactive_bk:
		figManager = matplotlib._pylab_helpers.Gcf.get_active()
		if figManager is not None:
			canvas = figManager.canvas
			if canvas.figure.stale:
				canvas.draw()
			canvas.start_event_loop(interval)
			return

def find_red(frame):
	# range of red color
	lower_red = np.array([110,190,70])
	upper_red = np.array([130,255,255])

	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	imgThreshHigh = cv2.inRange(hsv, lower_red, upper_red)
	thresh = imgThreshHigh.copy()

	_,contours,_ = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	max_area=500
	found=False
	h,w=frame.shape[:2]
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > max_area:# and area>0.5*barea:
			# A = cv2.contourArea(cv2.minAreaRect(cnt))
			# if area>0.5*barea:
			rotatedRect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rotatedRect)
			box = np.int0(box)
			x1,x2,y1,y2=max(np.min(box[:,0]),0),min(np.max(box[:,0]),w),max(0,np.min(box[:,1])),min(h,np.max(box[:,1]))
			A=abs(x1-x2)*abs(y1-y2)
			w1,w2=abs(x1-x2),abs(y1-y2)
			if area>0.5*A:# and w2<1.5*w1 and w1<1.5*w2:
				max_area = area
				best_cnt = cnt
				found=True
	if not found:
		return False
	else:
		# print(max_area)
		rotatedRect = cv2.minAreaRect(best_cnt)
		box = cv2.boxPoints(rotatedRect)
		box = np.int0(box)
		x1,x2,y1,y2=max(np.min(box[:,0]),0),min(np.max(box[:,0]),w),max(0,np.min(box[:,1])),min(h,np.max(box[:,1]))
		# img=frame[y1:y2,x1:x2]
		return (x1,x2,y1,y2)#((x1+x2)//2,(y1+y2)//2)

def callback(data, drone):
	command=data.data
	drone.send_command(command)

def subscribe():
	rospy.Subscriber("command", String, callback, drone)
	#rospy.spin()

def state_updater():
	global drone,STOP,tello_state,state_dict,state_lock,state_ready,print_state
	print('State updater started')
	interval=0.5
	last_time=time.time()+interval
	try:
		while not (rospy.is_shutdown() or STOP):
			state=drone.read_state()
			if state is None or len(state) == 0:
				continue
			tello_state="".join(state)
			if state_lock.acquire(False):
				state_dict=parse_state(tello_state)
				state_ready=True
				state_lock.release()
			state_pub.publish(tello_state)
			t=time.time()
			if t>last_time and print_state:
				print(tello_state)
				last_time=t+interval
			time.sleep(0.01)
	except rospy.ROSInterruptException:
		pass

def get_state():
	global drone,STOP
	global state_dict,state_lock
	state_lock.acquire(True)
	s=state_dict.copy()
	state_lock.release()
	if s['mid']==-1 and not STOP:
		print('Lose mid, going up...')
		drone.send_command('up 50')
		time.sleep(2)
		return get_state()
	return s

def image_updater():
	print('Image updater started')
	global frame,drone,STOP,tello_state,dot_pos,dot_lock
	found_dot=False
	dot_lock.acquire(True)
	show_plt=True
	if show_plt:
		plt.ion()
	try:
		while not (rospy.is_shutdown() or STOP):
			frame = drone.read_frame()
			if frame is None or frame.size == 0:
				continue
			if show_plt:
				# try:
				img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
				red_dot=find_red(img)
				if red_dot:
					# print('find!')
					if not found_dot:
						dot_lock.release()
					found_dot=True
					x1,x2,y1,y2=red_dot
					frame[y1:y2,x1:x2,:]=255
					h,w=frame.shape[:2]
					dot_pos=((x1+x2)/w/2,(y1+y2)/h/2)
				else:
					if found_dot:
						dot_lock.acquire()
					found_dot=False
				# except Exception as e:
				# 	print(e)
				# print('image:%f'%time.time())
				cv2.imshow('image',frame)
				cv2.waitKey(20)
				# plt.imshow(frame[:,:,[2,1,0]])
				# plt.title('%d,%d'%(rx,ry))#tello_state)
				# time.sleep(0.02)
				# mypause(0.1)

	except rospy.ROSInterruptException:
		pass

def parse_state(statestr):
	state={}
	# example mid:1;x:55;y:103;z:-163;mpry:0,-20,0;pitch:-10;roll:1;yaw:94;
	# example mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;

	for item in statestr.split(';'):
		if 'mid:' in item:
			mid = int(item.split(':')[-1])
			state['mid'] = mid
		elif 'x:' in item:
			x = int(item.split(':')[-1])
			state['x'] = x
		elif 'z:' in item:
			z = int(item.split(':')[-1])
			state['z'] = z
		elif 'mpry:' in item:
			mpry = item.split(':')[-1]
			mpry = mpry.split(',')
			state['mpry'] = [int(mpry[0]),int(mpry[1]),int(mpry[2])]
		# y can be recognized as mpry, so put y first
		elif 'y:' in item:
			y = int(item.split(':')[-1])
			state['y'] = y
		elif 'pitch:' in item:
			pitch = int(item.split(':')[-1])
			state['pitch'] = pitch
		elif 'roll:' in item:
			roll = int(item.split(':')[-1])
			state['roll'] = roll
		elif 'yaw:' in item:
			yaw = int(item.split(':')[-1])
			state['yaw'] = yaw
	return state

def check_z():
	min_z=120
	max_z=180
	global drone,STOP
	if not STOP:
		s=get_state()
		z=abs(s['z'])
		if z<min_z:
			drone.send_command('up 30')
			time.sleep(1)
		elif z>max_z:
			drone.send_command('down 20')
			time.sleep(1)

	

def control():
	global drone,state_ready,STOP,dot_pos,dot_lock
	ex=drone.send_command
	ex('speed 10')
	if state_ready:
		# take off and find mid
		ex('takeoff')
		ex('up 50')
		retry=5
		while retry>0:
			if STOP: return
			print('trying to find mid-1 ... %d'%retry)
			sdict=get_state()
			if sdict['mid']!=-1:
				break
			retry-=1
			time.sleep(1)
		if sdict['mid']==-1:
			print('Failed, quit.')
			return
		Adjusted=True
		while not STOP and Adjusted:
			Adjusted=False
			check_z()
			# adjust roll angle
			eps=3
			last_roll=0
			target_roll=0 # if you change this, you'll have to modify a/w/s/d
			while True:
				if STOP: return
				print('adjust roll')
				s=get_state()
				roll=s['mpry'][1]
				if abs(roll-last_roll)<3:
					if abs(roll-target_roll)>eps:
						if roll>target_roll:
							ex('ccw %d'%(roll-target_roll))
						else:
							ex('cw %d'%(target_roll-roll))
						Adjusted=True
						check_z()#time.sleep(1)
					else:
						break
				else:
					print('roll not steady... %d'%roll)
				last_roll=roll
				time.sleep(1)
			print('Adjust roll done, roll: %d'%roll)
			Jump=False
			retry=10
			while retry>0:
				retry-=1
				if dot_lock.acquire(False): # if you can acquire it ,means it's not ready
					dot_lock.release()
					Adjusted=True
					time.sleep(0.02)
				else:
					pos=dot_pos
					Jump=True
					print('JUMP!')
					break
			if not Jump:
				# adjust x
				eps=10
				last_x=0
				target_x=70
				while True:
					if STOP: return
					print('adjust x')
					s=get_state()
					x=s['x']
					if abs(x-last_x)<10:
						if abs(x-target_x)>eps:
							if abs(x-target_x)<20:
								print('Too close x: %d'%x)
								break
								# if x>target_x:
								# 	ex('forward 20')
								# else:
								# 	ex('back 20')
							else:
								print('Flying to position from x:%d'%x)
								if x>target_x:
									ex('back %d'%(x-target_x))
								else:
									ex('forward %d'%(target_x-x))
							check_z()#time.sleep(1)
							Adjusted=True
						else:
							break
					else:
						print('x not steady... %d'%x)
					last_x=x
					time.sleep(0.2)
				print('Adjust x done, x: %d'%x)

				# adjust y
				eps=10
				last_y=0
				target_y=100
				while True:
					if STOP: return
					print('adjust y')
					s=get_state()
					y=s['y']
					if abs(y-last_y)<10:
						if abs(y-target_y)>eps:
							if abs(y-target_y)<20:
								print('Too close y: %d'%y)
								break
								# if y>target_y:
								# 	ex('left 20')
								# else:
								# 	ex('right 20')
							else:
								print('Flying to position from y:%d'%y)
								if y>target_y:
									ex('left %d'%(y-target_y))
								else:
									ex('right %d'%(target_y-y))
							check_z()#time.sleep(1)
							Adjusted=True
						else:
							break
					else:
						print('y not steady... %d'%y)
					last_y=y
					time.sleep(0.2)
				print('Adjust y done, y: %d'%y)
				if dot_lock.acquire(False): # if you can acquire it ,means it's not ready
					dot_lock.release()
					Adjusted=True
				else:
					pos=dot_pos
		while True:
			yy,zz=dot_pos
			print('adjust dot')
			if STOP: return
			ad=False
			s=get_state()
			x=s['x']
			if x>120:
				min_z=0.4
				max_z=0.9
				max_y=0.8
				min_y=0.3
			else:
				min_z=0.5
				max_z=0.8
				max_y=0.6
				min_y=0.4
			if zz>max_z:
				ex('down 20')
				ad=True
			elif zz<min_z:
				ex('up 20')
				ad=True
			if yy>max_y:
				ex('right 20')
				ad=True
			elif yy<min_y:
				ex('left 20')
				ad=True
			if ad:
				time.sleep(1)
			else:
				s=get_state()
				x=s['x']
				if x<120:
					ex('forward 30')
				else:
					ex('forward 150')
					break
			time.sleep(0.5)
		print('Done!')

	else:
		print('state is NOT ready yet, try again later.')
	# you can send command to tello without ROS, for example:(if you use this function, make sure commit "pass" above!!!)
	# drone.send_command("takeoff")
	# drone.send_command("go 0 50 0 10")
	# drone.send_command("land")
	# print drone.send_command("battery?")


if __name__ == '__main__':
	global drone,frame,state_pub,img_pub,tello_state,STOP, \
			state_lock,state_ready,print_state,dot_lock

	state_lock = threading.Lock()
	dot_lock = threading.Lock()
	STOP=False
	state_ready=False
	print_state=False

	drone = tello.Tello('', 8888)
	rospy.init_node('tello_state', anonymous=True)

	state_pub = rospy.Publisher('tello_state',String, queue_size=3)
	img_pub = rospy.Publisher('tello_image', Image, queue_size=5)

	# you can subscribe command directly, or you can just commit this function
	sub_thread = threading.Thread(target = subscribe)
	sub_thread.start()

	state_thread=threading.Thread(target=state_updater)
	state_thread.start()

	image_thread=threading.Thread(target=image_updater)
	image_thread.start()

	# you can control tello without ROS, or you can just commit this function
	# control_thread = threading.Thread(target = control)
	# con_thread.start()


	try:
		while not (rospy.is_shutdown() or STOP):
			drone.send_command('command')
			cmd=input()
			cmd1={'a':'left 20',
				  'w':'forward 20',
				  's':'back 20',
				  'd':'right 20',
				  'j':'ccw 10',
				  'i':'up 20',
				  'k':'down 20',
				  'l':'cw 10',
				  'u':'takeoff',
				  'o':'land',
				  'e':'emergency',
				  'b':'battery?',
				  'p':'stop',
				  'z':'rc 10 0 0 0',
				  'x':'rc -10 0 0 0',
				  'c':'rc 0 0 0 0'}
			if cmd=='x':
				print('Exiting...')
				STOP=True
			elif cmd=='start':
				control_thread = threading.Thread(target = control)
				control_thread.start()
			elif cmd=='echo':
				print_state=not print_state
			elif cmd in cmd1:
				threading.Thread(target=(lambda x:print(drone.send_command(x))),args=(cmd1[cmd],)).start()
			# img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			# try:
			# 	img_msg = CvBridge().cv2_to_imgmsg(img, 'bgr8')
			# 	img_msg.header.frame_id = rospy.get_namespace()
			# except CvBridgeError as err:
			# 	rospy.logerr('fgrab: cv bridge failed - %s' % str(err))
			# 	continue
			# img_pub.publish(img_msg)
	except rospy.ROSInterruptException:
		pass

