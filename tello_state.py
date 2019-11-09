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
	global state_dict,state_lock
	state_lock.acquire(True)
	s=state_dict.copy()
	state_lock.release()
	return s

def image_updater():
	print('Image updater started')
	global frame,drone,STOP,tello_state
	show_plt=True
	if show_plt:
		plt.ion()
	try:
		while not (rospy.is_shutdown() or STOP):
			frame = drone.read_frame()
			if frame is None or frame.size == 0:
				continue
			if show_plt:
				plt.imshow(frame)
				plt.title(tello_state)
				mypause(0.1)

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

def control():
	global state_ready,STOP
	ex=drone.send_command
	if state_ready:
		# take off and find mid
		ex('takeoff')
		ex('up 50')
		retry=5
		while retry>0:
			print('trying to find mid-1 ... %d'%retry)
			sdict=get_state()
			if sdict['mid']==1:
				break
			retry-=1
			time.sleep(1)
		if sdict['mid']!=1:
			print('Failed, quit.')
			return

		# adjust roll angle
		eps=10
		last_roll=0
		while True:
			s=get_state()
			roll=s['mpry'][1]
			if abs(roll-last_roll)<3:
				if abs(roll)>eps:
					if roll>0:
						ex('ccw %d'%roll)
					else:
						ex('cw %d'%-roll)
					time.sleep(1)
				else:
					break
			else:
				print('not steady... %d'%roll)
			last_roll=roll
			time.sleep(1)
		print('Adjust done, roll: %d'%roll)

		# adjust x


	else:
		print('state is NOT ready yet, try again later.')
	# you can send command to tello without ROS, for example:(if you use this function, make sure commit "pass" above!!!)
	# drone.send_command("takeoff")
	# drone.send_command("go 0 50 0 10")
	# drone.send_command("land")
	# print drone.send_command("battery?")


if __name__ == '__main__':
	global drone,frame,state_pub,img_pub,tello_state,STOP, \
			state_lock,state_ready,print_state

	state_lock = threading.Lock()
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
			drone.send_command('Command')
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
				  'e':'emergency'}
			if cmd=='x':
				print('Exiting...')
				STOP=True
			elif cmd=='start':
				control_thread = threading.Thread(target = control)
				control_thread.start()
			elif cmd=='echo':
				print_state=not print_state
			elif cmd in cmd1:
				drone.send_command(cmd1[cmd])
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

