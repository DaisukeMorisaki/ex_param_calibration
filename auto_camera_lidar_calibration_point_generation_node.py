#! /usr/bin/env python
import rospy

from mask_rcnn_ros.msg import Result
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Empty
from vector_map_msgs.msg import SignalArray
from vector_map_msgs.msg import PoleArray
from vector_map_msgs.msg import VectorArray
from vector_map_msgs.msg import PointArray

from cv_bridge import CvBridge, CvBridgeError
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tf
import math
import random

class AutoCameraLidarCalibrationPointGenerationNode:
	def __init__(self):
		# parameters
		self.detected_class_names = ['traffic light']
		self.distance_threshold = 10.0
		self.viewing_angle = 40.0
		self.maximum_stored_points = 9
		# variables
		self.bridge = CvBridge()

		self.signals = []
		self.poles   = []
		self.vectors = []
		self.points  = []

		self.screenpoints = []
		self.clickedpoints = []

		self.ndt_pose = PoseStamped()

		# publishers
		self.mask_img_pub     = rospy.Publisher('~mask_img', Image, queue_size=10)
		self.screenpoint_pub  = rospy.Publisher('screenpoint', PointStamped, queue_size=10)
		self.clickedpoint_pub = rospy.Publisher('clickedpoint', PointStamped, queue_size=10)
		# subscribers
		self.mask_rcnn_result_sub  = rospy.Subscriber('mask_rcnn_result', Result, self.mask_rcnn_result_callback)
		self.vector_map_signal_sub = rospy.Subscriber('vector_map_info/signal', SignalArray, self.signal_callback)
		self.vector_map_pole_sub   = rospy.Subscriber('vector_map_info/pole', PoleArray, self.pole_callback)
		self.vector_map_vector_sub = rospy.Subscriber('vector_map_info/vector', VectorArray, self.vector_callback)
		self.vector_map_point_sub  = rospy.Subscriber('vector_map_info/point', PointArray, self.point_callback)
		# tf
		self.tf_listener = tf.TransformListener()

	def run(self):
			rospy.spin()

	def mask_rcnn_result_callback(self, result):
		print("[AutoCalib] In callback")
		print("[AutoCalib] Detected objects: " + str(len(result.class_names)))
		# detection for target classes
		target_detected = False
		self.screenpoints = []
		for (i, class_name) in enumerate(result.class_names):
			target_detected = True
			print("[AutoCalib] Index " + str(i) + ", class_name " + str(class_name))
			if class_name in self.detected_class_names:
				print("[AutoCalib] target class is detected")
				screenpoint = self.get_center_of_gravity(result.masks[i])
				print("\n")
				if screenpoint == None:
					rospy.logwarn("center has not been obtained")
					continue

				# store screenpoint to list
				self.screenpoints.append(screenpoint)

				# publish mask_img
				mask_img = self.bridge.cv2_to_imgmsg(self.cv_image, "mono8")
				self.mask_img_pub.publish(mask_img)

			# tf
			now = rospy.Time(0)
			try:
				self.tf_listener.waitForTransform("/map", "/base_link", now, rospy.Duration(0.1))
				(self.trans, self.rot) = self.tf_listener.lookupTransform("/map", "/base_link", now)
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				continue
			
			print("[AutoCalib] tf: (" + str(self.trans[1]) + ", " + str(self.trans[0]) + "," + str(self.trans[2]) + ")")
			print("\n")
			# print("[AutoCalib] distance_threshold: " + str(self.distance_threshold))
			print("signals: " + str(len(self.signals)))
			self.clickedpoints = []
			if (len(self.signals) != 0) and (len(self.poles) != 0) and (len(self.vectors) != 0) and (len(self.points) != 0):
				for i in range(len(self.signals)):
					# if(i == 3 or i == 13 or i == 26 or i == 39 or i == 53):
					pole   = self.get_pole_by_plid(self.signals[i].plid)
					vector = self.get_vector_by_vid(self.signals[i].vid)
					point  = self.get_point_by_pid(vector.pid)
					hang   = vector.hang
					vang   = vector.vang

					x_squared = math.pow(point.bx - self.trans[1], 2)
					y_squared = math.pow(point.ly - self.trans[0], 2)
					distance  = math.sqrt(x_squared + y_squared)


					if distance <= self.distance_threshold:
						print("distance: " + str(distance) + "\n")
						# string = "Trafficlight "  + str(i) + \
						# 		" map bx " + str(point.bx) + \
						# 		" map ly " + str(point.ly) + \
						# 		" height " + str(point.h)  + \
						# 		" hang   " + str(hang) + \
						# 		" vang   " + str(vang)
						# rospy.loginfo(string)
						self.clickedpoints.append(point)


		print("[AutoCalib] screenpoints: " + str(len(self.screenpoints)) + ", clickedpoints: " + str(len(self.clickedpoints)))

		# select points for publishing to calib node
		published_screenpoints = []
		published_clickedpoints = []
		if len(self.screenpoints) < len(self.clickedpoints):
			if len(self.screenpoints) != 0:
				published_screenpoints = self.screenpoints
				for i in range(len(self.screenpoints)):
					index = random.randint(0, len(self.clickedpoints)-1)
					published_clickedpoints.append(self.clickedpoints.pop(index))
		elif len(self.clickedpoints) > len(self.screenpoints):
			if len(self.clickedpoints) != 0:
				published_clickedpoints = self.clickedpoints
				for i in range(len(self.clickedpoints)):
					index = random.randint(0, len(self.screenpoints)-1)
					published_screenpoints.append(self.screenpoints.pop(index))
		elif len(self.clickedpoints) == len(self.screenpoints):
			published_screenpoints = self.screenpoints
			published_clickedpoints = self.clickedpoints

		print("[AutoCalib] published_screenpoints: " + str(len(published_screenpoints)) + ", published_clickedpoints: " + str(len(published_clickedpoints)))


		if target_detected == False:
			rospy.loginfo("[AutoCalib] No target object is detected")

		print("\n")
		# print("screenpoints: " + len(self.screenpoints.count))
		# print("clickedpoints: " + len(self.clickedpoints.count))
		# print("\n")

		for i in range(len(published_screenpoints)):
			print(type(published_screenpoints[i]))
			print("x: " + str(published_screenpoints[i][0]))
			print("y: " + str(published_screenpoints[i][1]))
			print("z: " + str(published_screenpoints[i][2]))
			print(type(published_clickedpoints[i]))
			print(published_clickedpoints[i])

			screenpoint_stamped = PointStamped()
			screenpoint_stamped.point.x = published_screenpoints[i][0]
			screenpoint_stamped.point.y = published_screenpoints[i][1]
			screenpoint_stamped.point.z = published_screenpoints[i][2]
			clickedpoint_stamped = PointStamped()
			clickedpoint_stamped.point.x = published_clickedpoints[i].bx
			clickedpoint_stamped.point.y = published_clickedpoints[i].ly
			clickedpoint_stamped.point.z = published_clickedpoints[i].h
			self.screenpoint_pub.publish(screenpoint_stamped)
			self.clickedpoint_pub.publish(clickedpoint_stamped)

	def get_center_of_gravity(self, img_msg):
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "mono8")

			mu = cv2.moments(self.cv_image, False)
			x, y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
			print("[AutoCalib] Center of gravity: (" + str(x) + ", " + str(y) + ")")

			cv2.circle(self.cv_image, (x, y), 4, 100, 2, 4)

			return (float(x), float(y), float(0))

		except CvBridgeError as e:
			print(e)
			print("image_error occurs")
			return None

	def signal_callback(self, data):
		self.signals = data.data
		rospy.loginfo("[AutoCalib] signals.length: %s", len(self.signals))

	def pole_callback(self, data):
		self.poles = data.data

	def vector_callback(self, data):
		self.vectors = data.data

	def point_callback(self, data):
		self.points = data.data

	def get_pole_by_plid(self, plid):
		return self.poles[plid-1]

	def get_vector_by_vid(self, vid):
		return self.vectors[vid-1]

	def get_point_by_pid(self, pid):
		return self.points[pid-1]

def main():
	rospy.init_node('auto_camera_lidar_calibration_point_generation_node')
	node = AutoCameraLidarCalibrationPointGenerationNode()
	node.run()

if __name__ == '__main__':
	main()
