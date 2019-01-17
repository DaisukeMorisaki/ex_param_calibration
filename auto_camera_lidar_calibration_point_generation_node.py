#! /usr/bin/env python
# coding:utf-8
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

import tf2_ros
import geometry_msgs.msg

class AutoCameraLidarCalibrationPointGenerationNode:
	def __init__(self):
		# parameters
		self.detected_class_names = ['traffic light']
		self.distance_threshold_for_extraction = 10.0
		self.movement_distance_threshold = 5.0
		self.viewing_angle = 40.0
		self.maximum_stored_points = 9

		self.bridge = CvBridge()

		# signal information
		self.signals = []
		self.poles   = []
		self.vectors = []
		self.points  = []

		# initialization flag
		self.broadcasted        = False
		self.point_initialized  = False
		self.vector_initialized = False
		self.pole_initialized   = False
		self.signal_initialized = False

		self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()

		# calibration points
		self.screenpoints  = []
		self.clickedpoints = []
		self.published_screenpoints  = []
		self.published_clickedpoints = []

		# pose
		self.ndt_pose = PoseStamped()

		# tf
		self.tf_listener = tf.TransformListener()
		self.trans_previous = []
		self.rot_previous = []

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

		self.debug_mode = True

		self.executed = False

		# self.broadcast_signal_tf()

	def run(self):
			rospy.spin()
			# print("DEBUG: run")
			# while not rospy.is_shutdown():
			# 	print("DEBUG: not broadcasted: " + str(not self.broadcasted))
			# 	print("DEBUG: point_initialized: " + str(self.point_initialized))
			# 	print("DEBUG: vector_initialized: " + str(self.vector_initialized))
			# 	print("DEBUG: pole_initialized: " + str(self.pole_initialized))
			# 	print("DEBUG: signal_initialized: " + str(self.signal_initialized))
			# 	if (not self.broadcasted and (self.point_initialized and self.vector_initialized and self.pole_initialized and self.signal_initialized)):
			# 		print("DEBUG: signal, vector, ... has been initialized")
			# 		self.send_broadcast_tf()
			# 		print("DEBUG: static_broadcaster have been initialized")
			# 		self.broadcasted = True
			# 	else:
			# 		print("DEBUG: initialization has not been completed")
			# 	rospy.Rate(1).sleep()
			# 	print("DEBUG: sleep")

	def mask_rcnn_result_callback(self, result):
		if self.executed == False:
			self.executed = True
			# self.broadcast_signal_tf()

		if self.debug_mode:
			print("[AutoCalib] In callback")
			print("[AutoCalib] Detected objects: " + str(len(result.class_names)))



		print("DEBUG: not broadcasted: " + str(not self.broadcasted))
		print("DEBUG: point_initialized: " + str(self.point_initialized))
		print("DEBUG: vector_initialized: " + str(self.vector_initialized))
		print("DEBUG: pole_initialized: " + str(self.pole_initialized))
		print("DEBUG: signal_initialized: " + str(self.signal_initialized))
		if (not self.broadcasted and (self.point_initialized and self.vector_initialized and self.pole_initialized and self.signal_initialized)):
			print("DEBUG: signal, vector, ... has been initialized")
			self.send_broadcast_tf()
			print("DEBUG: static_broadcaster have been initialized")
			self.broadcasted = True
		elif (self.broadcasted and self.point_initialized and self.vector_initialized and self.pole_initialized and self.signal_initialized):
			print("DEBUG: initialization has been completed")
		else:
			print("DEBUG: initialization has not been completed")




		self.get_transform()
		moved = self.check_movement_distance()
		if not moved:
			return
		
		self.extract_screen_points(result)
		# self.publish_mask_img()
		# static tf debug
		# self.extract_clicked_points_using_distance()

		self.correspond_points_with_random()

		num_of_correspond_points = self.count_correspond_points()
		if num_of_correspond_points >= self.maximum_stored_points:
			print("[AutoCalib] " + str(num_of_correspond_points) + " points has been stored.")
			self.publish_points()
		else:
			print("[AutoCalib] " + str(num_of_correspond_points) + " points has been stored. Number of points are insufficient.")

	def extract_screen_points(self, result):
		# detection for target classes
		self.object_detected_from_screen = False
		self.screenpoints = []
		for (i, class_name) in enumerate(result.class_names):
			self.object_detected_from_screen = True
			if class_name in self.detected_class_names:
				# display object information in debugging
				# if self.debug_mode:
				# 	print("[AutoCalib] target class is detected")
				# 	print("[AutoCalib] Index " + str(i) + ", class_name " + str(class_name))
				# 	print("\n")

				screenpoint = self.get_center_of_gravity(result.masks[i])
				if screenpoint == None:
					print("[AuotCalib] No center of object has not been obtained.")
					continue

				self.store_screenpoint(screenpoint)
				# publish mask_img when the object is a target object
				self.publish_mask_img()

		# if self.debug_mode:
		# 	if self.object_detected_from_screen == False:
		# 		rospy.loginfo("[AutoCalib] No target object is detected")

	def store_screenpoint(self, screenpoint):
		if screenpoint != None:
			self.screenpoints.append(screenpoint)
		else:
			if self.debug_mode:
				print("[AutoCalib] Screenpoint has not been stored (Null reference).")

	def publish_mask_img(self):
		mask_img = self.bridge.cv2_to_imgmsg(self.cv_image, "mono8")
		self.mask_img_pub.publish(mask_img)

	def get_center_of_gravity(self, img_msg):
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "mono8")

			mu = cv2.moments(self.cv_image, False)
			x, y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
			# if self.debug_mode:
			# 	print("[AutoCalib] Center of gravity: (" + str(x) + ", " + str(y) + ")")

			cv2.circle(self.cv_image, (x, y), 4, 100, 2, 4)

			return (float(x), float(y), float(0))

		except CvBridgeError as e:
			print(e)
			print("image_error occurs")
			return None

	def get_transform(self):
		now = rospy.Time(0)
		try:
			self.tf_listener.waitForTransform("/map", "/base_link", now, rospy.Duration(0.1))
			(self.trans, self.rot) = self.tf_listener.lookupTransform("/map", "/base_link", now)
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			pass
		
		if len(self.trans_previous) == 0:
			self.update_previous_transform()

	def check_movement_distance(self):
		# calculate movement distance from previous obtained transform
		movement_distance = 0
		if len(self.trans) != 0 and len(self.trans_previous) != 0:
			tf_delta = [0,1,2]
			tf_delta[0] = self.trans[1] - self.trans_previous[1]	# x
			tf_delta[1] = self.trans[0] - self.trans_previous[0]	# y
			tf_delta[2] = self.trans[2] - self.trans_previous[2]	# z
			x_squared = math.pow(tf_delta[1], 2)
			y_squared = math.pow(tf_delta[0], 2)
			z_squared = math.pow(tf_delta[2], 2)
			movement_distance = math.sqrt(x_squared + y_squared + z_squared)
			print("[AutoCalib] movement_distance: " + str(movement_distance))

		# if self.debug_mode:
		# 	print("[AutoCalib] tf: (" + str(self.trans[1]) + ", " + str(self.trans[0]) + "," + str(self.trans[2]) + ")")
		# 	print("\n")

		# check that the distance is...
		if movement_distance >= self.movement_distance_threshold:
			if self.debug_mode:
				print("[AutoCalib] " + str(self.movement_distance_threshold) + "m moved.")
			self.update_previous_transform()
			return True
		else:
			return False

	def update_previous_transform(self):
		self.trans_previous = self.trans
		self.rot_previous   = self.rot

	def extract_clicked_points_using_distance(self):
		# print("[AutoCalib] distance_threshold: " + str(self.distance_threshold_for_extraction))
		# if self.debug_mode:
		# 	print("signals: " + str(len(self.signals)))
		self.clickedpoints = []
		if (len(self.signals) != 0) and (len(self.poles) != 0) and (len(self.vectors) != 0) and (len(self.points) != 0):
			for i in range(len(self.signals)):
				# if(i == 3 or i == 13 or i == 26 or i == 39 or i == 53):
				# pole   = self.get_pole_by_plid(self.signals[i].plid)
				vector = self.get_vector_by_vid(self.signals[i].vid)
				point  = self.get_point_by_pid(vector.pid)
				# hang   = vector.hang
				# vang   = vector.vang

				x_squared = math.pow(point.bx - self.trans[1], 2)
				y_squared = math.pow(point.ly - self.trans[0], 2)
				distance  = math.sqrt(x_squared + y_squared)

				if distance <= self.distance_threshold_for_extraction:
					# signalsとsignal_bloadcastersの要素数が同じ前提で
					target_frame_string = "/signal_" + str(i)
					now = rospy.Time(0)
					try:
						self.tf_listener.waitForTransform("/velodyne", target_frame_string, now, rospy.Duration(0.1))
						(signal_trans, signal_rot) = self.tf_listener.lookupTransform("/velodyne", target_frame_string, rospy.Time(0))
					except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
						pass

					print("signal_trans: " + str(signal_trans) + ", signal_rot: " + str(signal_rot))

					# print("distance: " + str(distance) + "\n")
					# string = "Trafficlight "  + str(i) + \
					# 		" map bx " + str(point.bx) + \
					# 		" map ly " + str(point.ly) + \
					# 		" height " + str(point.h)  + \
					# 		" hang   " + str(hang) + \
					# 		" vang   " + str(vang)
					# rospy.loginfo(string)
					self.clickedpoints.append(point)

	def correspond_points_with_random(self):
		# if self.debug_mode:
		# 	print("[AutoCalib] screenpoints: " + str(len(self.screenpoints)) + ", clickedpoints: " + str(len(self.clickedpoints)))

		# select points for publishing to calib node
		if len(self.clickedpoints) == len(self.screenpoints):
			if len(self.screenpoints) != 0:
				print("screenpoints == clickedpoints")
				for i in range(len(self.screenpoints)):
					screenpoint_pop = self.screenpoints.pop(0)
					clickedpoint_pop = self.clickedpoints.pop(0)
					print("screenpoint: " + str(screenpoint_pop) + ", clickedpoint: " + str(clickedpoint_pop))
					self.published_screenpoints.append(screenpoint_pop)
					self.published_clickedpoints.append(clickedpoint_pop)
					# self.published_screenpoints.append(self.screenpoints.pop(0))
					# self.published_clickedpoints.append(self.clickedpoints.pop(0))
		elif len(self.screenpoints) < len(self.clickedpoints):
			if len(self.screenpoints) != 0:
				print("screenpoints < clickedpoints")
				# self.published_screenpoints = self.screenpoints
				for i in range(len(self.screenpoints)):
					index = random.randint(0, len(self.clickedpoints)-1)
					self.published_screenpoints.append(self.screenpoints.pop(0))
					self.published_clickedpoints.append(self.clickedpoints.pop(index))
		elif len(self.clickedpoints) < len(self.screenpoints):
			if len(self.clickedpoints) != 0:
				print("clickedpoints < screenpoints")
				for i in range(len(self.clickedpoints)):
					index = random.randint(0, len(self.screenpoints)-1)
					self.published_screenpoints.append(self.screenpoints.pop(index))
					self.published_clickedpoints.append(self.clickedpoints.pop(0))
		print("[AutoCalib] published_screenpoints: " + str(len(self.published_screenpoints)) + ", published_clickedpoints: " + str(len(self.published_clickedpoints)))


		print("\n")
		# print("screenpoints: " + len(self.screenpoints.count))
		# print("clickedpoints: " + len(self.clickedpoints.count))
		# print("\n")

	def count_correspond_points(self):
		num_of_correspond_points = 0
		if self.published_screenpoints != 0:
			num_of_correspond_points = len(self.published_screenpoints)
		elif self.published_clickedpoints != 0:
			num_of_correspond_points = len(self.published_clickedpoints)
		else:
			num_of_correspond_points = 0
		
		return num_of_correspond_points

	def publish_points(self):
		# publish messages
		print("[AutoCalib] Publishing points...")
		for i in range(len(self.published_screenpoints)):
			if self.debug_mode:
				print(type(self.published_screenpoints[i]))
				print("x: " + str(self.published_screenpoints[i][0]))
				print("y: " + str(self.published_screenpoints[i][1]))
				print("z: " + str(self.published_screenpoints[i][2]))
				print(type(self.published_clickedpoints[i]))
				print(self.published_clickedpoints[i])

			screenpoint_stamped = PointStamped()
			screenpoint_stamped.point.x = self.published_screenpoints[i][0]
			screenpoint_stamped.point.y = self.published_screenpoints[i][1]
			screenpoint_stamped.point.z = self.published_screenpoints[i][2]
			clickedpoint_stamped = PointStamped()
			clickedpoint_stamped.point.x = self.published_clickedpoints[i].bx
			clickedpoint_stamped.point.y = self.published_clickedpoints[i].ly
			clickedpoint_stamped.point.z = self.published_clickedpoints[i].h
			self.screenpoint_pub.publish(screenpoint_stamped)
			self.clickedpoint_pub.publish(clickedpoint_stamped)

			self.screenpoints  = []
			self.clickedpoints = []
			self.published_screenpoints  = []
			self.published_clickedpoints = []

	def signal_callback(self, data):
		self.signals = data.data
		rospy.loginfo("[AutoCalib] signals.length: %s", len(self.signals))
		print("DEBUG: signal initialized")
		self.signal_initialized = True
		print("DEBUG: number of signal: " + str(len(self.signals)))
		# self.broadcast_signal_tf()

	def pole_callback(self, data):
		self.poles = data.data
		print("DEBUG: pole initialized")
		self.pole_initialized = True

	def vector_callback(self, data):
		self.vectors = data.data
		print("DEBUG: vector initialized")
		self.vector_initialized = True

	def point_callback(self, data):
		self.points = data.data
		print("DEBUG: point initialized")
		self.point_initialized = True

	def get_pole_by_plid(self, plid):
		return self.poles[plid-1]

	def get_vector_by_vid(self, vid):
		return self.vectors[vid-1]

	def get_point_by_pid(self, pid):
		return self.points[pid-1]

	def send_broadcast_tf(self):
		# static broadcasters debug
		print("broadcast_signal_tf")
		self.signal_broadcasters = []

		if self.signals == []:
			print("[AutoCalib] No signals have been found")
		else:
			# for文で信号機の数だけ回して、static_broadcasterを生成
			for (i, signal) in enumerate(self.signals):
				print("DEBUG: signal" + str(i))
				# tfの設定（チュートリアルを参考に）
				# self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
					# 上のbroadcasterは1個でよい。__init__で行う
				stf_stamped = geometry_msgs.msg.TransformStamped()

				stf_stamped.header.stamp = rospy.Time.now()
				stf_stamped.header.frame_id = "map"
				child_id_string = "signal_" + str(i)
				print("DEBUG: child_id_string: " + str(child_id_string))
				stf_stamped.child_frame_id = child_id_string

				# 信号機の座標はmap基準
				# pole   = self.get_pole_by_plid(self.signals[i].plid)
				vector = self.get_vector_by_vid(self.signals[i].vid)
				point  = self.get_point_by_pid(vector.pid)

				stf_stamped.transform.translation.x = point.bx
				stf_stamped.transform.translation.y = point.ly
				stf_stamped.transform.translation.z = 0 # point.height
				# vectorの水平角からクォータニオンを生成
				rot = tf.transformations.quaternion_from_euler(0, 0, vector.hang)
				stf_stamped.transform.rotation.x = rot[0]
				stf_stamped.transform.rotation.y = rot[1]
				stf_stamped.transform.rotation.z = rot[2]
				stf_stamped.transform.rotation.w = rot[3]
				self.static_broadcaster.sendTransform(stf_stamped)

				# self.signal_broadcasters.append(self.static_broadcaster)
				appended_str = "signal broadcasted"
				print(appended_str)

				# キャリブレーションが終了したら、ブロードキャスターのオブジェクトを掃除しないといけない
				# ブロードキャスターのオブジェクトはノードを終了したら削除される？この処理はひとまず保留

def main():
	rospy.init_node('auto_camera_lidar_calibration_point_generation_node')
	node = AutoCameraLidarCalibrationPointGenerationNode()
	node.run()

if __name__ == '__main__':
	main()
