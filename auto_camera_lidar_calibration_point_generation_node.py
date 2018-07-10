#! /usr/bin/env python
import rospy

from mask_rcnn_ros.msg import Result
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Empty

from cv_bridge import CvBridge, CvBridgeError
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tf
import threading
import math

class AutoCameraLidarCalibrationPointGenerationNode:
	def __init__(self):
		self.bridge = CvBridge()

		# parameters
		self.detected_class_names = ['traffic light']
		self.vector_map_info_list = []
		self.used_vector_map_ns = ['signal']
		self.distance_threshold = 5.0
		self.viewing_angle = 40.0
		# variables
		self.screenpoint = (float(0), float(0), float(0))
		self.ndt_pose = PoseStamped()
		# publishers
		self.mask_img_pub = rospy.Publisher('~mask_img', Image, queue_size=10)
		self.screenpoint_pub = rospy.Publisher('~screenpoint', PointStamped, queue_size=10)
		self.clickedpoint_pub = rospy.Publisher('~clickedpoint', PointStamped, queue_size=10)
		#subscribers
		self.mask_rcnn_result_sub = rospy.Subscriber('~mask_rcnn_result', Result, self.mask_rcnn_result_callback)
		self.ndt_pose_sub = rospy.Subscriber('~ndt_pose', PoseStamped, self.ndt_pose_callback)
		self.vector_map_sub = rospy.Subscriber('~vector_map', MarkerArray, self.vector_map_callback)
		self.parameter_changed_sub = rospy.Subscriber('~parameter_changed', Empty, self.parameter_changed_callback)
		# tf_thread
		self.tf_cycle = 0.1
		tf_thread = threading.Thread(target=self.tf_threading)
		tf_thread.start()

	def run(self):
		if __name__ == '__main__':
			rospy.spin()
		else:
			pass

	def mask_rcnn_result_callback(self, result):
		rospy.loginfo("[MaskRCNN] Detected objects: %d", len(result.class_names))
		target_detected = False
		for (i, class_name) in enumerate(result.class_names):
			if class_name in self.detected_class_names:
				target_detected = True
				# rospy.loginfo("[MaskRCNN] ID: %d", i)
				# rospy.loginfo("[MaskRCNN] Class name: %s", result.class_names[i])
				# rospy.loginfo("[MaskRCNN] Score: %f", result.scores[i])
				# rospy.loginfo("[MaskRCNN] Bound: x_offset=%f, y_offset=%f", result.boxes[i].x_offset, result.boxes[i].y_offset)
				# rospy.loginfo("[MaskRCNN]        height=%d, width=%d, do_rectify=%s", result.boxes[i].height, result.boxes[i].width, result.boxes[i].do_rectify)
				# rospy.loginfo(result.masks[i])

				screenpoint = self.get_center_of_gravity(result.masks[i])
				print("\n")
				if(screenpoint == None):
					rospy.logwarn("center has not been obtained")
					return

				screenpoint_msg = PointStamped()
				# print(screenpoint_msg)
				screenpoint_msg.point = screenpoint
				# print(screenpoin  t_msg)
				self.screenpoint_pub.publish(screenpoint_msg)

				mask_img = copy.deepcopy(result.masks[i])
				# print(type(mask_img))
				# test_img = self.bridge.cv2_to_imgmsg(self.cv_image)
				# print(type(test_img))
				mask_img = self.bridge.cv2_to_imgmsg(self.cv_image, "mono8")
				self.mask_img_pub.publish(mask_img)

		if target_detected == False:
			rospy.loginfo("[MaskRCNN] No target object is detected")

	def get_center_of_gravity(self, img_msg):
		try:
			# plt.close()
			self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "mono8")
			# print(type(self.cv_image))
			
			mu = cv2.moments(self.cv_image, False)
			x, y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
			rospy.loginfo("[MaskRCNN] Center of gravity: (%d, %d)", x, y)

			cv2.circle(self.cv_image, (x, y), 4, 100, 2, 4)
			# plt.imshow(self.cv_image)
			# plt.colorbar()
			# plt.show()
			return (float(x), float(y), float(0))

		except CvBridgeError as e:
			print(e)
			print("image_error")
			return None
		# cv2.imshow("test_image", self.cv_image)

	def ndt_pose_callback(self, ndt_pose):
		self.ndt_pose = ndt_pose
		# rospy.loginfo("[%s] In ndt_pose_callback", self.__class__.__name__)
		# rospy.loginfo("ndt_pose:")
		# print(self.ndt_pose)
		
		# markers_count = 0
		# for i, marker in enumerate(self.vector_map_info_list):
		# 	x_squared = (marker.pose.position.x - self.ndt_pose.pose.position.x) * (marker.pose.position.x - self.ndt_pose.pose.position.x)
		# 	y_squared = (marker.pose.position.y - self.ndt_pose.pose.position.y) * (marker.pose.position.y - self.ndt_pose.pose.position.y)
		# 	z_squared = (marker.pose.position.z - self.ndt_pose.pose.position.z) * (marker.pose.position.z - self.ndt_pose.pose.position.z)
		# 	distance = math.sqrt(x_squared + y_squared + z_squared)

		# 	if distance <= self.distance_threshold:
		# 		print("marker[" + str(i) + "]:")
		# 		print("ns: " + marker.ns)
		# 		print("position:")
		# 		print(marker.pose.position)
		# 		print("distance: " + str(distance))
		# 		print(" ")

		# 		markers_count+=1
				
		# print("markers_count: " + str(markers_count))

	def vector_map_callback(self, vector_map):
		rospy.loginfo("[%s] vector_map:", self.__class__.__name__)
		# rospy.loginfo(vector_map)
		# add vector_map information to list for generating clicked point
		for i, marker in enumerate(vector_map.markers):
			if marker.ns in self.used_vector_map_ns:
				rospy.loginfo("marker[%s].ns: %s", i, marker.ns)
				self.vector_map_info_list.append(marker)
		rospy.loginfo(" ")

	def tf_threading(self):
		listener = tf.TransformListener()

		while not rospy.is_shutdown():
			# rospy.loginfo("[%s] In tf_threading", self.__class__.__name__)

			now = rospy.Time(0)
			try:
				listener.waitForTransform("/map", "/base_link", now, rospy.Duration(self.tf_cycle))
			except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				continue

			(self.trans, self.rot) = listener.lookupTransform("/map", "/base_link", now)
			# rospy.loginfo("Transform: (%f, %f, %f)", self.trans[0], self.trans[1], self.trans[2])
			# print("")

			markers_count = 0
			# print(self.vector_map_info_list)
			for i, marker in enumerate(self.vector_map_info_list):
				x_squared = (marker.pose.position.x - self.trans[0]) * (marker.pose.position.x - self.trans[0])
				y_squared = (marker.pose.position.y - self.trans[1]) * (marker.pose.position.y - self.trans[1])
				z_squared = (marker.pose.position.z - self.trans[2]) * (marker.pose.position.z - self.trans[2])
				distance = math.sqrt(x_squared + y_squared + z_squared)
				# print("distance: " + str(distance))

				if distance <= self.distance_threshold:
					# print("marker[" + str(i) + "]:")
					# print("ns: " + marker.ns)
					# print("position:")
					# print(marker.pose.position)
					# print("distance: " + str(distance))
					# print(" ")

					markers_count+=1

			if markers_count >0:
				print("markers_count: " + str(markers_count))
			else:
				pass

	def parameter_changed_callback(self, empty_msg):
		self.distance_threshold = rospy.get_param('~distance_threshold', default=self.distance_threshold)
		print("[" + self.__class__.__name__ + "] Parameters changed:")
		print("\t distance_threshold: " + self.distance_threshold)

def main():
	rospy.init_node('extrinsic_parameter_calibration_node')
	node = AutoCameraLidarCalibrationPointGenerationNode()
	node.run()

if __name__ == '__main__':
	main()
