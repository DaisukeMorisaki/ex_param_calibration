#!/usr/bin/env python

import rospy
from mask_rcnn_ros.msg import Result
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

class ExtrinsicParameterCalibrationNode:
	def __init__(self):
		self.bridge = CvBridge()

		# parameters
		self.detected_class_names = ['traffic light', 'tv', 'book']
		# variables
		self.screenpoint = (float(0), float(0), float(0))
		self.ndt_pose = PoseStamped()
		# publishers
		self.mask_img_pub = rospy.Publisher('~mask_img', Image, queue_size=10)
		self.screenpoint_pub = rospy.Publisher('~screenpoint', PointStamped, queue_size=10)
		self.clickedpoint_pub = rospy.Publisher('~clickedpoint', PointStamped, queue_size=10)
		# subscribers
		self.mask_rcnn_result_sub = rospy.Subscriber('~mask_rcnn_result', Result, self.mask_rcnn_result_callback)
		self.ndt_pose_sub = rospy.Subscriber('~ndt_pose', PoseStamped, self.ndt_pose_callback)
	
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
				rospy.loginfo("[MaskRCNN] ID: %d", i)
				rospy.loginfo("[MaskRCNN] Class name: %s", result.class_names[i])
				rospy.loginfo("[MaskRCNN] Score: %f", result.scores[i])
				rospy.loginfo("[MaskRCNN] Bound: x_offset=%f, y_offset=%f", result.boxes[i].x_offset, result.boxes[i].y_offset)
				rospy.loginfo("[MaskRCNN]        height=%d, width=%d, do_rectify=%s", result.boxes[i].height, result.boxes[i].width, result.boxes[i].do_rectify)
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

def main():
	rospy.init_node('extrinsic_parameter_calibration_node')
	node = ExtrinsicParameterCalibrationNode()
	node.run()

if __name__ == '__main__':
	main()
