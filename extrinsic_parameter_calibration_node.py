#!/usr/bin/env python

import rospy
from mask_rcnn_ros.msg import Result
import sensor_msgs.msg._Image
import copy
import cv2
from cv_bridge import CvBridge

class ExtrinsicParameterCalibrationNode:
	def __init__(self):
		self.detected_class_names = ['traffic light', 'tv']
		# publishers
		self.mask_img_pub = rospy.Publisher('~mask_img', sensor_msgs.msg.Image, queue_size=10)
		# subscribers
		self.mask_rcnn_result_sub = rospy.Subscriber('~mask_rcnn_result', Result, self.mask_rcnn_result_callback)
		pass
	
	def run(self):
		if __name__ == '__main__':
			rospy.spin()
		else:
			pass

	def mask_rcnn_result_callback(self, result):
		rospy.loginfo('detected objects: %d', len(result.class_names))
		for (i, class_name) in enumerate(result.class_names):
			if class_name in self.detected_class_names:
				rospy.loginfo(result.class_names[i])
				rospy.loginfo(result.scores[i])
				rospy.loginfo(result.boxes[i])
				# rospy.loginfo(result.masks[i])

				mask_img = copy.deepcopy(result.masks[i])
				self.mask_img_pub.publish(mask_img)

def main():
	rospy.init_node('extrinsic_parameter_calibration_node')
	node = ExtrinsicParameterCalibrationNode()
	node.run()

if __name__ == '__main__':
	main()
