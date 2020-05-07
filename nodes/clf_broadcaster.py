#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import ScalarField, VectorField, rot, getAngle
from turtlebot3_cbfs.msg import Field
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D


class CLFControl:
    def __init__(self):
        self.model_dim = rospy.get_param('/model_dim')
        self.clf_dim = rospy.get_param('/clf_dim')
        self.visualize = rospy.get_param('~visualize')
        if self.visualize:
            self.clf_marker = Marker()

        self.threshold = 0.000001
        self.robot_pose = np.zeros(self.model_dim)
        self.robot_position = np.zeros(2)
        self.robot_rotation = np.eye(2)

        self.pf_error = VectorField(input_dim=self.model_dim, output_dim=self.clf_dim)
        self.P = np.diag(rospy.get_param('~lambdas'))
        self.clf = ScalarField(self.clf_dim, "quadratic", 0.5 * self.P, np.zeros(self.clf_dim), 0.0)

        self.clf_angle = 0.0
        self.clf_rot = rot(self.clf_angle)
        self.omega = self.clf_dim * (self.clf_dim - 1) / 2
        self.last_omega = self.clf_dim * (self.clf_dim - 1) / 2

        self.last_time = rospy.get_time()
        self.current_time = rospy.get_time()
        self.sample_time = 0.0

    def pose_callback(self, data):
        self.robot_pose = np.array([data.x, data.y, data.theta])
        self.robot_position = self.robot_pose[:2]
        self.robot_rotation = rot(self.robot_pose[2])

    def error_callback(self, error_data):
        self.pf_error.getField(error_data)
        self.compute_modelCLF()

    # Compute V(y), nablaV_x = nablaV_y(y) Q, HessianV_x = Q' HessianV_y(y) Q, where y = Qx
    def compute_modelCLF(self):
        self.clf.computeField(self.clf_rot.dot(self.pf_error.field))
        self.clf.field = self.clf.field
        self.clf.gradient = np.matmul(self.clf.gradient, self.clf_rot)
        self.clf.hessian = np.matmul(np.transpose(self.clf_rot), np.matmul(self.clf.hessian, self.clf_rot))

    def rotateCLF(self, omega):
        self.last_time = self.current_time
        self.current_time = rospy.get_time()
        self.sample_time = self.current_time - self.last_time

        self.last_omega = self.omega
        self.omega = omega.data

        self.clf_angle = self.clf_angle + (self.sample_time / 2) * (self.omega + self.last_omega)
        self.clf_rot = rot(self.clf_angle)

        self.compute_modelCLF()

    def set_marker(self):

        # Compute ellipse parameters
        A = 0.5 * np.matmul(self.robot_rotation, np.matmul(self.clf.hessian, np.transpose(self.robot_rotation)))
        lambdas, rot = np.linalg.eig(A)
        #if np.abs(np.linalg.det(rot) - 1) > self.threshold:
        #    swap = [1, 0]
        #    rot = rot[:, swap]
        #    lambdas = lambdas[swap]
        theta = getAngle(rot)

        error_pzero = self.pf_error.field - np.transpose(self.robot_rotation).dot(self.robot_position)
        b = self.robot_rotation.dot(self.clf.hessian.dot(error_pzero))
        p_center = -0.5 * np.linalg.inv(A).dot(b)

        # c = 0.5 * np.dot(error_pzero, self.clf.hessian.dot(error_pzero))

        self.clf_marker.header.frame_id = "odom"
        self.clf_marker.type = self.clf_marker.SPHERE
        self.clf_marker.action = self.clf_marker.ADD
        self.clf_marker.scale.x = np.sqrt(1/lambdas[0])
        self.clf_marker.scale.y = np.sqrt(1/lambdas[1])
        self.clf_marker.scale.z = 0.1
        self.clf_marker.color.a = 0.5
        self.clf_marker.color.r = 0.0
        self.clf_marker.color.g = 0.0
        self.clf_marker.color.b = 1.0
        self.clf_marker.pose.position.x = p_center[0]
        self.clf_marker.pose.position.y = p_center[1]
        self.clf_marker.pose.position.z = 0
        self.clf_marker.pose.orientation.x = 0.0
        self.clf_marker.pose.orientation.y = 0.0
        self.clf_marker.pose.orientation.z = np.sin(theta/2)
        self.clf_marker.pose.orientation.w = np.cos(theta/2)


if __name__ == '__main__':
    try:
        rospy.init_node('clf_broadcaster', anonymous=True)
        rospy.loginfo("Starting CLF broadcaster...")

        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        clf_ctrl = CLFControl()

        # Subscribers
        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, clf_ctrl.pose_callback)
        error_sub = rospy.Subscriber("error", Field, clf_ctrl.error_callback)
        clf_omega = rospy.Subscriber("clf_omega", Float64, clf_ctrl.rotateCLF)

        # Publishers
        clf_pub = rospy.Publisher('clf', Field, queue_size=1)
        clf_marker_pub = rospy.Publisher('clf_visualization', Marker, queue_size=1)

        while not rospy.is_shutdown():
            clf_pub.publish(clf_ctrl.clf.sendField())
            if clf_ctrl.visualize:
                clf_ctrl.set_marker()
                clf_marker_pub.publish(clf_ctrl.clf_marker)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
