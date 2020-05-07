#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import VectorField, rot, skew
from turtlebot3_cbfs.msg import Field
from geometry_msgs.msg import Pose2D, PointStamped


class PathFollowingControl:
    def __init__(self):
        self.pf_error = VectorField(input_dim=rospy.get_param('/model_dim'), output_dim=rospy.get_param('/clf_dim'))
        self.epsilon = np.array(rospy.get_param('~epsilon'))

        self.robot_state = np.zeros(self.pf_error.input_dim)
        self.reference = np.zeros(self.pf_error.output_dim)
        self.error = np.zeros(self.pf_error.output_dim)
        self.error_jacobian = np.zeros((self.pf_error.output_dim, self.pf_error.input_dim))
        self.error_j1_jacobian = np.zeros((self.pf_error.output_dim, self.pf_error.input_dim))
        self.error_j2_jacobian = np.zeros((self.pf_error.output_dim, self.pf_error.input_dim))
        self.error_j3_jacobian = np.zeros((self.pf_error.output_dim, self.pf_error.input_dim))

        self.pf_error.var = self.robot_state
        self.pf_error.field = self.error
        self.pf_error.jacobian = self.error_jacobian
        self.pf_error.list_jacobians[0, :, :] = self.error_j1_jacobian
        self.pf_error.list_jacobians[1, :, :] = self.error_j2_jacobian
        self.pf_error.list_jacobians[2, :, :] = self.error_j3_jacobian

    def set_state(self, data):
        self.robot_state = np.array([data.x, data.y, data.theta])
        self.pf_error.var = self.robot_state
        self.compute_path_following_error()

    def set_reference(self, data):
        self.reference = np.array([data.point.x, data.point.y])
        rospy.loginfo("Changing reference...")
        self.compute_path_following_error()

    def compute_path_following_error(self):
        robot_position = self.robot_state[:2]
        robot_orientation = self.robot_state[2]
        robot_rotation = rot(robot_orientation)

        self.error = np.transpose(robot_rotation).dot(robot_position - self.reference) + self.epsilon
        self.error_jacobian = np.concatenate(
            (np.transpose(robot_rotation), np.transpose([-skew(1).dot(self.error - self.epsilon)])), axis=1)
        self.error_j1_jacobian = np.array(([0, 0, -np.sin(robot_orientation)],
                                           [0, 0, -np.cos(robot_orientation)]))
        self.error_j2_jacobian = np.array(([0, 0,  np.cos(robot_orientation)],
                                           [0, 0, -np.sin(robot_orientation)]))
        self.error_j3_jacobian = np.array(([-np.sin(robot_orientation),  np.cos(robot_orientation), -(self.pf_error.field[0] - self.epsilon[0])],
                                           [-np.cos(robot_orientation), -np.sin(robot_orientation), -(self.pf_error.field[1] - self.epsilon[1])]))

        self.pf_error.field = self.error
        self.pf_error.jacobian = self.error_jacobian
        self.pf_error.list_jacobians[0, :, :] = self.error_j1_jacobian
        self.pf_error.list_jacobians[1, :, :] = self.error_j2_jacobian
        self.pf_error.list_jacobians[2, :, :] = self.error_j3_jacobian


if __name__ == '__main__':
    try:
        rospy.init_node('error_broadcaster', anonymous=True)

        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        PF_ctrl = PathFollowingControl()

        ref_sub = rospy.Subscriber("clicked_point", PointStamped, PF_ctrl.set_reference)
        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, PF_ctrl.set_state)
        error_pub = rospy.Publisher('error', Field, queue_size=1)

        while not rospy.is_shutdown():
            error_pub.publish(PF_ctrl.pf_error.sendField())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
