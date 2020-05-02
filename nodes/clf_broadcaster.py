#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import Field, Mapping
from turtlebot3_cbfs.msg import ScalarField
from geometry_msgs.msg import Pose2D, PointStamped


def skew(omega):
    skew_matrix = np.array(((0.0, -omega), (omega, 0.0)))
    return skew_matrix


def rot(th):
    rotation_matrix = np.array(((np.cos(th), -np.sin(th)),
                                (np.sin(th), np.cos(th))))
    return rotation_matrix


class CLFControl:
    def __init__(self):
        self.model_dim = rospy.get_param('/model_dim')
        self.model_clf = Field(self.model_dim, "")
        self.model_state = np.zeros(self.model_dim)

        self.clf_dim = rospy.get_param('~dim')
        self.clf = Field(self.clf_dim, "quadratic", 0.5 * np.diag(rospy.get_param('~lambdas')), np.zeros(self.clf_dim),
                         0.0)
        self.clf_state = np.zeros(self.clf_dim)
        self.clf_jacobian = np.zeros((self.clf_dim, self.model_dim))

        self.epsilon = np.array(rospy.get_param('~epsilon'))
        self.reference = np.zeros(self.clf_dim)

        # self.compute_transformation()

    def set_pose(self, data):
        self.model_state = np.array([data.x, data.y, data.theta])
        self.compute_transformation()

    def set_reference(self, data):
        self.reference = np.array([data.point.x, data.point.y])
        rospy.loginfo("Changing reference...")
        self.compute_transformation()

    # PF error transformation
    def compute_transformation(self):
        robot_position = self.model_state[:2]
        robot_orientation = self.model_state[2]
        robot_rotation = rot(robot_orientation)

        self.clf_state = np.transpose(robot_rotation).dot(robot_position - self.reference) + self.epsilon
        self.clf_jacobian = np.concatenate(
            (np.transpose(robot_rotation), np.transpose([-skew(1).dot(self.clf_state - self.epsilon)])), axis=1)

        self.compute_modelCLF()

    def compute_modelCLF(self):
        self.clf.computeField(self.clf_state)
        self.model_clf.field = self.clf.field
        self.model_clf.gradient = np.matmul(self.clf.gradient, self.clf_jacobian)


if __name__ == '__main__':
    try:
        rospy.init_node('clf_broadcaster', anonymous=True)
        rospy.loginfo("Starting CLF broadcaster...")

        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        clf_controller = CLFControl()

        ref_sub = rospy.Subscriber("clicked_point", PointStamped, clf_controller.set_reference)
        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, clf_controller.set_pose)
        clf_pub = rospy.Publisher('clf', ScalarField, queue_size=1)

        while not rospy.is_shutdown():
            clf_pub.publish(clf_controller.model_clf.getField())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
