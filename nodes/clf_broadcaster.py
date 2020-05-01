#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import Field
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
        self.dim = rospy.get_param('/dim')
        self.clf = Field(self.dim, "")
        self.error_clf = Field(self.dim-1, "quadratic", 0.5 * np.diag(rospy.get_param('~lambdas')), np.zeros(self.dim-1), 0.0)
        self.epsilon = np.array(rospy.get_param('~epsilon'))
        self.reference = np.zeros(self.dim-1)
        self.state = np.zeros(self.dim)
        self.error = np.zeros(self.dim-1)
        self.grad_error = np.zeros((self.dim, self.dim-1))
        self.compute_error()

    def set_pose(self, data):
        self.state = np.array([data.x, data.y, data.theta])
        self.compute_error()

    def set_reference(self, data):
        self.reference = np.array([data.point.x, data.point.y])
        rospy.loginfo("Changing reference...")
        self.compute_error()

    def compute_error(self):
        robot_position = self.state[:2]
        robot_orientation = self.state[2]
        robot_rotation = rot(robot_orientation)
        self.error = np.transpose(robot_rotation).dot(robot_position - self.reference) + self.epsilon
        self.grad_error = np.vstack([robot_rotation, -skew(1).dot(self.error - self.epsilon)])
        self.compute_CLF()

    def compute_CLF(self):
        self.error_clf.setVar(self.error)
        self.clf.field = self.error_clf.field
        self.clf.gradient = np.matmul(self.grad_error, self.error_clf.gradient)
        self.clf.hessian = np.zeros((self.dim, self.dim))


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
            clf_pub.publish(clf_controller.clf.getField())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
