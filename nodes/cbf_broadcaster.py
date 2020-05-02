#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import Field, Mapping
from turtlebot3_cbfs.msg import Obstacles, ScalarField
from geometry_msgs.msg import Pose2D


class CBFControl:

    def __init__(self):
        self.barriers = []

        self.model_dim = rospy.get_param('/model_dim')
        self.model_cbf = Field(self.model_dim, "")
        self.model_state = np.zeros(self.model_dim)

        self.cbf_dim = rospy.get_param('~dim')
        self.cbf = Field(self.cbf_dim, "quadratic", np.zeros((self.cbf_dim, self.cbf_dim)), np.zeros(self.cbf_dim), 1)
        self.cbf_state = np.zeros(self.cbf_dim)
        self.cbf_jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def obstacles_callback(self, obstacles):
        self.barriers = []
        for i in range(0, obstacles.num):
            obstacle_position = np.array([obstacles.obstacles[i].pose.x, obstacles.obstacles[i].pose.y])
            axis_x = obstacles.obstacles[i].shape[0]
            axis_y = obstacles.obstacles[i].shape[1]
            a = np.diag([(1 / np.power(axis_x, 2)), (1 / np.power(axis_y, 2))])
            b = -2 * a.dot(obstacle_position)
            c = np.dot(obstacle_position, a.dot(obstacle_position)) - 1
            cbf_i = Field(self.cbf_dim, "quadratic", a, b, c)
            self.barriers.append(cbf_i)

    def set_pose(self, data):
        self.model_state = np.array([data.x, data.y, data.theta])
        self.compute_transformation()
        self.merge_CBFs()
        self.compute_modelCBF()

    def compute_transformation(self):
        self.cbf_state = self.cbf_jacobian.dot(self.model_state)
        # self.cbf_jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def merge_CBFs(self):
        self.cbf = Field(self.cbf_dim, "quadratic", np.zeros((self.cbf_dim, self.cbf_dim)), np.zeros(self.cbf_dim), 1)
        self.cbf.computeField(self.cbf_state)
        for i in range(0, len(self.barriers)):
            self.barriers[i].computeField(self.cbf_state)
            self.cbf = self.cbf * self.barriers[i]

    def compute_modelCBF(self):
        self.model_cbf.field = self.cbf.field
        self.model_cbf.gradient = np.matmul(self.cbf.gradient, self.cbf_jacobian)


if __name__ == '__main__':
    try:
        rospy.init_node('cbf_broadcaster', anonymous=True)
        rospy.loginfo("Starting CBF broadcaster...")

        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        clf_controller = CBFControl()

        obstacle_sub = rospy.Subscriber("obstacles_topic", Obstacles, clf_controller.obstacles_callback)
        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, clf_controller.set_pose)
        cbf_pub = rospy.Publisher('cbf', ScalarField, queue_size=1)

        while not rospy.is_shutdown():
            cbf_pub.publish(clf_controller.model_cbf.getField())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
