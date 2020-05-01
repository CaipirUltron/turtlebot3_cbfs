#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import Field
from turtlebot3_cbfs.msg import Obstacles, ScalarField
from geometry_msgs.msg import Pose2D


class MergeCBFs:

    def __init__(self):
        self.barriers = []
        self.dim = rospy.get_param('/dim')
        self.field = Field(self.dim, "quadratic", np.zeros((self.dim, self.dim)), np.zeros(self.dim), 1)
        self.dim = rospy.get_param('/dim')

    def obstacles_callback(self, obstacles):
        self.barriers = []
        for i in range(0, obstacles.num):
            obstacle_position = np.array([obstacles.obstacles[i].pose.x, obstacles.obstacles[i].pose.y, 0])
            axis_x = obstacles.obstacles[i].shape[0]
            axis_y = obstacles.obstacles[i].shape[1]
            a = np.diag([(1 / np.power(axis_x, 2)), (1 / np.power(axis_y, 2)), 0])
            b = -2*a.dot(obstacle_position)
            c = np.dot(obstacle_position, a.dot(obstacle_position)) - 1
            cbf = Field(self.dim, "quadratic", a, b, c)
            self.barriers.append(cbf)

    def pose_callback(self, robot_pose):
        state = np.array([robot_pose.x, robot_pose.y, robot_pose.theta])
        self.field = Field(self.dim, "quadratic", np.zeros((self.dim, self.dim)), np.zeros(self.dim), 1)
        self.field.setVar(state)
        for i in range(0, len(self.barriers)):
            self.barriers[i].setVar(state)
            self.field = self.field * self.barriers[i]


if __name__ == '__main__':
    try:
        rospy.init_node('cbf_broadcaster', anonymous=True)
        rospy.loginfo("Starting CBF broadcaster...")

        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        merged_cbf = MergeCBFs()

        obstacle_sub = rospy.Subscriber("obstacles_topic", Obstacles, merged_cbf.obstacles_callback)
        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, merged_cbf.pose_callback)
        cbf_pub = rospy.Publisher('cbf', ScalarField, queue_size=1)

        while not rospy.is_shutdown():
            cbf_pub.publish(merged_cbf.field.getField())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
