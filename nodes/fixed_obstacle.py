#!/usr/bin/env python3

import rospy
import random
import roslib

roslib.load_manifest('turtlebot3_CBFs')
from visualization_msgs.msg import Marker


def place_obstacle():
    rospy.init_node('fixed_obstacle')
    rate = rospy.Rate(10)  # 10hz
    # Get the radius
    radius = rospy.get_param('~radius')

    obstacle_pub = rospy.Publisher('obstacle', Marker, queue_size=10)
    obstacle = Marker()
    obstacle.header.frame_id = "world"
    obstacle.type = obstacle.CYLINDER
    obstacle.action = obstacle.ADD
    obstacle.scale.x = radius
    obstacle.scale.y = radius
    obstacle.scale.z = 0
    obstacle.color.a = 1.0
    obstacle.color.r = 1.0
    obstacle.color.g = 0.0
    obstacle.color.b = 0.0
    obstacle.pose.position.x = random.random() * 10
    obstacle.pose.position.y = random.random() * 10
    obstacle.pose.position.z = 0
    obstacle.pose.orientation.x = 0.0
    obstacle.pose.orientation.y = 0.0
    obstacle.pose.orientation.z = 0.0
    obstacle.pose.orientation.w = 1.0

    while not rospy.is_shutdown():
        obstacle_pub.publish(obstacle)
        rate.sleep()


if __name__ == '__main__':
    try:
        place_obstacle()
    except rospy.ROSInterruptException:
        pass
