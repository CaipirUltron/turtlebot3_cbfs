#!/usr/bin/env python

import rospy
import random
import roslib
roslib.load_manifest('turtlebot3_cbfs')
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D


def place_obstacle(radius, height):

    obstacle = Marker()
    obstacle.header.frame_id = "odom"
    obstacle.type = obstacle.CYLINDER
    obstacle.action = obstacle.ADD
    obstacle.scale.x = radius
    obstacle.scale.y = radius
    obstacle.scale.z = height
    obstacle.color.a = 1.0
    obstacle.color.r = 1.0
    obstacle.color.g = 0.0
    obstacle.color.b = 0.0
    obstacle.pose.position.x = random.random() * 4
    obstacle.pose.position.y = random.random() * 4
    obstacle.pose.position.z = height/2
    obstacle.pose.orientation.x = 0.0
    obstacle.pose.orientation.y = 0.0
    obstacle.pose.orientation.z = 0.0
    obstacle.pose.orientation.w = 1.0

    p.x = obstacle.pose.position.x
    p.y = obstacle.pose.position.y
    p.theta = 0

    while not rospy.is_shutdown():
        markerPub.publish(obstacle)
        posePub.publish(p)
        rate.sleep()


if __name__ == '__main__':
    try:
        p = Pose2D()
        rospy.init_node('fixed_obstacle')
        pub_freq = rospy.get_param('/rate')
        rate = rospy.Rate(pub_freq)
        markerPub = rospy.Publisher('obstacle', Marker, queue_size=10)
        posePub = rospy.Publisher('obstacle_pose', Pose2D, queue_size=10)

        # Get the radius and height of the obstacle
        radius = rospy.get_param('~radius')
        height = rospy.get_param('~height')

        place_obstacle(radius, height)
    except rospy.ROSInterruptException:
        pass
