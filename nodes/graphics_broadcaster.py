#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot3_cbfs.msg import Obstacle, Obstacles, Field
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped


def place_obstacles():
    obstacle_poses = rospy.get_param('~poses')
    obstacle_shapes = rospy.get_param('~shapes')
    num_obstacles = len(rospy.get_param('~poses'))
    obstacles.num = num_obstacles
    if num_obstacles != len(rospy.get_param('~shapes')):
        raise ValueError('Dimension mismatch: poses or shapes of the obstacles not fully specified.')

    for i in range(0, num_obstacles):
        # Description for topic
        obstacle = Obstacle()
        obstacle.pose.x = obstacle_poses[i][0]
        obstacle.pose.y = obstacle_poses[i][1]
        obstacle.pose.theta = np.radians(obstacle_poses[i][2])
        obstacle.shape.append(obstacle_shapes[i][0])
        obstacle.shape.append(obstacle_shapes[i][1])
        obstacles.obstacles.append(obstacle)

        # Markers for Rviz
        obstacle_marker = Marker()
        obstacle_marker.id = i
        obstacle_marker.header.frame_id = "odom"
        obstacle_marker.type = obstacle_marker.SPHERE
        obstacle_marker.action = obstacle_marker.ADD
        obstacle_marker.scale.x = 2*obstacle.shape[0]
        obstacle_marker.scale.y = 2*obstacle.shape[1]
        obstacle_marker.scale.z = 0.1
        obstacle_marker.color.a = 1.0
        obstacle_marker.color.r = 0.0
        obstacle_marker.color.g = 1.0
        obstacle_marker.color.b = 0.0
        obstacle_marker.pose.position.x = obstacle.pose.x
        obstacle_marker.pose.position.y = obstacle.pose.y
        obstacle_marker.pose.position.z = 0
        obstacle_marker.pose.orientation.x = 0.0
        obstacle_marker.pose.orientation.y = 0.0
        obstacle_marker.pose.orientation.z = np.cos(obstacle.pose.theta / 2)
        obstacle_marker.pose.orientation.w = np.sin(obstacle.pose.theta / 2)

        obstacles_markerArray.markers.append(obstacle_marker)


def place_reference(data):
    ref_marker.header.frame_id = "odom"
    ref_marker.type = ref_marker.SPHERE
    ref_marker.action = ref_marker.ADD
    ref_marker.scale.x = 0.15
    ref_marker.scale.y = 0.15
    ref_marker.scale.z = 0.1
    ref_marker.color.a = 1.0
    ref_marker.color.r = 0.0
    ref_marker.color.g = 0.0
    ref_marker.color.b = 0.0
    ref_marker.pose.position.x = data.point.x
    ref_marker.pose.position.y = data.point.y
    ref_marker.pose.position.z = 0
    ref_marker.pose.orientation.x = 0.0
    ref_marker.pose.orientation.y = 0.0
    ref_marker.pose.orientation.z = 0.0
    ref_marker.pose.orientation.w = 1.0

    reference_pub.publish(ref_marker)


if __name__ == '__main__':
    try:
        obstacles_markerArray = MarkerArray()
        clf_marker = Marker()
        ref_marker = Marker()
        obstacles = Obstacles()

        # Initialize node
        rospy.init_node('obstacle', anonymous=True)
        rospy.loginfo("Starting graphics broadcaster...")

        # Subscribers
        click_sub = rospy.Subscriber("clicked_point", PointStamped, place_reference)

        # Publishers
        reference_pub = rospy.Publisher('ref_marker', Marker, queue_size=10, latch=True)
        obstacle_pub = rospy.Publisher('obstacles_topic', Obstacles, queue_size=10, latch=True)
        obstacle_marker_pub = rospy.Publisher('obstacles', MarkerArray, queue_size=10, latch=True)

        # Initialize graphics
        origin = PointStamped()
        origin.point.x = 0.0
        origin.point.y = 0.0
        place_reference(origin)
        place_obstacles()

        # Publish obstacles and stop node
        obstacle_pub.publish(obstacles)
        obstacle_marker_pub.publish(obstacles_markerArray)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
