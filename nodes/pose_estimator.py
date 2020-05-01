#!/usr/bin/env python

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Pose2D


def compute_estimated_pose():
    # tf_listener.waitForTransform('/odom', '/base_link', rospy.Time(), rospy.Duration(5.0))

    est_pose.x = trans.transform.translation.x
    est_pose.y = trans.transform.translation.y

    quaternion = (
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w)
    rpy = tf_conversions.transformations.euler_from_quaternion(quaternion)
    est_pose.theta = rpy[2]


if __name__ == '__main__':
    try:
        rospy.init_node('pose_estimator')
        est_pose = Pose2D()

        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        pose_pub = rospy.Publisher('turtlebot_pose', Pose2D, queue_size=1)
        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        while not rospy.is_shutdown():
            try:
                trans = tf_buffer.lookup_transform('odom', 'base_link', rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue

            compute_estimated_pose()
            pose_pub.publish(est_pose)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
