#!/usr/bin/env python

import roslib

roslib.load_manifest('turtlebot3_cbfs')
import tf
import rospy
from geometry_msgs.msg import Pose2D


def publish_turtlebot_pose():
    tf_listener.waitForTransform('/odom', '/base_link', rospy.Time(), rospy.Duration(5.0))

    (trans, rot) = tf_listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
    euler = tf.transformations.euler_from_quaternion(rot)

    p.x = trans[0]
    p.y = trans[1]
    p.theta = euler[2]

    posePub.publish(p)


if __name__ == '__main__':
    try:
        p = Pose2D()
        rospy.init_node('turtlebot_broadcaster', anonymous=True)
        tf_listener = tf.TransformListener()
        posePub = rospy.Publisher('turtlebot3_pose', Pose2D, queue_size=10)
        pub_freq = rospy.get_param('/rate')
        rate = rospy.Rate(pub_freq)
        while not rospy.is_shutdown():
            publish_turtlebot_pose()
            # publish_obstacle_data()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
