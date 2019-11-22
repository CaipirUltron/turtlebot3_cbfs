#!/usr/bin/env python

import rospy
import tf
import turtlesim.msg
import numpy as np
from geometry_msgs.msg import Twist

freq = 10
gains_x, gains_y = 0.1, 0.1
v_d = 0.5

p_v = np.array([0, 0])
theta_v = 0.0


def compute_path(gamma):
    path_type = rospy.get_param('~path_type')
    ref_x = rospy.get_param('~ref_x')
    ref_y = rospy.get_param('~ref_y')
    ref_point = np.array([ref_x, ref_y])

    if path_type == 0:  # way-point
        pd = ref_point
        grad_pd = np.array([0, 0])
    elif path_type == 1:  # straight line
        line_x = rospy.get_param('~line_x')
        line_y = rospy.get_param('~line_y')
        line = np.array([line_x, line_y])
        norm_line = line / np.linalg.norm(line)
        pd = norm_line * gamma + ref_point
        grad_pd = norm_line
    elif path_type == 2:  # circle
        r = rospy.get_param('~circle_radius')
        pd = np.array([r * np.cos(gamma / r), r * np.sin(gamma / r)]) + ref_point
        grad_pd = np.array([-np.sin(gamma / r), np.cos(gamma / r)])
    else:
        pd = np.array([0, 0])
        grad_pd = np.array([0, 0])

    return pd, grad_pd


def controller(gamma):
    global freq, v_d
    global gains_x, gains_y
    global p_v, theta_v

    turtle_name = rospy.get_param('~turtle')
    pub = rospy.Publisher('/%s/cmd_vel' % turtle_name, Twist, queue_size=10)

    eps = rospy.get_param('~epsilon')
    kappa = np.array([[gains_x, 0], [0, gains_y]])
    delta = np.array([[1, 0], [0, eps]])
    inv_delta = np.linalg.inv(delta)

    c, s = np.cos(theta_v), np.sin(theta_v)
    Rt = np.array(((c, s), (-s, c)))

    h = 1 / freq
    new_gamma = gamma + h * v_d
    pd, grad_pd = compute_path(gamma)

    error = Rt.dot(p_v - pd) + np.array([eps, 0])

    ctrl = inv_delta.dot(-kappa.dot(error) + Rt.dot(grad_pd * v_d))

    twist = Twist()
    twist.linear.x = ctrl[0]
    twist.linear.y = 0.0
    twist.linear.z = 0.0

    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = ctrl[1]
    pub.publish(twist)

    rospy.loginfo("Position (%s,%s)", p_v[0], p_v[1])
    rospy.loginfo("Gamma = %s", gamma)

    return new_gamma


def handle_pose(msg, turtlename):
    global p_v, theta_v

    br = tf.TransformBroadcaster()
    br.sendTransform((msg.x, msg.y, 0),
                     tf.transformations.quaternion_from_euler(0, 0, msg.theta),
                     rospy.Time.now(),
                     turtlename,
                     "world")

    p_v = np.array([msg.x, msg.y])
    theta_v = msg.theta


if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)
        turtle_name = rospy.get_param('~turtle')
        rate = rospy.Rate(freq)
        gamma = 0.0
        while not rospy.is_shutdown():
            rospy.Subscriber('/%s/pose' % turtle_name, turtlesim.msg.Pose, handle_pose, turtle_name)
            gamma = controller(gamma)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
