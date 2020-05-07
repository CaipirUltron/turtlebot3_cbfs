#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from qpsolvers import solve_qp

MAX_LINEAR_VELOCITY = 0.22
MAX_ANGULAR_VELOCITY = 2.84


def sat(speed, min_speed, max_speed):
    if speed > max_speed:
        speed = max_speed
    elif speed < min_speed:
        speed = min_speed
    return speed


class QPController:

    def __init__(self, type=rospy.get_param('/path_type'), speed=0.1, gains=np.array([0.6, 0.6]), freq=rospy.get_param('/rate')):
        self.path_type = type
        self.target_speed = speed
        self.control_frequency = freq
        self.control_sample_time = 1 / freq
        self.kappa = np.array([[gains[0], 0], [0, gains[1]]])
        self.gammaCLF = rospy.get_param('~gammaCLF')
        self.alphaCBF = rospy.get_param('~alphaCBF')
        self.p_gain = rospy.get_param('~p_gain')

        eps = rospy.get_param('~epsilon')
        self.delta = np.array([[1, 0], [0, eps]])
        self.eps = np.array([eps, 0])
        self.inv_delta = np.linalg.inv(self.delta)

        # Initialize state variables
        ref_x = rospy.get_param('~ref_x')
        ref_y = rospy.get_param('~ref_y')
        self.ref_point = np.array([ref_x, ref_y])
        self.p_vehicle = np.array([0, 0])
        self.theta_vehicle = 0
        self.gamma_vehicle = 0
        self.pf_error = 0

        # Initialize vehicle positions
        self.p_obstacle = np.array([0, 0])

        # Controller publisher
        self.ctrlPub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def compute_path(self):
        if self.path_type == 0:  # way-point
            pd = self.ref_point
            grad_pd = np.array([0, 0])
        elif self.path_type == 1:  # straight line
            line_x = rospy.get_param('~line_x')
            line_y = rospy.get_param('~line_y')
            line = np.array([line_x, line_y])
            norm_line = line / np.linalg.norm(line)
            pd = norm_line * self.gamma_vehicle + self.ref_point
            grad_pd = norm_line
        elif self.path_type == 2:  # circle
            r = rospy.get_param('~circle_radius')
            pd = np.array([r * np.cos(self.gamma_vehicle / r), r * np.sin(self.gamma_vehicle / r)]) + self.ref_point
            grad_pd = np.array([-np.sin(self.gamma_vehicle / r), np.cos(self.gamma_vehicle / r)])
        else:
            pd = np.array([0, 0])
            grad_pd = np.array([0, 0])

        return pd, grad_pd

    def gamma_update(self):
        self.gamma_vehicle = self.gamma_vehicle + self.control_sample_time * self.target_speed

    def controller_callback(self):
        c = np.cos(self.theta_vehicle)
        s = np.sin(self.theta_vehicle)
        rot = np.array(((c, s), (-s, c)))

        # Update gamma
        self.gamma_update()

        # Compute the path expression
        pd, grad_pd = self.compute_path()

        # Compute the pf error
        self.pf_error = rot.dot(self.p_vehicle - pd) + self.eps

        # proposed Control Lyapunov Function
        W = np.array(((1.0, 0), (0, 5.0)))
        V = 0.5 * np.dot(self.pf_error, W.dot(self.pf_error))

        # CLF constraints
        a_clf = np.concatenate((np.dot(self.pf_error, np.matmul(W,self.delta)),[-1]), axis=0)
        b_clf = np.dot(np.matmul(rot, W).dot(self.pf_error), grad_pd) * self.target_speed - self.gammaCLF * V
        b_clf = b_clf.reshape((1,))

        # Quadratic programming problem of the type: min 1/2x'Px + q'x s.t. Gx <= h, Ax = b
        # P = np.eye(3, dtype=np.double)
        P = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, self.p_gain)))
        q = np.zeros(3, dtype=np.double)

        # Control law
        # ctrl = self.inv_delta.dot(-self.kappa.dot(self.pf_error) + rot.dot(grad_pd * self.target_speed))
        ctrl = solve_qp(P, q, a_clf, b_clf, solver="quadprog")

        # Saturation on velocities
        lin_speed = ctrl[0]
        ang_speed = ctrl[1]
        delta = ctrl[2]
        # lin_speed = sat(ctrl[0], -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
        # ang_speed = sat(ctrl[1], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        # Publish twist velocity
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = lin_speed, 0.0, 0.0
        twist.angular.x, twist.angular.y, twist.angular.z = 0.0, 0.0, ang_speed
        self.ctrlPub.publish(twist)

        # rospy.loginfo("Position (%s,%s)", self.p_vehicle[0], self.p_vehicle[1])
        rospy.loginfo("Error norm %s", np.linalg.norm(self.pf_error))
        rospy.loginfo("Delta = %s", delta)
        # rospy.loginfo("Gamma = %s", self.gamma_vehicle)
        # rospy.loginfo("Control = (%s,%s)", lin_speed, ang_speed)
        # rospy.loginfo("Obstacle = (%s,%s)", self.p_obstacle[0], self.p_obstacle[1])

    def turtlebot_pose_callback(self, data):
        self.p_vehicle = np.array([data.x, data.y])
        self.theta_vehicle = data.theta

    def obstacle_callback(self, data):
        self.p_obstacle = np.array([data.x, data.y])

    def nav_goal_callback(self, data):
        self.ref_point = np.array([data.pose.position.x, data.pose.position.y])

if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)

        # Creates the controller object
        controller = PFController(2, 0.5, np.array([0.6, 0.6]), 100.0)

        # Creates subscribers for both turtlebot and obstacle poses
        poseSub = rospy.Subscriber("turtlebot3_pose", Pose2D, controller.turtlebot_pose_callback)

        LyapunovSub = rospy.Subscriber("LyapunovFun", Pose2D, controller.LyapunovCallback)
        barrierSub = rospy.Subscriber("barrierFun", Pose2D, controller.barrierCallback)

        obstacleSub = rospy.Subscriber("obstacle_pose", Pose2D, controller.obstacle_callback)
        navGoalSub = rospy.Subscriber("clicked_point", PointStamped, controller.nav_goal_callback)

        # Control frequency
        rate = rospy.Rate(controller.control_frequency)
        while not rospy.is_shutdown():
            controller.controller_callback()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
