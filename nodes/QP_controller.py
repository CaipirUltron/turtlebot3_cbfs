#!/usr/bin/env python3

import rospy
import numpy as np
import scipy.linalg as sci_linalg
from turtlebot_common.Maths import Field, AffineModel
from turtlebot3_cbfs.msg import ScalarField
from geometry_msgs.msg import Twist, Pose2D
from qpsolvers import solve_qp

MAX_LINEAR_VELOCITY = 0.22
MAX_ANGULAR_VELOCITY = 2.84

MIN_LINEAR_VELOCITY = -MAX_LINEAR_VELOCITY
MIN_ANGULAR_VELOCITY = -MAX_ANGULAR_VELOCITY


def sat(speed, min_speed, max_speed):
    if speed > max_speed:
        speed = max_speed
    elif speed < min_speed:
        speed = min_speed
    return speed


class QPController:

    def __init__(self, state_dim, ctrl_dim):
        self.state_dim = state_dim
        self.ctrl_dim = ctrl_dim
        self.control_frequency = rospy.get_param('~rate')
        self.control_sample_time = 1 / self.control_frequency
        self.gamma, self.alpha = rospy.get_param('~gamma'), rospy.get_param('~alpha')

        self.ctrl_cost = np.array(rospy.get_param('~ctrl_cost'))
        self.delta_cost = rospy.get_param('~delta_cost')
        self.H = sci_linalg.block_diag(self.ctrl_cost, self.delta_cost)

        self.a_clf = self.a_cbf = np.zeros(self.ctrl_dim+1)
        self.b_clf = self.b_cbf = 0.0
        self.a = np.vstack([self.a_clf, self.a_cbf])
        self.b = np.array([self.b_clf, self.b_cbf])

        self.clf = Field(self.state_dim, "")
        self.cbf = Field(self.state_dim, "")

        self.model = AffineModel("unicycle")
        self.model.computeModel(np.zeros(self.model.state_dim))

        self.QP_solution = np.zeros(self.ctrl_dim+1)
        self.ctrl, self.delta = np.zeros(self.ctrl_dim), 0.0
        self.ctrl_twist = Twist()

    def set_state(self, data):
        state = np.array([data.x, data.y, data.theta])
        self.model.computeModel(state)

    def solve_QP(self):
        # CLF constraints

        # Quadratic programming problem
        self.a_clf = np.concatenate([np.matmul(self.clf.gradient, self.model.g_x), [-1.0]])
        self.b_clf = np.array([-np.matmul(self.clf.gradient, self.model.f_x) - self.gamma * self.clf.field])

        self.a_cbf = np.concatenate([-np.matmul(self.cbf.gradient, self.model.g_x), [0.0]])
        self.b_cbf = np.array([np.matmul(self.cbf.gradient, self.model.f_x) + self.alpha * self.cbf.field])

        self.a = np.vstack([self.a_clf, self.a_cbf])
        self.b = np.vstack([self.b_clf, self.b_cbf]).reshape((2,))

        a_linear = np.array([[1, 0, 0], [-1, 0, 0]])
        b_linear = np.array([MAX_LINEAR_VELOCITY, MIN_LINEAR_VELOCITY])

        a_angular = np.array([[0, 1, 0], [0, -1, 0]])
        b_angular = np.array([MAX_ANGULAR_VELOCITY, MIN_ANGULAR_VELOCITY])

        a_sat = np.vstack([a_linear, a_angular])
        b_sat = np.concatenate([b_linear, b_angular])

        a = np.vstack([self.a_clf, a_sat])
        b = np.concatenate([self.b_clf, b_sat])

        # Solve Quadratic Program of the type: min 1/2x'Hx s.t. ax <= b with quadprog
        self.QP_solution = solve_qp(self.H, np.zeros(self.ctrl_dim + 1), self.a, self.b, solver="quadprog")

        # Saturation on velocities
        self.ctrl = self.QP_solution[:self.ctrl_dim]
        self.delta = self.QP_solution[-1]

        lin_speed = self.ctrl[0]
        ang_speed = self.ctrl[1]
        # lin_speed = sat(self.ctrl[0], -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
        # ang_speed = sat(self.ctrl[1], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        # Publish twist velocity
        self.ctrl_twist.linear.x, self.ctrl_twist.linear.y, self.ctrl_twist.linear.z = lin_speed, 0.0, 0.0
        self.ctrl_twist.angular.x, self.ctrl_twist.angular.y, self.ctrl_twist.angular.z = 0.0, 0.0, ang_speed

        # rospy.loginfo("cmd_vel = (%s,%s)", lin_speed, ang_speed)

        return self.ctrl_twist

    def clf_callback(self, clf_data):
        self.clf.setField(clf_data)

    def cbf_callback(self, cbf_data):
        self.cbf.setField(cbf_data)


if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)

        QP_controller = QPController(state_dim=rospy.get_param('/model_dim'), ctrl_dim=2)

        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, QP_controller.set_state)
        clf_sub = rospy.Subscriber("clf", ScalarField, QP_controller.clf_callback)
        cbf_sub = rospy.Subscriber("cbf", ScalarField, QP_controller.cbf_callback)
        ctrl_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # Control frequency
        rate = rospy.Rate(QP_controller.control_frequency)
        while not rospy.is_shutdown():
            ctrl_pub.publish(QP_controller.solve_QP())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
