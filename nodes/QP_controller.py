#!/usr/bin/env python3

import rospy
import numpy as np
import scipy.linalg as sci_linalg
from turtlebot_common.Maths import ScalarField, VectorField, AffineModel, On, delGv, projection, sigma
from turtlebot3_cbfs.msg import Field, Model
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
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

    def __init__(self):
        self.model_dim = rospy.get_param('/model_dim')
        self.clf_dim = rospy.get_param('/clf_dim')
        self.cbf_dim = rospy.get_param('/cbf_dim')
        self.ctrl_dim = rospy.get_param('/ctrl_dim')
        self.rot_dim = round(self.clf_dim * (self.clf_dim - 1) / 2)

        self.control_frequency = rospy.get_param('~rate')
        self.control_sample_time = 1 / self.control_frequency
        self.gamma = rospy.get_param('~gamma')
        self.alpha = rospy.get_param('~alpha')
        self.beta = rospy.get_param('~beta')

        self.ctrl_cost = np.array(rospy.get_param('~ctrl_cost'))
        self.omega_cost = np.array(rospy.get_param('~omega_cost'))
        self.delta_cost = np.array(rospy.get_param('~delta_cost'))
        self.H = sci_linalg.block_diag(self.ctrl_cost, self.omega_cost, self.delta_cost)

        self.clf = ScalarField(self.clf_dim, "")
        self.clf_model = ScalarField(self.model_dim, "")
        self.clf_model_rotation_grad = np.zeros(self.rot_dim)

        self.cbf = ScalarField(self.clf_dim, "")
        self.cbf_model = ScalarField(self.model_dim, "")

        self.pf_error = VectorField(self.model_dim, self.clf_dim)
        self.position_jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        self.distance_threshold = np.array(rospy.get_param('~threshold'))
        self.initial_tolerance = np.array(rospy.get_param('~initial_tolerance'))
        self.distance = 0.0
        self.grad_distance = np.zeros(self.model_dim)
        self.gradQ_distance = np.zeros(self.rot_dim)

        self.live = 0.0
        self.live_grad = np.zeros(self.model_dim)
        self.live_gradQ = np.zeros(self.rot_dim)

        self.model = AffineModel("unicycle")
        self.model.computeModel(np.zeros(self.model_dim))

        self.a_clf = self.a_cbf = self.a_live = np.zeros(self.ctrl_dim + self.rot_dim + 1)
        self.b_clf = self.b_cbf = self.b_live = 0.0
        self.a = np.vstack([self.a_clf, self.a_cbf, self.a_live])
        self.b = np.vstack([self.b_clf, self.b_cbf, self.b_live])

        self.spin = rospy.get_param('~spin')
        self.QP_solution = np.zeros(self.ctrl_dim + self.rot_dim + 1)
        self.ctrl = np.zeros(self.ctrl_dim)
        self.omega = Float64()
        self.omega_turn = 0.1
        self.delta = 0.0
        self.ctrl_twist = Twist()

    def error_callback(self, error_data):
        self.pf_error.getField(error_data)

    def clf_callback(self, clf_data):
        self.clf.setField(clf_data)
        self.clf_model.field = self.clf.field
        self.clf_model.gradient = np.transpose(self.pf_error.jacobian).dot(self.clf.gradient)
        self.clf_model.hessian = np.matmul(np.transpose(self.pf_error.jacobian),
                                           np.matmul(self.cbf.hessian, self.pf_error.jacobian)) + delGv(
            self.clf.gradient, self.pf_error.list_jacobians)
        self.clf_model_rotation_grad = [np.transpose(On(self.pf_error.field)).dot(self.clf.gradient)]

    def cbf_callback(self, cbf_data):
        self.cbf.setField(cbf_data)
        self.cbf_model.field = self.cbf.field
        self.cbf_model.gradient = np.transpose(self.position_jacobian).dot(self.cbf.gradient)
        self.cbf_model.hessian = np.matmul(np.transpose(self.position_jacobian),
                                           np.matmul(self.cbf.hessian, self.position_jacobian))

    def compute_distance(self):
        projection_f = projection(self.model.f_x)
        G = np.matmul(self.model.g_x, np.transpose(self.model.g_x))
        projection_GV = projection(G.dot(self.clf_model.gradient))
        projection_Gh = projection(G.dot(self.cbf_model.gradient))

        measure = np.matmul(np.matmul(G, projection_f + projection_Gh), G)
        self.distance = 0.5 * np.dot(self.clf_model.gradient, measure.dot(self.clf_model.gradient))

        # print(G.dot(self.clf_model.gradient))

        term1 = np.matmul(np.transpose(self.model.jacobian_f_x), projection_GV).dot(self.model.f_x)
        term2 = np.matmul(
            np.matmul(self.clf_model.hessian, G) + np.transpose(delGv(self.clf_model.gradient, self.model.list_jacobians)),
            np.matmul(projection_f + projection_Gh, G)).dot(self.clf_model.gradient)
        term3 = np.matmul(
            np.matmul(self.cbf_model.hessian, G) + np.transpose(delGv(self.cbf_model.gradient, self.model.list_jacobians)),
            np.matmul(projection_GV, G)).dot(self.cbf_model.gradient)
        self.grad_distance = term1 + term2 + term3

        matrix = np.transpose(np.matmul(self.clf.hessian, On(self.pf_error.field)) - On(self.clf.gradient))
        self.gradQ_distance = np.matmul(matrix, np.matmul(self.pf_error.jacobian, measure)).dot(self.clf_model.gradient)

    def compute_liveness(self):
        sigmaValue, dsigmaValue = sigma(self.cbf_model.field)
        self.live = sigmaValue * (self.distance - self.distance_threshold)
        self.live_grad = sigmaValue * self.grad_distance + dsigmaValue * (self.distance - self.distance_threshold) * self.cbf_model.gradient
        self.live_gradQ = sigmaValue * self.gradQ_distance

    def model_callback(self, model_msg):
        self.model.setModel(model_msg)

    def solve_QP(self):

        # Compute distance from manifold and liveness barrier
        self.compute_distance()
        self.compute_liveness()

        # Define quadratic program
        self.a_clf = np.concatenate([np.matmul(self.clf_model.gradient, self.model.g_x),
                                     self.clf_model_rotation_grad,
                                     [-1.0]])
        self.b_clf = np.array([-np.matmul(self.clf_model.gradient, self.model.f_x) - self.gamma * self.clf.field])

        self.a_cbf = np.concatenate([-np.matmul(self.cbf_model.gradient, self.model.g_x),
                                     [0.0],
                                     [0.0]])
        self.b_cbf = np.array([np.matmul(self.cbf_model.gradient, self.model.f_x) + self.alpha * self.cbf.field])

        a_WITHOUTspin = np.vstack([self.a_clf, self.a_cbf])
        b_WITHOUTspin = np.vstack([self.b_clf, self.b_cbf]).reshape((2,))

        self.a_live = np.concatenate([-np.matmul(self.live_grad, self.model.g_x),
                                      [-self.live_gradQ],
                                      [0.0]])
        self.b_live = np.array([np.matmul(self.live_grad, self.model.f_x) + self.beta * self.live])

        a_WITHspin = np.vstack([self.a_clf, self.a_cbf, self.a_live])
        b_WITHspin = np.vstack([self.b_clf, self.b_cbf, self.b_live]).reshape((3,))

        if self.spin:
            self.a = a_WITHspin
            self.b = b_WITHspin
        else:
            self.a = a_WITHOUTspin
            self.b = b_WITHOUTspin

        # print(self.a)
        # print(self.b)

        # a_linear = np.array([[1, 0, 0], [-1, 0, 0]])
        # b_linear = np.array([MAX_LINEAR_VELOCITY, MIN_LINEAR_VELOCITY])

        # a_angular = np.array([[0, 1, 0], [0, -1, 0]])
        # b_angular = np.array([MAX_ANGULAR_VELOCITY, MIN_ANGULAR_VELOCITY])

        # a_sat = np.vstack([a_linear, a_angular])
        # b_sat = np.concatenate([b_linear, b_angular])

        # a = np.vstack([self.a_clf, a_sat])
        # b = np.concatenate([self.b_clf, b_sat])

        # Solve Quadratic Program of the type: min 1/2x'Hx s.t. ax <= b with quadprog
        if self.spin and self.distance < self.initial_tolerance:
            self.ctrl = np.zeros(self.ctrl_dim)
            self.omega.data = self.omega_turn
        else:
            self.QP_solution = solve_qp(self.H, np.zeros(self.ctrl_dim + self.rot_dim + 1), self.a, self.b,
                                        solver="quadprog")

            self.ctrl = self.QP_solution[:self.ctrl_dim]
            if self.spin:
                self.omega.data = self.QP_solution[self.ctrl_dim:self.ctrl_dim + self.rot_dim]
            else:
                self.omega.data = 0.0
            self.delta = self.QP_solution[-1]

        lin_speed = self.ctrl[0]
        ang_speed = self.ctrl[1]

        # Publish twist velocity
        self.ctrl_twist.linear.x, self.ctrl_twist.linear.y, self.ctrl_twist.linear.z = lin_speed, 0.0, 0.0
        self.ctrl_twist.angular.x, self.ctrl_twist.angular.y, self.ctrl_twist.angular.z = 0.0, 0.0, ang_speed

        # rospy.loginfo("cmd_vel = (%s,%s)", lin_speed, ang_speed)


if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)

        QP_controller = QPController()

        # Subscribers
        clf_sub = rospy.Subscriber("clf", Field, QP_controller.clf_callback)
        cbf_sub = rospy.Subscriber("cbf", Field, QP_controller.cbf_callback)
        model_sub = rospy.Subscriber("model", Model, QP_controller.model_callback)
        error_sub = rospy.Subscriber("error", Field, QP_controller.error_callback)

        # Publishers
        ctrl_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        omega_pub = rospy.Publisher('clf_omega', Float64, queue_size=1)

        # Control frequency
        rate = rospy.Rate(QP_controller.control_frequency)
        while not rospy.is_shutdown():
            QP_controller.solve_QP()
            ctrl_pub.publish(QP_controller.ctrl_twist)
            print(QP_controller.distance)
            if QP_controller.spin:
                omega_pub.publish(QP_controller.omega)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
