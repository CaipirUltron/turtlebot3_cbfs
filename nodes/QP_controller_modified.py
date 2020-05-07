#!/usr/bin/env python3

import rospy
import numpy as np
import scipy.linalg as sci_linalg
from turtlebot_common.Maths import ScalarField, VectorField, AffineModel, On, delGv, projection, sigma, Gamma
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
        self.cbf = ScalarField(self.clf_dim, "")

        self.pf_error = VectorField(self.model_dim, self.clf_dim)
        self.position = VectorField(self.model_dim, self.cbf_dim)
        self.position.jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.position.list_jacobians = np.zeros((self.model_dim, self.cbf_dim, self.model_dim))

        self.model_clf_grad = np.zeros(self.model_dim)
        self.model_clf_Qgrad = np.zeros(self.rot_dim)
        self.model_cbf_grad = np.zeros(self.model_dim)

        self.distance = ScalarField(self.model_dim, "")
        self.distance.field = 0.0
        self.distance.gradient = np.zeros(self.model_dim)
        self.distance.Qgradient = np.zeros(self.rot_dim)

        self.live = ScalarField(self.model_dim, "")
        self.live.field = 0.0
        self.live.gradient = np.zeros(self.model_dim)
        self.live.Qgradient = np.zeros(self.rot_dim)

        self.model = AffineModel("unicycle")
        self.model.computeModel(np.zeros(self.model_dim))

        self.a_clf = self.a_cbf = self.a_live = np.zeros(self.ctrl_dim + self.rot_dim + 1)
        self.b_clf = self.b_cbf = self.b_live = 0.0
        self.a = np.vstack([self.a_clf, self.a_cbf, self.a_live])
        self.b = np.vstack([self.b_clf, self.b_cbf, self.b_live])

        self.distance_threshold = np.array(rospy.get_param('~threshold'))
        self.initial_tolerance = np.array(rospy.get_param('~initial_tolerance'))

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

    def cbf_callback(self, cbf_data):
        self.cbf.setField(cbf_data)

    def compute_distance(self):
        Jx, Jy, Jf = self.pf_error.jacobian, self.position.jacobian, self.model.jacobian_f_x
        list_Jx = self.pf_error.list_jacobians
        list_Jy = self.position.list_jacobians

        V, gradV, H_V = self.clf.field, self.clf.gradient, self.clf.hessian
        h, gradh, H_h = self.cbf.field, self.cbf.gradient, self.cbf.hessian

        self.model_clf_grad = np.transpose(Jx).dot(gradV)
        self.model_clf_Qgrad = [np.dot(On(self.pf_error.field), gradV)]
        self.model_cbf_grad = np.transpose(Jy).dot(gradh)

        G = np.matmul(self.model.g_x, np.transpose(self.model.g_x))
        Gx = np.matmul(Jx, np.matmul(G, np.transpose(Jx)))
        Gxy = np.matmul(Jx, np.matmul(G, np.transpose(Jy)))

        Jxf, Gx_gradV, Gxy_gradh = Jx.dot(self.model.f_x), Gx.dot(gradV), Gxy.dot(gradh)
        proj_Jxf, proj_Gx_gradV, proj_Gxy_gradh = projection(Jxf), projection(Gx_gradV), projection(Gxy_gradh)

        measure = np.matmul(Gx, np.matmul(proj_Jxf + proj_Gxy_gradh, Gx))
        self.distance.field = 0.5 * np.dot(gradV, measure.dot(gradV))

        delf_Jx = delGv(self.model.f_x, self.pf_error.list_jacobians)
        matrix1 = np.transpose(np.matmul(Jx, Jf) + delf_Jx)
        term1 = np.matmul(matrix1, proj_Gx_gradV).dot(Jxf)

        delx_G = delGv(self.model_clf_grad, self.model.list_jacobians)
        Gamma_xx = Gamma(Jx, list_Jx, Jx, list_Jx, G, gradV)
        matrix2 = np.transpose(np.matmul(Gx, np.matmul(H_V, Jx)) + Gamma_xx + np.matmul(Jx, delx_G))
        term2 = np.matmul(matrix2, proj_Jxf + proj_Gxy_gradh).dot(Gx_gradV)

        dely_G = delGv(self.model_cbf_grad, self.model.list_jacobians)
        Gamma_xy = Gamma(Jx, list_Jx, Jy, list_Jy, G, gradh)
        matrix3 = np.transpose(np.matmul(Gxy, np.matmul(H_h, Jy)) + Gamma_xy + np.matmul(Jx, dely_G))
        term3 = np.matmul(matrix3, proj_Gx_gradV).dot(Gxy_gradh)

        self.distance.gradient = term1 + term2 + term3

        matrix4 = np.transpose(np.matmul(H_V, On(self.pf_error.field)) - On(gradV))
        self.distance.Qgradient = np.matmul(matrix4, measure).dot(gradV)

        # Compute liveness
        deltaD = self.distance.field - self.distance_threshold
        sigmaValue, dsigmaValue = sigma(h)

        # self.live.field = sigmaValue * deltaD
        # self.live.gradient = sigmaValue * self.distance.gradient + dsigmaValue * deltaD * self.model_cbf_grad
        # self.live.Qgradient = sigmaValue * self.distance.Qgradient

        kappa = 10
        self.live.field = sigmaValue * deltaD + (1 - sigmaValue) * kappa
        self.live.gradient = sigmaValue * self.distance.gradient + dsigmaValue * (deltaD - kappa) * self.model_cbf_grad
        self.live.Qgradient = sigmaValue * self.distance.Qgradient

    def model_callback(self, model_msg):
        self.model.setModel(model_msg)

    def solve_QP(self):

        # Compute distance from manifold and liveness barrier
        self.compute_distance()

        # Define quadratic program
        self.a_clf = np.concatenate([np.matmul(self.model_clf_grad, self.model.g_x),
                                     self.model_clf_Qgrad,
                                     [-1.0]])
        self.b_clf = np.array([-np.matmul(self.model_clf_grad, self.model.f_x) - self.gamma * self.clf.field])

        self.a_cbf = np.concatenate([-np.matmul(self.model_cbf_grad, self.model.g_x),
                                     [0.0],
                                     [0.0]])
        self.b_cbf = np.array([np.matmul(self.model_cbf_grad, self.model.f_x) + self.alpha * self.cbf.field])

        a_WITHOUTspin = np.vstack([self.a_clf, self.a_cbf])
        b_WITHOUTspin = np.vstack([self.b_clf, self.b_cbf]).reshape((2,))

        self.a_live = np.concatenate([-np.matmul(self.live.gradient, self.model.g_x),
                                      [-self.live.Qgradient],
                                      [-0.0]])
        self.b_live = np.array([np.matmul(self.live.gradient, self.model.f_x) + self.beta * self.live.field])

        # print(self.a_live)
        # print(self.b_live)

        a_WITHspin = np.vstack([self.a_clf, self.a_cbf, self.a_live])
        b_WITHspin = np.vstack([self.b_clf, self.b_cbf, self.b_live]).reshape((3,))

        if self.spin:
            self.a = a_WITHspin
            self.b = b_WITHspin
        else:
            self.a = a_WITHOUTspin
            self.b = b_WITHOUTspin

        # Solve Quadratic Program of the type: min 1/2x'Hx s.t. ax <= b with quadprog
        # if self.spin and self.distance < self.initial_tolerance:
        # if self.spin and self.distance.field < self.initial_tolerance:
        #     self.ctrl = np.zeros(self.ctrl_dim)
        #     self.omega.data = self.omega_turn
        #     rospy.loginfo("Distance too small...")
        # else:
        #     # print(self.a_live)
        #     # print(self.b_live)
        self.QP_solution = solve_qp(self.H, np.zeros(self.ctrl_dim + self.rot_dim + 1), self.a, self.b, solver="quadprog")

        self.ctrl = self.QP_solution[:self.ctrl_dim]
        if self.spin:
            self.omega.data = self.QP_solution[self.ctrl_dim:self.ctrl_dim + self.rot_dim]
        else:
            self.omega.data = 0.0
        self.delta = self.QP_solution[-1]

        lin_speed = self.ctrl[0]
        ang_speed = self.ctrl[1]

        lin_speed = sat(self.ctrl[0], -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
        ang_speed = sat(self.ctrl[1], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        # Publish twist velocity
        self.ctrl_twist.linear.x, self.ctrl_twist.linear.y, self.ctrl_twist.linear.z = lin_speed, 0.0, 0.0
        self.ctrl_twist.angular.x, self.ctrl_twist.angular.y, self.ctrl_twist.angular.z = 0.0, 0.0, ang_speed

        # rospy.loginfo("Control = (%s,%s)", lin_speed, ang_speed)


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
        distance_pub = rospy.Publisher('distance', Field, queue_size=1)
        liveness_pub = rospy.Publisher('liveness', Field, queue_size=1)

        # Control frequency
        rate = rospy.Rate(QP_controller.control_frequency)
        rospy.sleep(0.1)
        while not rospy.is_shutdown():
            QP_controller.solve_QP()
            ctrl_pub.publish(QP_controller.ctrl_twist)
            if QP_controller.spin:
                omega_pub.publish(QP_controller.omega)
            distance_pub.publish(QP_controller.distance.sendField())
            liveness_pub.publish(QP_controller.live.sendField())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
