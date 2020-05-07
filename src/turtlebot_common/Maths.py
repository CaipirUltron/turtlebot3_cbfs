#!/usr/bin/env python3

import numpy as np
from turtlebot3_cbfs.msg import Field, Model


def rot(th):
    rotation_matrix = np.array(((np.cos(th), -np.sin(th)),
                                (np.sin(th), np.cos(th))))
    return rotation_matrix


def getAngle(rotation_matrix):
    angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
    return angle


def skew(omega):
    dim = np.size(omega)
    if dim == 1:
        skew_matrix = np.array(((0.0, -omega), (omega, 0.0)))
    elif dim == 3:
        skew_matrix = np.array([[0.0, -omega[2], omega[1]],
                                [omega[2], 0.0, -omega[0]],
                                [-omega[1], omega[0], 0.0]])
    else:
        skew_matrix = np.array(((0.0, -omega), (omega, 0.0)))
    return skew_matrix


def On(x):
    dim = np.size(x)
    if dim == 2:
        operator = np.array([-x[1], x[0]])
    elif dim == 3:
        operator = -skew(x)
    else:
        operator = np.array([-x[1], x[0]])
    return operator


def delGv(vector, list_jacobians):
    number = len(vector)
    n = list_jacobians[0].shape[0]
    m = list_jacobians[0].shape[1]
    delta = np.zeros((n, m))
    for i in range(0, number):
        delta = delta + vector[i] * list_jacobians[i, :, :]

    return delta


def Gamma(X, listX, Y, listY, G, vector):
    nx, ny, n = X.shape[0], Y.shape[0], X.shape[1]
    gamma = np.zeros((nx, n))
    for i in range(0, nx):
        for j in range(0, n):
            gamma = gamma + G[i, j] * (np.dot(Y[:, j], vector) * listX[i, :, :] + np.outer(X[:, i], np.transpose(
                listY[j, :, :]).dot(vector)))
    return gamma


def projection(vector):
    dim = len(vector)
    P = np.power(np.linalg.norm(vector), 2) * np.eye(dim) - np.outer(vector, vector)
    return P


def sigma(data):
    gamma = 1
    f = np.exp(-np.power(data/gamma, 2))
    df = -(2*data/np.power(gamma, 2)) * f
    # f = -(1/gamma)*data + 1
    # df = -(1/gamma)
    return f, df


class ScalarField:

    def __init__(self, dim, field_type, *args):
        self.dim = dim
        self.field_type = field_type

        if self.field_type == "const":
            self.c = args[0]
        elif self.field_type == "linear":
            self.b = args[0]
            self.c = args[1]
        elif self.field_type == "quadratic":
            self.a = args[0]
            self.b = args[1]
            self.c = args[2]

        self.var = np.zeros(self.dim)
        self.field = 0.0
        self.gradient = np.zeros(self.dim)
        self.Qgradient = 0.0
        self.hessian = np.zeros([self.dim, self.dim])
        self.computeField(self.var)

    def __add__(self, other):
        if self.dim != other.dim:
            raise ValueError('Dimension mismatch. Cannot sum fields with a different number of dimensions.')

        if not np.array_equal(self.var, other.var):
            raise ValueError('Fields computed at different points.')

        new_field = ScalarField(self.dim, "")
        new_field.var = self.var
        new_field.field = self.field + other.field
        new_field.gradient = self.gradient + other.gradient
        new_field.Qgradient = self.Qgradient + other.Qgradient
        new_field.hessian = self.hessian + other.hessian

        return new_field

    def __sub__(self, other):
        if self.dim != other.dim:
            raise ValueError('Dimension mismatch. Cannot sum fields with a different number of dimensions.')

        if not np.array_equal(self.var, other.var):
            raise ValueError('Fields computed at different points.')

        new_field = ScalarField(self.dim, "")
        new_field.var = self.var
        new_field.field = self.field - other.field
        new_field.gradient = self.gradient - other.gradient
        new_field.Qgradient = self.Qgradient - other.Qgradient
        new_field.hessian = self.hessian - other.hessian

        return new_field

    def __mul__(self, other):
        if self.dim != other.dim:
            raise ValueError('Dimension mismatch. Cannot sum fields with a different number of dimensions.')

        if not np.array_equal(self.var, other.var):
            raise ValueError('Fields computed at different points.')

        new_field = ScalarField(self.dim, "")
        new_field.var = self.var
        new_field.field = self.field * other.field
        new_field.gradient = self.gradient * other.field + other.gradient * self.field
        grad1grad2 = np.outer(self.gradient, other.gradient)
        new_field.Qgradient = self.Qgradient * other.Qgradient
        new_field.hessian = self.hessian * other.field + grad1grad2 + np.transpose(
            grad1grad2) + self.field * other.hessian

        return new_field

    def computeField(self, data):
        self.var = data
        if self.field_type == "const":
            self.field = self.c
            self.gradient = np.zeros(self.dim)
            self.hessian = np.zeros((self.dim, self.dim))
        elif self.field_type == "linear":
            self.field = self.b.dot(data) + self.c
            self.gradient = self.b
            self.hessian = np.zeros((self.dim, self.dim))
        elif self.field_type == "quadratic":
            self.hessian = self.a + np.transpose(self.a)
            self.field = np.dot(data, self.a.dot(data)) + self.b.dot(data) + self.c
            self.gradient = self.hessian.dot(data) + self.b

    # Returns scalar_field
    def sendField(self):
        scalar_field = Field()
        scalar_field.input_dim = self.dim
        scalar_field.output_dim = 1
        scalar_field.var = []
        scalar_field.field = [self.field]
        scalar_field.gradient = []
        scalar_field.Qgradient = [self.Qgradient]
        scalar_field.hessian = []
        for i in range(0, self.dim):
            scalar_field.var.append(self.var[i])
            scalar_field.gradient.append(self.gradient[i])
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                scalar_field.hessian.append(self.hessian[i, j])

        return scalar_field

    # Receives scalar_field and converts to numpy
    def setField(self, scalar_field):
        self.dim = scalar_field.input_dim
        self.var = np.array(scalar_field.var)
        self.field = scalar_field.field[0]
        self.gradient = np.array(scalar_field.gradient)
        self.Qgradient = scalar_field.Qgradient[0]
        self.hessian = np.array(scalar_field.hessian).reshape((self.dim, self.dim))


class VectorField:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.var = np.zeros(self.input_dim)
        self.field = np.zeros(self.output_dim)
        self.jacobian = np.zeros((self.output_dim, self.input_dim))
        self.list_jacobians = np.zeros((self.input_dim, self.output_dim, self.input_dim))

    # Returns scalar_field
    def sendField(self):
        vector_field = Field()
        vector_field.input_dim = self.input_dim
        vector_field.output_dim = self.output_dim
        vector_field.var = []
        vector_field.field = []
        vector_field.jacobian = []
        vector_field.list_jacobians = []
        for i in range(0, self.input_dim):
            vector_field.var.append(self.var[i])
        for i in range(0, self.output_dim):
            vector_field.field.append(self.field[i])
        for i in range(0, self.output_dim):
            for j in range(0, self.input_dim):
                vector_field.jacobian.append(self.jacobian[i, j])
        for k in range(0, self.input_dim):
            for i in range(0, self.output_dim):
                for j in range(0, self.input_dim):
                    vector_field.list_jacobians.append(self.list_jacobians[k, i, j])

        return vector_field

    # Receives scalar_field and converts to numpy
    def getField(self, vector_field):
        self.input_dim = vector_field.input_dim
        self.output_dim = vector_field.output_dim
        self.var = np.array(vector_field.var)
        self.field = np.array(vector_field.field)
        self.jacobian = np.array(vector_field.jacobian).reshape((self.output_dim, self.input_dim))
        self.list_jacobians = np.array(vector_field.list_jacobians).reshape(
            (self.input_dim, self.output_dim, self.input_dim))


class AffineModel:

    def __init__(self, model_type, *args):
        self.model = model_type

        if self.model == "unicycle":
            self.model_dim = 3
            self.ctrl_dim = 2
        elif self.model == "linear":
            self.A = args[0]
            self.B = args[1]
        else:
            self.model_dim = 0
            self.ctrl_dim = 0

        self.state = np.zeros(self.model_dim)
        self.f_x = np.zeros(self.model_dim)
        self.jacobian_f_x = np.zeros((self.model_dim, self.model_dim))
        self.g_x = np.zeros((self.model_dim, self.ctrl_dim))
        self.list_jacobians = np.zeros((self.model_dim, self.model_dim, self.model_dim))
        self.vector_field = np.zeros(self.model_dim)

    def setState(self, state):
        self.state = state

    def computeModel(self, state):
        self.setState(state)

        if self.model == "unicycle":
            theta = self.state[2]
            self.f_x = np.zeros(self.model_dim)
            self.jacobian_f_x = np.zeros((self.model_dim, self.model_dim))
            self.g_x = np.array([[np.cos(theta), 0],
                                 [np.sin(theta), 0],
                                 [0, 1]])
            Jg1 = np.array([[0, 0, -2 * np.sin(theta) * np.cos(theta)],
                            [0, 0, np.power(np.cos(theta), 2) - np.power(np.sin(theta), 2)],
                            [0, 0, 0]])
            Jg2 = np.array([[0, 0, np.power(np.cos(theta), 2) - np.power(np.sin(theta), 2)],
                            [0, 0, 2 * np.sin(theta) * np.cos(theta)],
                            [0, 0, 0]])
            Jg3 = np.zeros((self.model_dim, self.model_dim))
            self.list_jacobians[0, :, :] = Jg1
            self.list_jacobians[1, :, :] = Jg2
            self.list_jacobians[2, :, :] = Jg3

        elif self.model == "linear":
            self.f_x = self.A * self.state
            self.g_x = self.B

    def computeDynamics(self, state, ctrl):
        self.computeModel(state)
        self.vector_field = self.f_x + self.g_x.dot(ctrl)

    # Returns model
    def sendModel(self):
        model_msg = Model()
        model_msg.model_dim = self.model_dim
        model_msg.ctrl_dim = self.ctrl_dim
        model_msg.state = []
        model_msg.f_x = []
        model_msg.J_x = []
        model_msg.g_x = []
        model_msg.jacobians = []
        for i in range(0, self.model_dim):
            model_msg.state.append(self.state[i])
            model_msg.f_x.append(self.f_x[i])
        for i in range(0, self.model_dim):
            for j in range(0, self.model_dim):
                model_msg.J_x.append(self.jacobian_f_x[i, j])
        for i in range(0, self.model_dim):
            for j in range(0, self.ctrl_dim):
                model_msg.g_x.append(self.g_x[i, j])
        for k in range(0, self.model_dim):
            for i in range(0, self.model_dim):
                for j in range(0, self.model_dim):
                    model_msg.jacobians.append(self.list_jacobians[k, i, j])

        return model_msg

    # Receives model and converts to numpy
    def setModel(self, model_msg):
        self.model_dim = model_msg.model_dim
        self.ctrl_dim = model_msg.ctrl_dim
        self.state = np.array(model_msg.state)
        self.f_x = np.array(model_msg.f_x)
        self.jacobian_f_x = np.array(model_msg.J_x).reshape((self.model_dim, self.model_dim))
        self.g_x = np.array(model_msg.g_x).reshape((self.model_dim, self.ctrl_dim))
        self.list_jacobians = np.array(model_msg.jacobians).reshape((self.model_dim, self.model_dim, self.model_dim))
