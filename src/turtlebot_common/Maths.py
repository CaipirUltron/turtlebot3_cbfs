#!/usr/bin/env python3

import numpy as np
from turtlebot3_cbfs.msg import ScalarField


class Field:

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

        self.var = np.zeros(dim)
        self.field = 0.0
        self.gradient = np.zeros(self.dim)
        self.hessian = np.zeros([self.dim, self.dim])
        self.compute_local()

    def __add__(self, other):
        if self.dim != other.dim:
            raise ValueError('Dimension mismatch. Cannot sum fields with a different number of dimensions.')

        if not np.array_equal(self.var, other.var):
            raise ValueError('Fields computed at different points.')

        new_field = Field(self.dim, "")
        new_field.var = self.var
        new_field.field = self.field + other.field
        new_field.gradient = self.gradient + other.gradient
        new_field.hessian = self.hessian + other.hessian

        return new_field

    def __sub__(self, other):
        if self.dim != other.dim:
            raise ValueError('Dimension mismatch. Cannot sum fields with a different number of dimensions.')

        if not np.array_equal(self.var, other.var):
            raise ValueError('Fields computed at different points.')

        new_field = Field(self.dim, "")
        new_field.var = self.var
        new_field.field = self.field - other.field
        new_field.gradient = self.gradient - other.gradient
        new_field.hessian = self.hessian - other.hessian

        return new_field

    def __mul__(self, other):
        if self.dim != other.dim:
            raise ValueError('Dimension mismatch. Cannot sum fields with a different number of dimensions.')

        if not np.array_equal(self.var, other.var):
            raise ValueError('Fields computed at different points.')

        new_field = Field(self.dim, "")
        new_field.setVar(self.var)
        new_field.field = self.field * other.field
        new_field.gradient = self.gradient * other.field + other.gradient * self.field
        grad1grad2 = np.outer(self.gradient, other.gradient)
        new_field.hessian = self.hessian * other.field + grad1grad2 + np.transpose(
            grad1grad2) + self.field * other.hessian

        return new_field

    def setVar(self, data):
        self.var = data
        self.compute_local()

    def compute_local(self):
        data = self.var
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
    def getField(self):
        scalar_field = ScalarField()
        scalar_field.dim = self.dim
        scalar_field.field = self.field
        scalar_field.gradient = []
        scalar_field.hessian = []
        for i in range(0, self.dim):
            scalar_field.gradient.append(self.gradient[i])
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                scalar_field.hessian.append(self.hessian[i, j])

        return scalar_field

    # Receives scalar_field and converts to numpy
    def setField(self, scalar_field):
        self.dim = scalar_field.dim
        self.field = scalar_field.field
        self.gradient = np.array(scalar_field.gradient)
        self.hessian = np.array(scalar_field.hessian).reshape((self.dim, self.dim))


class AffineModel:

    def __init__(self, model_type, *args):
        self.model = model_type

        if self.model == "unicycle":
            self.state_dim = 3
            self.ctrl_dim = 2
        elif self.model == "linear":
            self.A = args[0]
            self.B = args[1]
        else:
            self.state_dim = 0
            self.ctrl_dim = 0

        self.state = np.zeros(self.state_dim)

        self.f_x = np.zeros(self.state_dim)
        self.g_x = np.zeros((self.state_dim, self.ctrl_dim))
        self.vector_field = np.zeros(self.state_dim)

    def setState(self, state):
        self.state = state

    def computeModel(self, state):
        self.setState(state)

        if self.model == "unicycle":
            theta = self.state[2]
            self.f_x = np.zeros(self.state_dim)
            self.g_x = np.array([[np.cos(theta), 0],
                                 [np.sin(theta), 0],
                                 [0, 1]])
        elif self.model == "linear":
            self.f_x = self.A * self.state
            self.g_x = self.B

    def computeDynamics(self, state, ctrl):
        self.computeModel(state)
        self.vector_field = self.f_x + self.g_x.dot(ctrl)