import unittest
import numpy as np
import linear_regression as lr
from numpy.testing import assert_allclose
import pickle as pkl

seed = 10417617

with open("tests.pkl","rb") as f: 
    tests = pkl.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.TestLinearReg
# to run all tests: python -m unittest tests


class TestSecondOrderBasis(unittest.TestCase):
	def test(self):
		# the order of the basis function is not important 
		X, result = tests[0]
		test1 = lr.second_order_basis(X)
		for i, r in enumerate(result):
			assert_allclose(sorted(test1[i]), sorted(r), atol=TOLERANCE)

		X, result = tests[1]
		test2 = lr.second_order_basis(X)
		for i, r in enumerate(result):
			assert_allclose(sorted(test2[i]), sorted(r), atol=TOLERANCE)



class TestLinearReg(unittest.TestCase):
	def test(self):
		indim, outdim, X, T, Xn, Yn = tests[2]
		lr_model = lr.LinearReg(indim, outdim)
		lr_model.fit(X, T)
		test3 = lr_model.predict(Xn)
		assert_allclose(test3, Yn, atol=TOLERANCE)


		indim, outdim, X, T, Xn, Yn = tests[3]
		lr_model = lr.LinearReg(indim, outdim)
		lr_model.fit(X, T)
		test3 = lr_model.predict(Xn)
		assert_allclose(test3, Yn, atol=TOLERANCE)

