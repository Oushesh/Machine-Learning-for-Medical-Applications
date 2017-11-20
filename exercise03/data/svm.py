# pylint: skip-file
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linear_kernel(x, z, kernel_params=None):
	#TODO Compute the linear kernel
    return np.dot(x, z)

def compt_b(sv, svy, sva, kernel, kernel_params={'a' : 1, 'c' : 1, 'd' : 3, 'sigma' : 0.5}):
	b = 0
	#TODO Compute the bias b
	m = sv.shape[0]
	for j in range(m):
		sum = 0
		for i in range(m):
			sum += sva[i] * svy[i] * kernel(sv[i], sv[j], kernel_params)
		b+= svy[j] - sum
	b/= float(m)
	return b

def eval(x, sv, svy, sva, b, kernel, kernel_params={'a' : 1, 'c' : 1, 'd' : 3, 'sigma' : 0.5}):
	h = 0
	m = sv.shape[0]
	sum = 0
	for i in range(m):
		sum += sva[i] * svy[i] * kernel(sv[i], x, kernel_params)
	h = sum + b
	return np.sign(h)

class SVM(object):

	def __init__(self, kernel=linear_kernel, C=None):
		self.kernel = kernel
		self.C = C
		if self.C is not None: self.C = float(self.C)

	#X a feature vector with m x n where m are the number of samples and n the number of features
	def fit(self, X, y, gram_matrix, support_vectors, kernel_params=None):
		n_samples, n_features = X.shape

		# Gram matrix
		K = gram_matrix(X, self.kernel)		

		P = cvxopt.matrix(np.outer(y,y) * K)
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y, (1,n_samples))
		b = cvxopt.matrix(0.0)

		if self.C is None:
			G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
			h = cvxopt.matrix(np.zeros(n_samples))
		else:
			tmp1 = np.diag(np.ones(n_samples) * -1)
			tmp2 = np.identity(n_samples)
			G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
			tmp1 = np.zeros(n_samples)
			tmp2 = np.ones(n_samples) * self.C
			h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
		solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		# Lagrange multipliers
		a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers        
		self.a, self.sv, self.sv_y = support_vectors(a, X, y, self.C)
		print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
		self.b = compt_b(self.sv, self.sv_y, self.a, self.kernel)         
		self.w = None
		
	def project(self, X):
		if self.w is not None:
			return np.dot(X, self.w) + self.b
		else:
			y_predict = np.zeros(len(X))
			for i in range(len(X)):
				s = 0
				for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
					s += a * sv_y * self.kernel(X[i], sv)
				y_predict[i] = s
			return y_predict + self.b
	

	def predict(self, X):
		m = X.shape[0]
		H = np.empty(shape=(m))
		for i in range(m):
			H[i] = eval(X[i], self.sv, self.sv_y, self.a, self.b, self.kernel)
		return H