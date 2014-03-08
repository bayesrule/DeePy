
import numpy as np
import Utilities as utils

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""

	def __init__(self, architecture):
		super(NeuralNetwork, self).__init__()
		self.size = architecture
		self.n = len(architecture)
		self.W = [None] * (self.n - 1)
		self.vW = [None] * (self.n - 1)
		self.p = [None] * self.n


		self.set_options()

		for i in xrange(1,self.n):

			self.W[i - 1] = (np.random.rand(self.size[i], self.size[i - 1] + 1) - 0.5) * 2.0 * 8.0 * np.sqrt(
				6.0 / (self.size[i] + self.size[i - 1])
				)
			self.vW[i - 1] = np.zeros(self.W[i - 1].shape)
			self.p[i] = np.zeros((1, self.size[i]))

	def apply_gradients(self):
		
		for i in xrange(self.n - 1):
			if self.weight_penalty > 0:
				# TO-DO
				dW = self.dW[i] + self.weight_penalty * np.zeros((self.W[i].shape[0], 1)).append(self.W[i][:, 1:])
			else:
				dW = self.dW[i]

			dW = self.learning_rate * dW

			if self.momentum > 0:
				self.vW[i] = self.momentum * self.vW[i] + dW
				dW = self.vW[i]

			self.W[i] = self.W[i] - dW

	def feedforward(self, x, y):
		n = self.n
		m = x.shape[0]

		x = np.append(np.ones((m,1)), x)
		
		# TO-DO
		self.a = [None] * (n - 1)
		self.a[0] = x

		for i in xrange(1, n-1):

			if self.activation_function == "sigmoid":

				self.a[i] = utils.sigmoid(np.dot(self.a[i - 1], self.W[i - 1].T))
			elif self.activation_function == "tanh_opt":
				self.a[i] = utils. tanh_opt(np.dot(self.a[i - 1], self.W[i - 1].T))

			if self.dropout_fraction > 0:
				if self.testing:
					self.a[i] = self.a[i] * (1 - self.dropout_fraction)
				else:
					self.dropout_mask[i] = np.random.rand(self.a[i].shape) > self.dropout_fraction
					self.a[i] = self.a[i] * self.dropout_mask[i]

			if self.sparsity_penalty > 0:
				# What?
				self.p[i] = 0.99 * self.p[i] + 0.01 * np.mean(self.a[i], axis = 0)

			self.a[i] = np.append(np.ones((m, 1)), self.a[i])

			if self.output == "sigmoid":
				self.a[n - 1] = utils.sigmoid(np.dot(self.a[n - 1], self.W[i - 1].T))
			elif self.output == "linear":
				self.a[n - 1] = np.dot(self.a[n - 1], self.W[i - 1].T)
			elif self.output == "softmax":
				# TO-DO
				pass

			self.e = y - self.a[n]

			if self.output in ("sigmoid", "linear"):
				self.L = 1.0 / 2 * np.sum(self.e ** 2) / m
			elif self.output == "softmax":
				self.L = -np.sum(y * np.log(self.a[n - 1])) / m


	def train(self, train_x, train_y, opts, validate_x, validate_y):
		loss = {"train" : {"e" : [], "e_frac" : []}, "validate" : {"e" : [], "e_frac" : []}}

		m = train_x.shape[0]

		batchsize = opts["batchsize"]
		n_epochs = opts["n_epochs"]
		
		n_batches = m / batchsize

		L = np.zeros((n_epochs * n_batches,1))
		n = 0

		for i in xrange(n_batches):

			index = np.random.permutation(m)

			for j in xrange(n_batches):

				batch_x = train_x[index[(j - 1) * batchsize + 1 : j * batchsize],:]

				if self.zero_masked_fraction != 0:
					batch_x = batch_x * (np.random.rand(batch_x.shape) > self.zero_masked_fraction)

				batch_y = train_y[index[(j - 1) * batchsize + 1 : j * batchsize],:]

				self.feedforward(batch_x, batch_y)
				self.backpropogation()
				self.apply_gradients()

				L[n] = self.L
				n += 1

			self.learning_rate *= self.scaling_rate

	def test(self):
		labels = self.predict(x)
	    expected = y.argmax(axis = 1)
    	error = np.sum(labels != expected) / float(x.shape[0])
    	
    	return error

    def predict(self, x):
    	self.testing = True
	    self.feedforward(x, np.zeros((x.shape[0], nn.size[-1])))
    	self.testing = False
    	index = max(self.a.keys())
    	labels = self.a[index].argmax(axis = 1)

    	return labels

	def set_options(self):

		self.activation_function = "tanh_opt"
		self.learning_rate = 2.0
		self.momentum = 0.5
		self.scaling_rate = 1.0
		self.weight_penalty = 0.0
		self.sparsity_penalty = 0.0
		self.sparsity_target = 0.05
		self.zero_masked_fraction = 0.0
		self.dropout_fraction = 0.0
		self.testing = False
		self.output = "sigmoid"




nn = NeuralNetwork((4,65))
