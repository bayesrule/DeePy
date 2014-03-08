
import numpy as np


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


		# print self.n
		# print self.size

		for i in range(1,self.n):

			# print i
			# print self.size[i] + self.size[i - 1]
			# print (np.random.rand(self.size[i], self.size[i - 1] + 1) - 0.5) * 2.0 * 8.0 * np.sqrt(6.0 / (self.size[i] + self.size[i - 1]))

			self.W[i - 1] = (np.random.rand(self.size[i], self.size[i - 1] + 1) - 0.5) * 2.0 * 8.0 * np.sqrt(
				6.0 / (self.size[i] + self.size[i - 1])
				)
			self.vW[i - 1] = np.zeros(self.W[i - 1].shape)
			self.p[i] = np.zeros((1, self.size[i]))

	def apply_gradients(self):
		


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
