import Utilities as utils
import numpy as np

class RestrictedBoltzmannMachine(object):
	"""docstring for RestrictedBoltzmannMachine"""
	def __init__(self, architecture, opts):
		super(RestrictedBoltzmannMachine, self).__init__()
		self.alpha = opts.alpha
		self.momentum = opts.momentum

		self.W = np.zeros((architecture[1], architecture[0]))
		self.vW = np.zeros((architecture[1], architecture[0]))

		self.b = np.zeros((architecture[0], 1))
		self.vb = np.zeros((architecture[0], 1))

		self.c = np.zeros((architecture[1], 1))
		self.vc = np.zeros((architecture[1], 1))

	def train(self, x, opts):
		m = x.shape[0]
		n_batches = m / opts["batchsize"]

		for i in xrange(opts["n_epochs"]):
			index = np.random.permutation(m)
			error = 0.0
			for j in xrange(n_batches):
				batch = x[index[(j - 1) * opts["batchsize"] : j * opts[batchsize] - 1, :]

				v1 = batch
				h1 = utils.sigmoid_random(np.tile(self.c.T, (opts["batchsize"], 1)) + np.dot(v1, self.W.T))
				v2 = utils.sigmoid_random(np.tile(self.b.T, (opts["batchsize"], 1)) + np.dot(h1, self.W))
				h2 = sigmoid(np.tile(self.c.T, (opts["batchsize"], 1)) + np.dot(v2, self.W.T))

				c1 = np.dot(h1.T, v1)
				c2 = np.dot(h2.T, v2)

				self.vW = self.momentum * self.vW + self.alpha * (c1 - c2) / opts["batchsize"]
				self.vb = self.momentum * self.vb + self.alpha * np.sum(v1 - v2, axis = 0).T / opts["batchsize"]
				self.vc = self.momentum * self.vc + self.alpha * np.sum(h1 - h2, axis = 0).T / opts["batchsize"]

				self.W += self.vW
				self.b += self.vb
				self.c += self.vc

				error += np.sum((v1 - v2) ** 2) / opts["batchsize"]

			print "Epoch %d complete. Average reconstruction error: %.2f\n" % (i, error / n_batches)
		
	def up(self, x):
		x = utils.sigmoid(np.tile(self.c.T, (x.shape[0], 1)) + np.dot(x, self.W.T))
		return x

	def down(self, x):
		x = utils.sigmoid(np.tile(self.b.T, (x.shape[0], 1)) + np.dot(x, self.W))
		return x

