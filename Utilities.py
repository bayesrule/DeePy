
import numpy as np

class Utilities(object):
	"""docstring for Utilities"""
	def __init__(self):
		super(Utilities, self).__init__()
	
	def sigmoid(self, P):
		return 1.0 / (1 + np.exp(-P))

	def sigmoid_random(self, P):
		return self.sigmoid(P) > np.random.rand(P.shape[0],P.shape[1])

		