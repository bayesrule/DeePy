

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self, architecture):
		super(NeuralNetwork, self).__init__()
		self.size = architecture
		self.n = len(architecture)

		self._set_options()



	def _set_options(self):

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



nn = NeuralNetwork((1,))
