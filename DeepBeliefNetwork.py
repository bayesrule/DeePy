import NeuralNetwork as nn
import RestrictedBoltzmannMachine as rbm

class DeepBeliefNetwork(object):
	"""docstring for DeepBeliefNetwork"""
	def __init__(self):
		super(DeepBeliefNetwork, self).__init__()

	def train(self, x, opts):
		n = len(self.rbm)

		self.rbm[0] = rbm.train(self.rbm[0], x, opts)
		for i in range(1,n):
			x = rbm.up(self.rbm[i - 1], x)
			self.rbm[i] = rbm.train(self.rbm[i], x, opts)
			

	def unfold_to_neural_network(self, output_architecture = None):
		if output_architecture is not None:
			architecture = self.sizes.append(output_architecture)
		else:
			architecture = self.sizes

		neural_network = nn.NeuralNetwork(architecture)
		for i in range(len(self.rbm)):
			neural_network.W[i] = self.rbm[i].c.append(self.rbm[i].W)

		
		