import NeuralNetwork as nn

class StackedAutoencoder(object):
	"""docstring for StackedAutoencoder"""
	
	def __init__(self, size):
		super(StackedAutoencoder, self).__init__()
		
		for i in range(1,len(size)):
			layers = (size[i - 1],size[i],size[i - 1])
			sae.ae[i - 1] = nn.setup(layers)


	def train(self, x, opts):

		for i in range(len(self.ae)):
			self.ae[i] = nn.train(self.ae[i], x, x, opts)
			t = nn.feedforward(self.ae[i], x, x)
			x = t.a[1]
			x = x[:,1:]

