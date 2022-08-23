"""
The network.py module contains the Network class needed to build, train, 
and test a deep neural network.
"""

# Import libraries
import sys
import time
import numpy as np 
from layer import Layer
from costfunction import CostFunction
from regularizationmethod import RegularizationMethod
from optimizer import Optimizer

class Network:

	def __init__(self):
		"""Initialize the Network class.

		Initialization involves the creation of an attribute layers, 
		which is an empty list. This is a container for the Layer objects
		that create the network. The network timestep is also initialized to 0.
		"""
		
		# Initialize empty list of layers for the network
		self.layers = []

		# Timestep for optimization 
		self.t = 0

	def add_layer(self, layer_type=None, neurons=None, act_func=None, classifier=None):
		"""Method to add a layer to the neural network.
		
		This method creates a layer given the layer type, the number of neurons
		in the layer, and the activation function for the layer, and then 
		appends the layer to the Network layers list.

		Parameters 
		----------
		layer_type : str
			The type of the layer - either "input", "hidden", or "output". The
			layers must be added in sequential order (i.e. input first, then
			hidden layers, then output layer).
		neurons : int
			The number of neurons in the layer.
		act_func : str
			The type of activation function for the layer. Current supported
			options are: "sigmoid".
		classifier : str
			The type of classification (binary or non-binary) for the OUTPUT LAYER
			ONLY
		"""

		# Create a new layer based on the method input and append it to the object
		# list of layers
		new_layer = Layer(layer_type, neurons, act_func, classifier)
		self.layers.append(new_layer)

	def setup(self, optimizer='sgd', cost_func='quad', reg_method='none', reg_param=0, init_method='normal'):
		"""Method to set up the neural network.

		This method checks that the user defined a supported cost function,
		regularization method, and weight initialization method, and then 
		sets up the cost function, regularization method, and weights and 
		biases. 

		Parameters 
		----------
		cost_func : str
			The type of cost function for the network. Current supported 
			options are: "quad", "cross-entropy".
		reg_method : str
			The regularization method for the network. Current supported 
			options are: "none" and "L2". 
		reg_param : float
			The regularization parameter for the network.
		init_method : str
			The initializationn method for the weights in the network. 
			Current supported options are: "normal", "xavier".
		"""

		# Check that the cost function is supported
		if cost_func.lower() not in ['quad', 'cross-entropy']:
			raise Exception('Provide a supported cost function for the network!')

		# Check that the regularization method is supported
		if reg_method not in ['none', 'L2']:
			raise Exception('Provide a supported regularization method!')

		# Check that the weight and bias initialization method is supported 
		if init_method.lower() not in ['normal', 'xavier']:
			raise Exception('Provide a supported method for initialization of weights and biases!')

		# Instance the cost function
		self.cost_func = CostFunction(cost_func)

		# Instance the regularization method, checking that if there is no regularization then
		# the regularization parameter must be 0
		if reg_method == 'none' and reg_param != 0:
			raise Exception('If there is no regularization, the regularization parameter must be 0!')
		self.reg_method = RegularizationMethod(reg_method, reg_param)

		# Check that the layers are ordered correctly by type - input, hidden(s), output
		if self.layers[0].layer_type != 'input' or self.layers[-1].layer_type != 'output':
			raise Exception('The layers created for the network are not in the correct order!')
		if not (np.array([self.layers[i].layer_type for i in range(1, len(self.layers)-1)]) == 'hidden').all():
			raise Exception('The layers create for the network are not in the correct order!')

		# Initialize the weights randomly for each layer
		for i, layer in enumerate(self.layers):
			# If the layer is not an input layer, initialize the weights and biases
			if layer.layer_type != 'input':
				if init_method.lower() == 'normal':
					layer.weights = np.random.randn(self.layers[i].neurons, self.layers[i-1].neurons)
					layer.biases = np.random.randn(self.layers[i].neurons, 1)
				elif init_method.lower() == 'xavier':
					layer.weights = np.random.randn(self.layers[i].neurons, self.layers[i-1].neurons)*(1/np.sqrt(self.layers[i-1].neurons))
					layer.biases = np.random.randn(self.layers[i].neurons, 1)

		# Initialize the optimizer
		self.optimizer = Optimizer(optimizer, len(self.layers))

	def train(self, train_data=None, test_data=None, batch_size=10, epochs=25, learning_rate=1, beta1=0.9, beta2=0.999, eps=1e-8):
		"""Method to train the neural network and then measure its performance
		   on test data. 

		This method accepts the necessary training and testing data for the
		neural network as well as run parameters - batch size, number of epochs,
		and learning rate.

		Parameters
		----------
		train_data : tuple of ndarrays
			Tuple with the first element being an ndarray of training
			examples and the second element an ndarray of corresponding training
			answers. 
		test_data : tuple of ndarrays
			Tuple with the first element being an ndarray of test 
			examples and the second element an ndarray of test answers.
		batch_size : int
			Number of training examples for each batch.
		epochs : int
			Number of epochs for the training.
		learning_rate : float
			Learning rate for the optimization algorithm.
		beta1 : float
			Beta_1 parameter for the adam optimizer.
		beta2 : float 
			Beta_2 parameter for the adam optimizer.
		eps : float
			Epsilon parameter for the adam optimizer.
		"""

		# Set up optimizer with training parameters
		self.optimizer.setup(learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps)

		# Separate data into input/output data structures
		train_x = train_data[0]
		train_y = train_data[1].reshape(len(train_data[1]), 1)
		test_x = test_data[0]
		test_y = test_data[1].reshape(len(test_data[1]), 1)

		# Total training time counter
		total_time = 0

		# Perform training for given number of epochs
		for i in range(0, epochs):

			# Epoch start time
			start_time = time.time()

			# Shuffle the input data and answers for current epoch
			data = np.hstack((train_x, train_y))
			np.random.shuffle(data)
			epoch_x = data[:,0:-1]
			epoch_y = data[:,-1]

			# Split into batches
			batch_x = []; batch_y = []
			for j in range(0, int(np.ceil(len(data)/batch_size))):
				# Compute lower and upper indices of current batch
				lower = j*batch_size
				upper = (j+1)*batch_size
				# Slice out current batch data from shuffled epoch data 
				batch_x.append(epoch_x[lower:upper])
				batch_y.append(epoch_y[lower:upper])

			# Initialize the loading bar for this epoch
			msg = ''
			pounds = '#' * 50
			spaces = ' ' * 50

			# Loop over all of the batches to train
			for k, (bx, by) in enumerate(zip(batch_x, batch_y)):

				# Increment timestep 
				self.t += 1

				# Update loading status and message for loading bar
				sep = int(np.floor((k/len(batch_x))*50))
				load = pounds[0:sep] + spaces[sep:-1]
				msg = 'Epoch {0:0=2d} |{1}| Training {2:g}% complete'.format(i+1, load, round(k/(len(batch_x)-1), 4)*100)

				# Loop over all of the training examples and their answers
				for x, y in zip(bx, by):

					# Compute the output of the neural network
					self.feed_forward(x)
					# Compute the error of the output layer
					self.compute_error(y)
					# Back propagate the error
					self.back_propagate()

				# Update weights and biases for batch
				self.update_wb(len(bx), len(train_data[0]), learning_rate)

				# Reset deltas and activations for next batch
				self.reset_batch()

				print(msg, end='\r')

			# Initialize counter for number of correct classifications
			correct = 0

			# Update loading bar
			print(msg + ' | Running test data ...', end = '\r')

			# Epoch end time and elapsed time
			end_time = time.time()
			elapsed_time = (end_time - start_time)/60 # in minutes
			total_time += elapsed_time

			# Evaluate the classification accuracy of the test data
			acc = self.evaluate(test_x, test_y)

			# Update loading bar
			print(msg + ' | Accuracy: {0:.2f}% | Time: {1:.2f} min.'.format(acc, round(elapsed_time, 2)), end='')
			print("")

		# Print total training time
		print('Total training time: {0:g} min.'.format(round(total_time, 2)))

	def feed_forward(self, x):
		"""Method to feed the neural network forward based on a training
		   example. 

		This method accepts a training example x and feeds the neural network 
		forward. To feed forward, the method iterates over the non-input layers
		in order, computing each layers' activation. The output layer 
		activation is the neural network output. The output is also appended 
		to a list containing the outputs for the current batch, used for 
		stochastic gradient descent. 

		Parameters
		----------
		x : ndarray
		    A one-dimensional array training example. 
		"""

		# Set the input layer activation to the training example input
		self.layers[0].activation = x.reshape((len(x), 1))
		self.layers[0].activations.append(x)

		# Loop over the remaining hidden and output layers
		for i in range(1, len(self.layers)):

			# Compute the neuron weighted input z
			self.layers[i].z = np.matmul(self.layers[i].weights, self.layers[i-1].activation) + self.layers[i].biases
			# Compute the neuron activation by passing the weighted input into the activation function
			self.layers[i].activation = self.layers[i].act_func.func(self.layers[i].z)
			# Append the activation to a list of activations for the layer
			self.layers[i].activations.append(self.layers[i].activation)

	def compute_error(self, y):
		"""Method to compute the error of the neural network output layer.

		This method accepts the answer of a training example y and computes
		the error delta. The delta is appended to a list containing the deltas
		for the current batch, used for stochastic gradient descent.

		Parameters 
		----------
		y : int
			A one-dimensional array training example answer.
		"""
		
		# Create answer vector based on if the output layer is binary or 
		# non-binary classification
		y_vector = np.zeros((self.layers[-1].neurons, 1))
		if self.layers[-1].classifier.type == 'binary':
			y_vector[0] = int(y)
		elif self.layers[-1].classifier.type == 'non-binary':
			y_vector[int(y)] = 1

		# Compute the error of the output layer depending on the type of cost function
		if self.cost_func.cost_func == 'quad':
			self.layers[-1].delta = np.multiply(self.cost_func.delta_func(y_vector, self.layers[-1].activation),
				                                self.layers[-1].act_func.deriv_func(self.layers[-1].z))
		elif self.cost_func.cost_func == 'cross-entropy':
			self.layers[-1].delta = self.cost_func.delta_func(y_vector, self.layers[-1].activation)

		# Append the error to a list of errors for the output layer
		self.layers[-1].deltas.append(self.layers[-1].delta)

	def back_propagate(self):
		"""Method to back-propagate the error througout the non-input layers
		   of the network. 

		This method takes the output error and propagates it back through each
		layer in reverse order, sequentially. The deltas are also appended to a list
		containing the deltas for the current batch, used for stochastic
		gradient descent.
		"""

		# Loop over layers 2 through L-1 IN REVERSE ORDER
		for i in range(len(self.layers)-2, 0, -1):

			# Compute the error of the current layer
			self.layers[i].delta = np.multiply(np.matmul(np.transpose(self.layers[i+1].weights), self.layers[i+1].delta),
				                               self.layers[i].act_func.deriv_func(self.layers[i].z))
			# Append the error to a list of errors for the current layer
			self.layers[i].deltas.append(self.layers[i].delta)

	def update_wb(self, batch_size, train_size, learning_rate):
		"""Method to update the weights and biases based on the error 
		   propagated throughout the network.

		This method takes the errors computed throughout the network, 
		computes the change in weight and bias values for all weights and
		biases in the network, applies a regularization method if 
		applicable, and updates the weights and biases accordingly.

		Parameters
		----------
		batch_size : int
			Number of training examples for each batch.
		train_size : int
			Size of the training data.
		learning_rate : float
			Learning rate for the stochastic gradient descent algorithm.
		"""

		# Loop over over layers 2 through L IN REVERSE ORDER
		for i in range(len(self.layers)-1, 0, -1):

			# Compute cost function gradients with respect to the weights and biases
			# of the current layer
			weight_grad = np.zeros_like(self.layers[i].weights)
			bias_grad = np.zeros_like(self.layers[i].biases)

			# Loop over all training examples in current batch
			for j in range(0, batch_size):

				# Compute weight and bias change for training example
				weight_grad += np.matmul(self.layers[i].deltas[j], 
					                     np.transpose(self.layers[i-1].activations[j].reshape((len(self.layers[i-1].activations[j]), 1))))
				bias_grad += self.layers[i].deltas[j]

			# Compute the weight and bias changes
			weight_change = self.optimizer.calc_wbc(weight_grad/batch_size, 'weights', self.t, i)
			bias_change = self.optimizer.calc_wbc(bias_grad/batch_size, 'biases', self.t, i)

			# Compute the regularization
			regularization = learning_rate * self.reg_method.weight_multiplier(train_size) * self.reg_method.weight_func(self.layers[i].weights)

			# Update the weights and biases in the network
			self.layers[i].weights = self.layers[i].weights - regularization - weight_change
			self.layers[i].biases = self.layers[i].biases - bias_change

	def reset_batch(self):
		"""Method to reset the current batch after updating the weights 
		   and biases. 

		This method resets the batch by over-writing the lists of activations
		and deltas for the given batch that are used for stochastic gradient
		descent.
		"""

		# Loop over all the layers
		for i in range(0, len(self.layers)):
			# Reset layer errors and activations to empty lists
			self.layers[i].deltas = []
			self.layers[i].activations = []

	def evaluate(self, eval_x, eval_y):
		"""Method to evaluate the current the network based on a set of 
		   examples and answers.

		This method gets the ouptut of the neural network after feeding the
		input forward and then checks if the index of the maximum value of 
		the output corresponds to the answer.

		Parameters
		----------
		eval_x : ndarray
			One-dimensional array of examples for evaluation.
		eval_y : ndarray
			One-dimensional array of example answers for evaluation.

		Returns
		-------
		acc : float
			Accuracy of the network in classifying the data, in percent.
		"""

		# Initialize correct counter
		correct = 0

		# Loop over all of the test data to evaluate performance
		for x, y in zip(eval_x, eval_y):

			# Compute the output of the neural network
			self.feed_forward(x)
			# Check the output against the answer depending on the type of
			# classification using the class_func classification function
			output = self.layers[-1].activation
			classification = self.layers[-1].classifier.func(output)
			# If the classification is correct, add 1 to the correct counter
			if classification == y: 
				correct += 1

		# Percentage correct
		acc = round(correct/len(eval_x), 4) * 100

		return acc