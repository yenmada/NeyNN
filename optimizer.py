"""
The optimizer.py module contains the Optimizer class, containing
attributes and methods for supported optimization algorithms.
"""

# Import libraries
import numpy as np

class Optimizer:

	def __init__(self, optimizer, num_layers, learning_rate=1, beta1=0.9, beta2=0.999, eps=1e-8):
		"""Initialize the Optimizer class.

		During initialization, the optimizer weight change and 
		bias change attributes are set to Optimizer methods based on 
		the optimizer provided. 

		Parameters
		----------
		optimizer : str
			The type of optimizer for the neural network.
		num_layers : int
			The number of layers in the network.
		"""

		# If the optimizer type is not in the list of supported optimizers,
		# raise an exception.
		if optimizer.lower() not in ['sgd', 'adam']:
			raise Exception('Provide a supported optimizer for the network!')

		# Set optimizer name attribute
		self.optimizer = optimizer

		# Set the weight and bias change methods based on the defined
		# optimizer type
		if optimizer.lower() == 'sgd':
			self.calc_wbc = self.sgd_wbc
		elif optimizer.lower() == 'adam':
			self.calc_wbc = self.adam_wbc
			self.mtw = [0] * num_layers
			self.mtb = [0] * num_layers
			self.vtw = [0] * num_layers
			self.vtb = [0] * num_layers

	def setup(self, learning_rate=1, beta1=0.9, beta2=0.999, eps=1e-8):
		"""Setup the optimizer for training.

		Set the training-specific attributes for the optimizer.

		Parameters
		----------
		learning_rate : float
			The learning rate for the optimizer.
		beta1 : float 
			Beta_1 parameter for the adam optimizer.
		beta2 : float
			Beta_2 parameter for the adam optimizer.
		eps : float
			Epsilon parameter for the adam optimizer.
		"""

		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps

	def sgd_wbc(self, grad, update_type, *_):
		"""Stochastic gradient descent optimizer weight update rule.

		Compute and return the weight change for the stochastic gradient 
		descent optimizer.

		Parameters 
		----------
		grad : ndarray
			Gradients of the cost function with respect to the current 
			layer's weights or biases.
		update_type : str
			Denotes if the updates are for weights or for biases.
		"""

		return self.learning_rate*grad

	def adam_wbc(self, grad, update_type, t, index):
		"""Adam optimizer weight/bias update rule.

		Compute and return the weight change for the Adam optimizer.

		Parameters
		----------
		grad : ndarray
			Gradients of the cost function with respect to the current 
			layer's weights or biases.
		update_type : str
			Denotes if the updates are for weights or for biases.
		t : int
			Current timestep
		index : int
			Index of the current layer.
		"""

		# Check if the updates for weights or biases:
		if update_type.lower() == 'weights':

			# Compute the first and second moments of the gradients
			self.mtw[index] = self.beta1*self.mtw[index] + (1-self.beta1)*grad
			self.vtw[index] = self.beta2*self.vtw[index] + (1-self.beta2)*grad*grad

			# Compute the bias-corrected moments
			mt_hat = self.mtw[index]/(1-self.beta1**t)
			vt_hat = self.vtw[index]/(1-self.beta2**t)

		elif update_type.lower() == 'biases':

			# Compute the first and second moments of the gradients
			self.mtb[index] = self.beta1*self.mtb[index] + (1-self.beta1)*grad
			self.vtb[index] = self.beta2*self.vtb[index] + (1-self.beta2)*grad*grad

			# Compute the bias-corrected moments
			mt_hat = self.mtb[index]/(1-self.beta1**t)
			vt_hat = self.vtb[index]/(1-self.beta2**t)

		return self.learning_rate*(mt_hat/(np.sqrt(vt_hat) + self.eps))
