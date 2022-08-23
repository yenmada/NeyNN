"""
The activiationfunction.py contains the ActivationFunction class, defining
attributes and methods for supported activation functions.
"""

# Import libraries
import numpy as np
from scipy.special import expit

class ActivationFunction:

	def __init__(self, act_func):
		"""Initialize the ActivationFunction class.

		During initialization, the activation function and function
		derivative attributes are set based on the activation function
		provided.

		Parameters
		----------
		act_func : str
			The type of activation function for the layer.
		"""

		# If the activation function type is not in the list of supported
		# functions, raise an exception.
		if act_func.lower() not in ['sigmoid', 'relu']:
			raise Exception('Provide a supported activation function for the layer!')

		# Set the function and derivative attributes based on the defined
		# activation function.
		if act_func.lower() == 'sigmoid':
			self.func = self.sigmoid
			self.deriv_func = self.sigmoid_deriv
		elif act_func.lower() == 'relu':
			self.func = self.relu
			self.deriv_func = self.relu_deriv

	def sigmoid(self, z):
		"""Sigmoid activation function.

		Parameters
		----------
		z : float
			Weighted inputs of the neurons in a layer.

		Returns
		-------
		Sigmoid function evaluated at weighted inputs z.
		"""

		return expit(z)

	def sigmoid_deriv(self, z):
		"""Sigmoid activation function derivative. 

		Parameters
		----------
		z : float
			Weighted inputs of the neurons in a layer.

		Returns 
		-------
		Sigmoid function derivative evaluated at weighted inputs z.
		"""

		return expit(z)*(1-expit(z))

	def relu(self, z):
		"""Rectified linear unit activation function.

		Parameters
		----------
		z : float
			Weighted inputs of the neurons in a layer.

		Returns
		-------
		Relu function evaluated at weighted inputs z.
		"""

		return np.maximum(0, z)

	def relu_deriv(self, z):
		"""Rectified linear unit activation function derivative. 

		Parameters
		----------
		z : float
			Weighted inputs of the neurons in a layer.

		Returns
		-------
		Rectified linear unit derivative evaluated at weighted inputs z.
		"""

		z[z > 0] = 1
		z[z <= 0] = 0
		return z