"""
The costfunction.py contains the CostFunction class, defining attributes
and methods for the neural network cost function.
"""

# Import libraries
import numpy as np

class CostFunction:

	def __init__(self, cost_func):
		"""Initialize the CostFunction class.

		During initialization, the cost function and delta function attributes are set 
		based on the activation function provided.

		Parameters
		----------
		cost_func : str
			The type of cost function for the network. 			
		"""

		# Set the cost function and cost delta function attributes based on the defined
		# cost function.
		if cost_func.lower() == 'quad':
			self.func = self.quad
			self.delta_func = self.quad_delta
		elif cost_func.lower() == 'cross-entropy':
			self.func = self.cross_entropy
			self.delta_func = self.cross_entropy_delta

		# Set cost function name attribute
		self.cost_func = cost_func

	def quad(self, y, a):
		"""Quadratic cost function.

		Parameters
		----------
		y : ndarray
			One-dimensional array representing the answer to the network 
			input.
		a : ndarray
			One-dimensional array of the output of the network based on 
			the input with answer y.

		Returns
		-------
		Quadratic cost function evaluated for the network output and network
		input answer.		
		"""

		return (1/2)*np.sum(a-y)**2

	def quad_delta(self, y, a):
		"""Quadratic cost function function used for derivative computation 
		of weights and biases. 

		Parameters
		----------
		y : ndarray
			One-dimensional array representing the answer to the network 
			input.
		a : ndarray
			One-dimensional array of the output of the network based on 
			the input with answer y.

		Returns
		-------
		Quadratic cost function delta function evaluated for the network output
		and network input answer.
		"""

		return a - y

	def cross_entropy(self, y, a):
		"""Cross-entropy cost function.

		Parameters
		----------
		y : ndarray
			One-dimensional array representing the answer to the network 
			input.
		a : ndarray
			One-dimensional array of the output of the network based on 
			the input with answer y.

		Returns
		-------
		Cross-entropy cost function evaluated for the network output and 
		network input answer.		
		"""

		return -1*np.sum(y*np.log(a) + (1-y)*np.log(1-a))

	def cross_entropy_delta(self, y, a):
		"""Cross-entropy cost function function used for derivative computation 
		of weights and biases. 

		Parameters
		----------
		y : ndarray
			One-dimensional array representing the answer to the network 
			input.
		a : ndarray
			One-dimensional array of the output of the network based on 
			the input with answer y.

		Returns
		-------
		Cross-entropy cost function delta function evaluated for the network output
		and network input answer.
		"""

		# NOTE: THIS IST ONLY VALID FOR AN OUTPUT LAYER WITH A SIGMOID 
		# ACTIVATION FUNCTION

		return a - y