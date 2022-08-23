"""
The regularizationmethod.py module contains the RegularizationMethod class, 
defining attributes and methods for the supported regularization methods.
"""

# Import libraries
import numpy as np

class RegularizationMethod:

	def __init__(self, reg_method, reg_param):
		"""Initialize the RegularizationMethod class.

		During initialization, the regularization parameter, weight function,
		and weight multiplier attributes are set based on user input.

		Parameters
		----------
		reg_method : str
			The regularization method for the network. 
		reg_param : float
			The regularization parameter for the network.
		"""

		# Set the regularization parameter and weight multiplier
		# function methods.
		self.param = reg_param
		self.weight_multiplier = self.multiplier

		# Set the weight function method based on the regularization method
		# defined by the user.
		if reg_method == 'none':
			self.weight_func = self.identity
		elif reg_method == 'L2':
			self.weight_func = self.identity

	def identity(self, w):
		"""Weight identify function.

		Simply return the weights that are input into the function. 
		This method is necessary in order to generalize the calculations
		of the Network.update_wb method.

		Parameters
		----------
		w : ndarray
			Array of weights in the neural network layer.

		Returns
		-------
		w : ndarray
			Array of weights in the neural network layer.
		"""

		return w

	def multiplier(self, n):
		"""Weight multiplier function. 

		Computes the weight multiplier for regularization, the ratio of the 
		regularization parameter and the number of training examples in the 
		training dataset. 

		Parameters
		----------
		n : int
			Number of training examples in the training dataset.

		Returns
		-------
		Ratio of regularization parameter to the number of training inputs.
		"""

		return self.param/n