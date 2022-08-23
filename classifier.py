"""
The classifier.py module contains the Classifier class, defining attributes
and methods for the supported classifiers.
"""

# Import libraries
import numpy as np

class Classifier: 

	def __init__(self, classifier):
		"""Initialize the Classifier clas.

		During initialization, the classification function method
		is set based on user input.

		Parameters
		----------
		classifier : str
			The type of classification (binary or non-binary)
		"""

		# Check that the classifier type is supported
		if classifier.lower() not in ['binary', 'non-binary']:
			raise Exception('Provide a supported classifier for the output layer!')

		# Set the classifier type attribute and classifier function method 
		# based on user input.
		if classifier.lower() == 'binary':
			self.type = 'binary'
			self.func = self.binary
		elif classifier.lower() == 'non-binary':
			self.type = 'non-binary'
			self.func = self.non_binary

	def binary(self, a):
		"""Binary classification function.

		Take the output of the neural network (for a binary classifier the
		output layer has only one neuron). If the activation is less than 
		0.5, it is classified as a 0. If the activation is greater than or
		equal to 0.5, it is classified as a 1.

		Parameters
		----------
		y : ndarray
			One-dimensional array representing the answer to the network 
			input.

		Returns
		-------
		classification : int
			Classification of the current training example.
		"""

		return (lambda a: 1 if a >= 0.5 else 0)(a)

	def non_binary(self, a):
		"""Non binary classification function.

		Take the output layer of the neural network and find the
		index of the largest value in the layer's activiations.
		The largest value is the classification.

		Parameters
		----------
		a : ndarray
			One-dimensional array of the output of the network based on 
			the input with answer y.

		Returns
		-------
		classification : int
			Classification of the current training example.
		"""

		return np.argmax(a)