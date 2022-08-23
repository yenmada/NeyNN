"""
The layer.py module contains the Layer class, defining all attributes 
for a neural network layer.
"""

# Import libraries
import numpy as np
from activationfunction import ActivationFunction
from classifier import Classifier

class Layer:

	def __init__(self, layer_type, neurons, act_func, classifier):
		"""Initialize the Layer class. 

		During initialization, neurons, layer_type, and act_func attributes
		of the layer are set. Additionally, deltas and activations lists, which
		track the layer's delta and initialization during a batch, are initialized
		to empty lists.

		Parameters 
		----------
		layer_type : str
			The type of the layer - either "input", "hidden", or "output". The
			layers must be added in sequential order (i.e. input first, then
			hidden layers, then output layer).
		neurons : int
			The number of neurons in the layer.
		act_func : str
			The type of activation function for the layer.
		classifier : str
			The type of classification (binary or non-binary) for the OUTPUT LAYER
			ONLY
		"""

		# Initialize the deltas and activations attributes to empty lists.
		self.deltas = []
		self.activations = []

		# Set the neurons, layer_type, act_func, and classifier attributes based on the 
		# input arguments. 
		self.neurons = neurons
		self.layer_type = layer_type

		if act_func != None:
			self.act_func = ActivationFunction(act_func)

		if classifier != None:
			self.classifier = Classifier(classifier)