import numpy
import scipy.special
class neuralNetwork :

	def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):

		self.ins = input_nodes
		self.hns = hidden_nodes
		self.ons = output_nodes

		self.lr = learning_rate
		
		self.wih = numpy.random.normal(0.0,pow(self.hns,-0.5),(self.hns,self.ins))
		self.who = numpy.random.normal(0.0,pow(self.ons,-0.5),(self.ons,self.hns))

		self.activation_function = lambda x: scipy.special.expit(x)

		pass


	def train(self, inputs_list, targets_list):

		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		pass
		
	def query(self,inlst):

		inputs = numpy.array(inlst, ndmin=2).T
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
		
