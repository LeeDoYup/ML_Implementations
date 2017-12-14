import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pdb

input_file = 'Stock Price Samsung.txt'

if input_file == 'Stock Price Samsung.txt':
	from config_stock_samsung import *
if input_file == 'two_moon.txt':
	from config_two_moon import *
if input_file == 'data_batch':
	from config_CIFAR10 import *


def read_CIFAR10(filename):
	import cPickle
	fo = open(filename, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	train_x = dict['data']
	labels = dict['labels']
	batch_length = len(train_x)
	train_y = np.zeros([batch_length,10])
	for i in range(batch_length):
		train_y[i,labels[i]] = 1
	return train_x, train_y
	#dict['data'], 'labels, label_names'
'''
def read_line(file_name):
	open_data = open(file_name,'r')
	read_data = open_data.readlines()
	open_data.close()
	return read_data

def split_data(result_data,split):
	length = len(result_data)
	split_result = []
	for i in range(length):
		split_result.append(result_data[i].split(split))
	return split_result
'''
#Above two function was only used in preprocessing

def preprocess_stock_data(file_name,lag_number,ahead_number):

	price_data = np.loadtxt(file_name)
	price_data = ( price_data - np.mean(price_data) ) / np.std(price_data)
	data_length = len(price_data)

	if lag_number + ahead_number > data_length:
		print "Lag & Prediction Ahead Number Error!!!"
		return

	available_sample_length = data_length - (lag_number+ahead_number) + 1
	x_data = np.zeros([available_sample_length , lag_number])
	y_data = np.zeros([available_sample_length , 1])

	for i in range(available_sample_length):
		x_data[i,:] = price_data[i:i+lag_number]
		y_data[i,0] = price_data[i+(lag_number+ahead_number-1)]
	
	#x_data, y_data = shuffle_data(x_data, y_data)
	return x_data, y_data

def preprocess_two_moon_data(filename):
	raw_data = np.loadtxt(filename)
	x = raw_data[:,:2]
	#x = (x-np.mean(x,axis=0))/np.std(x,axis=0)
	temp_y = np.reshape(raw_data[:,2], [len(raw_data),1])
	y = np.zeros([len(raw_data),2])
	for i in range(len(raw_data)):
		if temp_y[i] == 1 :
			y[i,1] = 1
		else:
			y[i,0] = 1
	return x, y

def one_hot_accuracy(y_out, y_true):
	total_number ,class_number = np.shape(y_out)
	accurate_number = 0.0
	for i in range(total_number):
		if y_out[i,:].argmax() == y_true[i,:].argmax():
			accurate_number +=1

	return accurate_number/total_number

	
def relu(input):
	shape = np.shape(input)
	temp = np.zeros(shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			if input[i,j]>0:
				temp[i,j]+=input[i,j]
	return temp

def gradient_relu(input):
	shape = np.shape(input)
	temp = np.zeros(shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			if input[i,j]>0:
				temp[i,j] = 1
	return temp

def identity(input):
	return input

def gradient_identity(input):
	return np.ones(np.shape(input))

def sigmoid(input):
	return 1/(1+np.exp(-input))

def gradient_sigmoid(input):
	return sigmoid(input)*(1-sigmoid(input))

def softmax(input):
	temp= np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1),[len(input),1])
	return temp

	#Input and output = [# of input] * [input dimension]

def gradient_activation(input,type_activation,bias=True):
	if type_activation == 'Identity':
		temp = gradient_identity(input)
	if type_activation == 'Sigmoid':
		temp = gradient_sigmoid(input)
	if type_activation == 'ReLU':
		temp = gradient_relu(input)

	if bias==True:
		temp[:,0] = 1
		return temp
	else:
		return temp

def xavier_init(n_inputs, n_outputs, shape, uniform=True):
	"""
	Args:
	n_inputs: The number of input nodes into each output.
	n_outputs: The number of output nodes for each input.
	uniform: If true use a uniform distribution, otherwise use a normal.
	Returns:
	An initializer.
	"""
	if uniform:
		init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
		return np.random.uniform(-init_range, init_range, shape)
	else:
	# 3 gives us approximately the same limits as above since this repicks
	# values greater than 2 standard deviations from the mean.
		stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
		return np.random.normal(scale=stddev,size = shape)

#Make Wieght Variables with random Gaussian Initialzer
def weight_variable(name, shape, initializer = 'normal',  xavier_n_in=10, xavier_n_out=10,xavier_uniform = True):
	#Random normal initializer
	if initializer == 'normal':
		return np.random.normal(scale=0.1,size=shape)
	#Xavier Initializer with uniform distribution
	if initializer =='Xavier' and xavier_uniform == True:
		n_inputs, n_outputs = shape
		return xavier_init(xavier_n_in,xavier_n_out,shape)
	#Xavier Initializer with uniform distribution
	if initializer == 'Xavier' and xavier_uniform == False:
		n_inputs, n_outputs = shape
		return xavier_init(xavier_n_in,xavier_n_out,shape,uniform = False)
	if initializer == 'Zeros':
		return np.zeros(shape)


#Make Feedforward hidden layer with Activations
#input = [Number of Data * Number of Input Features], wegiht = [Number of input features * output features]
#bias = [# of output features]
#return = [Number of Data * Number of Output Features]
def full_connected(inputs,weight,bias=True,activation='ReLU'):
	if activation == 'ReLU':
		temp = relu(np.dot(inputs,weight))
	if activation == 'Sigmoid':
		temp = sigmoid(np.dot(inputs,weight))
	if activation == 'Softmax':
		temp = softmax(np.dot(inputs,weight))
	if activation == "Identity":
		temp = identity(np.dot(inputs,weight))

	if bias == True:
		temp[:,0] = 1

	return temp

def loss_function(y_out,y_true,para,loss_type = 'cross_entropy', type_regularization = 'L2', regularization_para=0.01):
	#Chosse Loss function Type: Cross_entropy --> classification, MSE --> Regression
	if loss_type == 'cross_entropy':
		loss = -np.sum(y_true*np.log(y_out+1e-5))
	if loss_type == 'MSE':
		loss = np.mean(np.square(y_true-y_out))
	if regularization == False:
		return loss

	#Choose Regularizaton Type: L1, L2 
	vars = ('W1','W2','W_out')
	if type_regularization == 'L2':
		loss = loss + np.sum([ np.sum((np.square(para[v]))) for v in vars ]) * regularization_para
	if type_regularization == 'L1':
		loss = loss + np.sum([ (np.abs(para[v])) for v in vars ]) * regularization_para
	return loss

#Suffle data for Dataset (before dividing train, validation, test data set)
#Input type is numpy
def shuffle_data(x_train,y_train):
    temp_index = np.arange(len(x_train))

    #Random suffle index
    np.random.shuffle(temp_index)
    
    #Re-arrange x and y data with random shuffle index
    x_temp = np.zeros(np.shape(x_train))
    y_temp = np.zeros(np.shape(y_train))

    
    x_temp = x_train[temp_index]
    y_temp = y_train[temp_index]

    return x_temp, y_temp


def read_numpy(X_numpy_data, Y_numpy_data=None):
	if Y_numpy_data == None:
		x_data = np.load(X_numpy_data)
		#y_data processing needed
		return x_data
	else:
		x_data = np.load(X_numpy_data)
		y_data = np.load(Y_numpy_data)
		return x_data, y_data

#Default batch size is 1
#Tt is needed to control the total number of training data to divide with batch_size
def divide_train_valid_test(x_data,y_data,train_ratio,valid_ratio,test_ratio,batch_size = 1):
	#Exception Handling with sum of ratio = 1
	if train_ratio + test_ratio + valid_ratio != 1:
		print "Error: Control training, validation, and test ratio with sum 1"
		return

	batch_number = int(len(x_data)/batch_size)
	test_batch_number = int(batch_number*test_ratio)
	valid_batch_number = int(batch_number*valid_ratio)
	train_batch_number = batch_number - test_batch_number - valid_batch_number

	x_train = x_data[:train_batch_number*batch_size]
	y_train = y_data[:train_batch_number*batch_size]
	
	if valid_ratio != 0:
		x_valid = x_data[train_batch_number*batch_size:(train_batch_number+valid_batch_number)*batch_size]
		y_valid = y_data[train_batch_number*batch_size:(train_batch_number+valid_batch_number)*batch_size]

	x_test = x_data[(train_batch_number+valid_batch_number)*batch_size:(train_batch_number+valid_batch_number+test_batch_number)*batch_size]
	y_test = y_data[(train_batch_number+valid_batch_number)*batch_size:(train_batch_number+valid_batch_number+test_batch_number)*batch_size]

	if valid_ratio != 0:
		return x_train, y_train, x_valid, y_valid, x_test, y_test
	if valid_ratio == 0:
		return x_train, y_train, x_test, y_test


class Model(object):
	def __init__(self,lag,TT):
		print "Initialize New Model"
		self.lag = lag
		self.TT = TT

	def initialize_variable(self):
		#How the Inference Graph (Main Model) is decribed?
		#If you want to change the model, change this part and use proper loss fucntion.
		print "Create Inference Graph of Main Model"

		self.para = {}
		#self.para['W1'] = weight_variable('W1',[input_dim,layer1_nodes+1],'Xavier', input_dim, layer2_nodes)
		#self.para['W2'] = weight_variable('W2',[layer1_nodes+1,layer2_nodes+1],'Xavier', layer1_nodes+1, output_dim)
		self.para['W1'] = weight_variable('W1',[self.lag,layer1_nodes+1],W1_init)
		self.para['W2'] = weight_variable('W2',[layer1_nodes+1,layer2_nodes+1],W2_init)
		self.para['W_out'] = weight_variable('W_out',[layer2_nodes+1,output_dim],W_out_init)

		#Variable for saving gradient values during backprogation
		self.dpara = {}
		#### (+1) in each dimension means """BIAS term""!!
		self.train_record = []


	#Run Optimizer in order to minimize loss function and return its loss_value
	def batch_train(self,x_batch,y_batch,type_optimizer = 'GD'):
		batch_output = self.batch_out(x_batch) #FF
		self.backpropagation(y_batch,batch_output,x_batch)
		if type_optimizer == 'GD':
			for key in ('W1','W2','W_out'):
				self.para[key] += learning_rate * self.dpara[key]
		loss_value = loss_function(batch_output,y_batch,self.para,loss_type,type_regularization_,regularization_parameter)

		return loss_value

	#Backpropagation: Get Gradient of Loss function w.r.t. Each Variables
	def backpropagation(self,y_batch,y_output,x_batch):
		delta_last = y_batch - y_output
		delta_fc2 = np.dot(delta_last,self.para['W_out'].T) * self.gradient_fc2
		delta_fc1 = np.dot(delta_fc2,self.para['W2'].T) * self.gradient_fc1
		

		self.dpara['W1'] = weight_variable('dW1',[self.lag,layer1_nodes+1],'Zeros')
		self.dpara['W2'] = weight_variable('dW2',[layer1_nodes+1,layer2_nodes+1],'Zeros')
		self.dpara['W_out'] = weight_variable('dW_out',[layer2_nodes+1,output_dim],'Zeros')

		#In mini-batch learning, we get gradient as the average of gradient value for all samples in batch
		for i in range(batch_size):
			temp_out1 = np.dot(np.reshape(self.fc2[i,:],[len(self.fc2[i,:]),1]), 
				np.reshape(delta_last[i,:],[1,len(delta_last[i,:])]))
			#self.dpara['W_out'] += (temp_out - self.dpara['W_out'])/(i+1)
			self.dpara['W_out'] += temp_out1

			temp_out2 = np.dot(np.reshape(self.fc1[i,:],[len(self.fc1[i,:]),1]),
				np.reshape(delta_fc2[i,:],[1,len(delta_fc2[i,:])]))
			#self.dpara['W2'] += (temp_out - self.dpara['W2'])/(i+1)
			self.dpara['W2'] +=temp_out2

			temp_out3 = np.dot(np.reshape(x_batch[i,:],[len(x_batch[i,:]),1]),
				np.reshape(delta_fc1[i,:],[1,len(delta_fc1[i,:])]))
			#self.dpara['W1'] += (temp_out - self.dpara['W1'])/(i+1)
			self.dpara['W1'] +=temp_out3

		vars = ('W1','W2','W_out')
		for key in vars:
			self.dpara[key] = self.dpara[key]/batch_size
		#Weight Decaying Term of Loss Function		
		if regularization == True:
			for v in vars:
				self.dpara[v] += regularization_parameter*self.para[v]
	#return loss value of feed_dict input

	def batch_loss(self,y_out,y_true):
		loss_value = loss_function(y_out,y_true,self.para,loss_type,type_regularization_,regularization_parameter)
		return loss_value

	#return output result of feed_dict input
	def batch_out(self,x_in):
		self.fc1 = full_connected(x_in,self.para['W1'],bias=W1_bias,activation=fc1_activation)
		self.gradient_fc1 = gradient_activation(self.fc1,fc1_activation)
		self.fc2 = full_connected(self.fc1,self.para['W2'],bias=W2_bias,activation=fc2_activation)
		self.gradient_fc2 = gradient_activation(self.fc2,fc2_activation)
		self.output = full_connected(self.fc2,self.para['W_out'],bias=False,activation=fc_out_activation)
		outputs = self.output
		return outputs

	def get_data(self,X_numpy_data,Y_numpy_data=None):
		#temp_x, temp_y = read_numpy(X_numpy_data,Y_numpy_data)
		if X_numpy_data == 'Stock Price Samsung.txt':
			fc_out_activation = 'Identity'
			loss_type = 'MSE'
			temp_x, temp_y = preprocess_stock_data(X_numpy_data,self.lag,self.TT)

		if X_numpy_data == 'two_moon.txt':
			fc_out_activation = 'Softmax'
			loss_type = 'cross_entropy'
			temp_x, temp_y = preprocess_two_moon_data(X_numpy_data)

		if X_numpy_data == 'data_batch':
			temp_x, temp_y = read_CIFAR10('data_batch_1')
			for i in range(2,6):
				file_name = X_numpy_data + '_' + str(i)
				temp_x1, temp_y1 = read_CIFAR10(file_name)
				temp_x = np.concatenate((temp_x,temp_x1),axis=0)
				temp_y = np.concatenate((temp_y,temp_y1),axis=0)
			self.x_train, self.y_train = temp_x, temp_y
			self.x_test, self.y_test = read_CIFAR10('test_batch')
			return

		#devide train, valid, test data
		if valid_ratio !=0:
			self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = divide_train_valid_test(temp_x,temp_y,train_ratio,valid_ratio,test_ratio,batch_size)
		else:
			self.x_train, self.y_train, self.x_test, self.y_test = divide_train_valid_test(temp_x,temp_y,train_ratio,valid_ratio,test_ratio,batch_size)
		'''
		self.x_train = (self.x_train - np.reshape(np.mean(self.x_train,axis=1),[len(self.x_train),1]))/np.reshape(np.std(self.x_train,axis=1),[len(self.x_train),1])
		self.x_valid = (self.x_valid - np.reshape(np.mean(self.x_valid,axis=1),[len(self.x_valid),1]))/np.reshape(np.std(self.x_valid,axis=1),[len(self.x_valid),1])
		self.x_test = (self.x_test - np.reshape(np.mean(self.x_test,axis=1),[len(self.x_test),1]))/np.reshape(np.std(self.x_test,axis=1),[len(self.x_test),1])

		self.y_train = (self.y_train - np.mean(self.y_train))/np.std(self.y_train)
		self.y_valid = (self.y_valid - np.mean(self.y_valid))/np.std(self.y_valid)
		self.y_test = (self.y_test - np.mean(self.y_test))/np.std(self.y_test)
		'''
	def shuffle_train_data(self):
		temp_index = range(len(self.x_train))
		#Random suffle index
		np.random.shuffle(temp_index)

		#Re-arrange x and y data with random shuffle index
		x_temp = np.zeros(np.shape(self.x_train))
		y_temp = np.zeros(np.shape(self.y_train))

		for i in range(len(self.x_train)):
			x_temp[i,:] = self.x_train[temp_index[i],:]
			y_temp [i,:]= self.y_train[temp_index[i],:]
		self.x_train = x_temp
		self.y_train = y_temp

	def run_train_epoch(self):
		#parameter initializes: loss, cumulative loss
		loss = 0
		cumulative_loss = 0
		train_loss = 0
		average_loss = 0

		batch_number = int(len(self.x_train)/batch_size)

		self.shuffle_train_data()
		for i in range(batch_number):
			x_batch = self.x_train[i*batch_size:(i+1)*batch_size,:]
			y_batch = self.y_train[i*batch_size:(i+1)*batch_size,:]

			loss = self.batch_train(x_batch,y_batch)
			self.train_record.append(loss)
			cumulative_loss += loss
			#calculate average_loss for display
			average_loss = cumulative_loss / (i+1)

			sys.stdout.write("\r training loss : "+str(average_loss)+" | "+str(i+1)+"th/"+str(batch_number)+"batches")
			sys.stdout.flush()

		return average_loss

	def run_validation(self):
		valid_size = len(self.x_valid)
		validation_out = self.batch_out(self.x_valid)
		validation_loss = self.batch_loss(validation_out,self.y_valid)
		print "\nValidation loss:  ", validation_loss
		return validation_loss

	def run_test(self):
		test_size = len(self.x_test)
		self.test_out = self.batch_out(self.x_test)
		self.test_loss = self.batch_loss(self.test_out,self.y_test)
		print "Test loss:  ", self.test_loss
		return self.test_loss, self.test_out

	def run_train_outputs(self):
		train_size = len(self.x_train)
		fc1 = full_connected(self.x_train,self.para['W1'],bias=W1_bias,activation=fc1_activation)
		fc2 = full_connected(fc1,self.para['W2'],bias=W2_bias,activation=fc2_activation)
		output = full_connected(fc2,self.para['W_out'],bias=False,activation=fc_out_activation)
		self.train_output = output
	
		self.train_out = self.batch_out(self.x_train)
		if input_file == 'two_moon.txt':
			fc1 = full_connected(self.x_valid,self.para['W1'],bias=W1_bias,activation=fc1_activation)
			fc2 = full_connected(fc1,self.para['W2'],bias=W2_bias,activation=fc2_activation)
			output = full_connected(fc2,self.para['W_out'],bias=False,activation=fc_out_activation)
			self.valid_output = output

	#def binary_accuracy(self):

	def run(self):
		self.initialize_variable()
		self.get_data(input_file)
		train_loss = []
		validation_loss = []
		
		for i in range(epoch):
			_train_loss_ = self.run_train_epoch()
			train_loss.append(_train_loss_)
			if input_file != 'data_batch' and valid_ratio != 0:
				_validation_loss_ = self.run_validation()
				validation_loss.append(_validation_loss_)
				#if i>1000 and np.max(validation_loss[i-500:i]) < validation_loss[i]:
				#	break
		_,out = self.run_test()
		
		##Save the results
		if input_file != 'Samsung Stock Price.txt':
			accuracy = one_hot_accuracy(out,self.y_test)
			print accuracy
		
		np.save(input_file+'_train_loss'+str(self.lag)+'_'+str(self.TT)+'.npy',train_loss[100:])
		#np.save(input_file+'all_train_loss'+'.npy',self.train_record)
		if valid_ratio == 0:
			np.save(input_file+'_validation_loss'+str(self.lag)+'_'+str(self.TT)+'.npy',validation_loss[100:])
		np.save(input_file+'output'+str(self.lag)+'_'+str(self.TT)+'.npy',out)
		np.save(input_file+'_test_truth'+str(self.lag)+'_'+str(self.TT)+'.npy',self.y_test)
		np.save(input_file+'_test_loss'+str(self.lag)+'_'+str(self.TT)+'.npy',self.test_loss)
		if input_file != 'Samsung Stock Price.txt':
			np.save(input_file+'_test_accuracy'+'.npy',accuracy)
			self.run_train_outputs()
			np.save(input_file+'train_output'+'.npy',self.train_out)
			np.save(input_file+'y_train'+'.npy',self.y_train)

			if input_file == 'two_moon.txt':
				np.save(input_file+'valid_output'+'.npy',self.valid_output)
				np.save(input_file+'y_valid'+'.npy',self.y_valid)	
		return
for inp in [5,10,15,20]:
	for T_temp in [1,10,30,180]: 
			input_dim = inp
			out_ahead = T_temp
			model1  = Model(inp,T_temp)
			model1.run()














