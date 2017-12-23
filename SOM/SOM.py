import numpy as np 
import pdb
import matplotlib.pyplot as plt

def make_H_data(num_data):
	data = np.random.rand(num_data,2)
	data[:,0] = 3 * data[:,0]
	for i in range(num_data):
		if data[i,0] < 1 or data[i,0] >2:
			data[i,1] = data[i,1] * 3
		else:
			data[i,1] = data[i,1] + 1
	return data

def shuffle_data(x_train):
    temp_index = np.arange(len(x_train))
    #Random suffle index
    np.random.shuffle(temp_index)
    #Re-arrange x and y data with random shuffle index
    x_temp = np.zeros(x_train.shape)
    x_temp = x_train[temp_index]    
    return x_temp

def Gaussian_Neighbor(r1, r2, sig):
	return np.exp(-np.dot(r1-r2,r1-r2)/(2*sig*sig))

def Flat_Neighbor(r1, r2, sig):
	if np.sqrt(np.dot(r1-r2,r1-r2)) <= sig:
		return 1
	else:
		return 0

class SOM(object):
	def __init__(self):
		print "Default Initializer"

	def __init__(self, learning_rate, neighbor_type, sig):
		print "Create SOM Class with Learning Rate"
		self.learning_rate = learning_rate
		self.neighbor_type = neighbor_type
		self.sig = sig
		self.grad_radius = False
		self.grad_learning_rate = True
		self.epoch = 0 
		self.total_epoch = 5000
		self.distance = []
		self.opt_distance = 9999.9
		self.save_result = []
		self.K = 500

	def input_data(self, data):
		self.data = data
		self.num_data, self.num_feature = np.shape(data)

	def construct_som(self, som_shape, som_type = 'L2'):
		#Construnct SOM structure and initialize weights
		self.som_type = som_type
		self.neuron_shape = som_shape
		self.weight_shape = som_shape + (self.num_feature,)
		data_min = np.tile(np.min(self.data, axis=0), som_shape+(1,))
		data_max = np.tile(np.max(self.data, axis=0), som_shape+(1,))
		if som_type == 'L1':
			self.weights = data_min + (data_max-data_min)*np.random.rand(self.weight_shape[0], self.weight_shape[1])
		elif som_type == 'L2':
			self.weights = data_min + (data_max-data_min)*np.random.rand(self.weight_shape[0], self.weight_shape[1], self.weight_shape[2])
		self.is_weight_init = True

	def find_closest_neuron(self, target):
		temp = self.weights - target
		temp = np.power(temp,2)

		temp = np.sum(temp, axis=1)
		closest_index = np.argmin(temp)
		return closest_index

	def neighborhood_value(self, r1, r2, sig):
		if self.neighbor_type == 'Gaussian':
			return Gaussian_Neighbor(r1,r2,sig)
		elif self.neighbor_type == 'Flat':
			return Flat_Neighbor(r1,r2,sig)
		else:
			print "Neighborhood Type is wrong"
			pdb.set_trace()

	def neighborhood_function(self, pivot):
		nf = np.ones(self.neuron_shape)
		if self.som_type == 'L1':
			for i in range(self.neuron_shape[0]):
				nf[i] = self.neighborhood_value(pivot, i, self.sig)

		elif self.som_type == 'L2':
			pivot_row = pivot%(self.neuron_shape[0])
			pivot_column = pivot - pivot_row * self.neuron_shape[1]
			for i in range(self.neuron_shape[0]):
				for j in range(self.neuron_shape[1]):
					nf[i,j] = self.neighborhood_value([pivot_row, pivot_column], [i,j], self.sig)
		else:
			print "Neighborhood Function Error"
			pdb.set_trace()

		return nf

	def update_weight(self, target):
		closest_index = self.find_closest_neuron(target)
		neighbor_effect = self.neighborhood_function(closest_index)

		if self.som_type == 'L1':
			for i in range(self.weight_shape[0]):
				self.weights[i] += self.learning_rate * neighbor_effect[i] * (target - self.weights[i])
		elif self.som_type == 'L2':
			for i in range(self.weight_shape[0]):
				for j in range(self.weight_shape[1]):
					self.weights[i,j] += self.learning_rate * neighbor_effect[i,j] * (target - self.weights[i,j])

	def batch_run(self,batch_x):
		for i in range(len(batch_x)):
			self.update_weight(batch_x[i])

	def epoch_run(self, batch_size):
		self.data = shuffle_data(self.data)
		batch_num = int(len(self.data)/batch_size)
		for iter in range(batch_num):
			self.batch_run(self.data[iter*batch_size:(iter+1)*batch_size])

		self.epoch +=1
		if self.grad_radius == True:
			self.sig = self.sig * (1 - self.epoch / self.total_epoch)

		if self.grad_learning_rate == True:
			self.learning_rate = self.learning_rate * (1 - self.epoch / (self.total_epoch+self.K))
			return

	def TSP_result(self):
		index_list = np.zeros([len(self.data)])
		result = np.zeros(self.data.shape)

		for i in range(len(self.data)):
			index_temp = self.find_closest_neuron(self.data[i])
			index_list[i] = index_temp

		index_result = np.argsort(index_list)
		for i in range(len(index_result)):
			result[i] = self.data[index_result[i]]

		self.result = result
		self.save_result.append(result)
		temp_distance = self.TSP_distance()
		self.distance.append(temp_distance)

		if self.opt_distance > temp_distance:
			self.opt_distance = temp_distance
			self.opt_soltuion = result
			self.opt_weights = self.weights

	def TSP_distance(self):
		distance = 0.0
		for i in range(len(self.result)):
			if i < len(self.result)-1:
				distance += np.sqrt(np.dot(self.result[i]-self.result[i+1],self.result[i]-self.result[i+1]))
			else:
				distance += np.sqrt(np.dot(self.result[i]-self.result[0],self.result[i]-self.result[0]))
		print distance
		return distance

#Command for TSP

TSP_data = np.array([[0.2, 0.1], [0.15,0.2], [0.4,0.45], [0.2, 0.77], [0.5, 0.9], [0.83, 0.65], [0.7,0.5], [0.82,0.35], [0.65,0.23], [0.6,0.28]])
som1d = SOM(0.5,'Gaussian',1.0)
#def __init__(self, learning_rate, neighbor_type, sig, grad_radius, grad_learning_rate, K):
som1d.input_data(TSP_data)
som1d.construct_som((10,),'L1')

for iter in range(som1d.total_epoch):
	som1d.epoch_run(1)
	if iter%50 == 0:
		som1d.TSP_result()

plt.plot(TSP_data[:,0], TSP_data[:,1], 'ro')
plt.plot(som1d.opt_weights[:,0], som1d.opt_weights[:,1], 'bo')
plt.plot(som1d.opt_weights[:,0], som1d.opt_weights[:,1])
print som1d.distance
plt.show()
pdb.set_trace()


'''

#Command for H data

H_data = make_H_data(250)
som1d = SOM(0.5,'Gaussian',1)
som1d.input_data(H_data)
som1d.construct_som((30,),'L1')

for iter in range(som1d.total_epoch):
	som1d.epoch_run(1)

plt.plot(som1d.data[:,0], som1d.data[:,1], 'bo')
plt.plot(som1d.weights[:,0], som1d.weights[:,1], 'bo')
plt.plot(som1d.weights[:,0], som1d.weights[:,1])
plt.show()

'''











	


