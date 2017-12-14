import numpy as np
import math
import sklearn as sl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy as sp
import metric_learn as ml
import pdb

def read_iris(file_name, include_response=True, file_type = '.txt'):
	if file_type == '.txt':
		data = np.loadtxt(file_name)

	x_data  = data[:,:-1]
	y_data = data[:,-1]

	y_data = y_data -1

	return x_data, y_data


def similarity(x1, x2):
	sim  = np.dot(x1-x2, x1-x2)
	sim = np.sqrt(sim)
	return sim

def k_NN_classifier(target, train_x, train_y, k=25, criteria='min'):
	num_data = len(train_y)
	num_class = 3
	sim_list = []
	for i in range(num_data):
		sim_list.append(similarity(target, train_x[i]))

	sort_sim_index = np.argsort(sim_list)
	class_result_list = np.zeros([num_class])

	search_result = sort_sim_index[:k]

	for i in range(len(search_result)):
		class_result_list[int(train_y[search_result[i]])] += 1

	result = np.argmax(class_result_list)

	return result


x_iris, y_iris = read_iris('iris.txt')

def shuffle_data(x_data, y_data):
	temp_index = range(len(x_data))

	np.random.shuffle(temp_index)

	x_temp = np.zeros(x_data.shape)
	y_temp = np.zeros(y_data.shape)
	x_temp = x_data[temp_index]
	y_temp = y_data[temp_index]

	return x_temp, y_temp

def class_accuracy(y_out, y_test):
	count = 0.0
	for i in range(len(y_out)):
		if int(y_out[i]) == int(y_test[i]):
			count+=1

	return count/len(y_out)

def iris_LFDA(x_data, y_data):
	x_shuffle, y_shuffle = shuffle_data(x_data, y_data)
	x_fold = []
	y_fold = []

	for i in range(4):
		x_fold.append(x_shuffle[30*i:30*(i+1),:])
		y_fold.append(y_shuffle[30*i:30*(i+1)])
	x_fold.append(x_shuffle[120:,:])
	y_fold.append(y_shuffle[120:])

	accuracy=[]

	for i in range(5):
		temp = range(5)
		temp.remove(i)
		x_train = np.concatenate([x_fold[j] for j in temp],axis=0)
		y_train = np.concatenate([y_fold[j] for j in temp],axis=0)
		x_test = x_fold[i]
		y_test = y_fold[i]

		class_result = []
		lfda = ml.LFDA(k=2, dim=3)
		lfda.fit(x_train, y_train)

		result_y = result_y = lfda.transform(x_test)
		class_result = []
		for k in range(len(result_y)):
			class_result.append(k_NN_classifier(result_y[k], result_y, y_test))

		accuracy.append(class_accuracy(class_result, y_test))

	return accuracy

#iris_LFDA(x_iris, y_iris)

def iris_FLA(x_data, y_data):
	x_shuffle, y_shuffle = shuffle_data(x_data, y_data)
	x_fold = []
	y_fold = []

	for i in range(4):
		x_fold.append(x_shuffle[30*i:30*(i+1),:])
		y_fold.append(y_shuffle[30*i:30*(i+1)])
	x_fold.append(x_shuffle[120:,:])
	y_fold.append(y_shuffle[120:])

	accuracy=[]

	for i in range(5):
		temp = range(5)
		temp.remove(i)
		x_train = np.concatenate([x_fold[j] for j in temp],axis=0)
		y_train = np.concatenate([y_fold[j] for j in temp],axis=0)
		x_test = x_fold[i]
		y_test = y_fold[i]

		class_result = []
		fla = LinearDiscriminantAnalysis()
		fla.fit(x_train, y_train)

		result_y = fla.transform(x_test)
		class_result = []
		for k in range(len(result_y)):
			class_result.append(k_NN_classifier(result_y[k], result_y, y_test))

		accuracy.append(class_accuracy(class_result, y_test))

	return accuracy

'''


	for iter in range(5):
		class_result = []
		fla = LinearDiscriminantAnalysis()
		fla.fit(x_data, y_data)
		for i in range(len(y_data)):
			class_result.append(int(fla.predict(x_data[i])[0]))
'''
accuracy_LFDA30 = []
accuracy_FLA30 = []

accuracy = []
std = []

for i in range(30):
	accuracy_LFDA30.append(iris_LFDA(x_iris, y_iris))
	accuracy_FLA30.append(iris_FLA(x_iris, y_iris))
accuracy.append(np.mean(accuracy_LFDA30))
accuracy.append(np.mean(accuracy_FLA30))

accuracy_LFDA15 = []
accuracy_FLA15 = []

for i in range(15):
	accuracy_LFDA15.append(iris_LFDA(x_iris, y_iris))
	accuracy_FLA15.append(iris_FLA(x_iris, y_iris))

accuracy.append(np.mean(accuracy_LFDA15))
accuracy.append(np.mean(accuracy_FLA15))

accuracy_LFDA5 = []
accuracy_FLA5 = []

for i in range(5):
	accuracy_LFDA5.append(iris_LFDA(x_iris, y_iris))
	accuracy_FLA5.append(iris_FLA(x_iris, y_iris))

accuracy.append(np.mean(accuracy_LFDA5))
accuracy.append(np.mean(accuracy_FLA5))


std.append(np.std(accuracy_LFDA30))
std.append(np.std(accuracy_FLA30))

std.append(np.std(accuracy_LFDA15))
std.append(np.std(accuracy_FLA15))

std.append(np.std(accuracy_LFDA5))
std.append(np.std(accuracy_FLA5))


pdb.set_trace()

