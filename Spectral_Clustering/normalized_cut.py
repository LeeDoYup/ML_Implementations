import numpy as np
import scipy.io 
import pdb
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize
from PIL import Image

def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:         
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
  return im_arr

def K_means(input,num_cluster):
	print np.shape(input)
	num_data, num_dim = np.shape(input)
	centroids = np.zeros([num_cluster,num_dim])
	temp = range(len(input))
	np.random.shuffle(temp)

	iteration = 20
	
	for i in range(num_cluster):
		centroids[i,:] = input[temp[i],:]

	for iter in range(iteration):
		temp = np.zeros([len(input),num_cluster])
		for i in range(len(input)):
			for j in range(num_cluster):
				temp[i,j] = np.matmul((centroids[j,:]-input[i,:]).T,(centroids[j,:]-input[i,:]))
		cluster_result = np.argmin(temp,axis=1)
		#pdb.set_trace()
		for i in range(num_cluster):
			num = 0
			for j in range(len(cluster_result)):
				if cluster_result[j] ==  i :
					num = num + 1
					centroids[i,:] += input[j,:]
			centroids[i,:] /= num
	return cluster_result, centroids

def Gaussian_similarity(x1, x2, sigma):
	return np.exp(-np.matmul((x1-x2).T,(x1-x2))/(2*sigma*sigma))

#v means vertices
def e_Neighbor_Similarity_Graph(v,epsilon):
	print "Construnct Similarity Graph\n"
	epsilon = 0.7
	data_size = len(v)
	sim_graph = np.zeros([data_size, data_size])
	for i in range(data_size):
		for j in range(data_size):
			sim_graph[i,j] = Gaussian_similarity(v[i],v[j],sigma=0.5)
			if sim_graph[i,j] > epsilon:
				sim_graph[i,j] = 1
			else:
				sim_graph[i,j] = 0


	print "Complete Construnction of  Similarity Graph\n"
	return sim_graph

def Similarity_Graph(v):
	print "Construnct Similarity Graph\n"
	data_size = len(v)
	sim_graph = np.zeros([data_size, data_size])
	for i in range(data_size):
		for j in range(data_size):
			sim_graph[i,j] = Gaussian_similarity(v[i],v[j],sigma=1)

	print "Complete Construnction of  Similarity Graph\n"
	return sim_graph

def Degree_Graph(sim_graph):
	print "Calculate Degree Graph\n"
	data_size = len(sim_graph)
	degree_list = np.sum(sim_graph,axis=1)
	degree_graph = np.zeros([data_size,data_size])
	for i in range(data_size):
		degree_graph[i,i] = degree_list[i]
	print "Complete Degree Graph\n"
	return degree_graph

def N_rw_Graph_Laplacian(sim_graph):
	print "Calculate Normalized Graph Laplacian\n"
	degree_graph = Degree_Graph(sim_graph)
	N_rw_GL = np.matmul(np.linalg.pinv(degree_graph),(degree_graph-sim_graph))
	print "Complete Normalized Graph Laplacian"
	return N_rw_GL

def N_Spectral_Clustering(vertices, k):
	sim_graph = Similarity_Graph(vertices)
	N_rw_GL = N_rw_Graph_Laplacian(sim_graph)

	eigenvalues, eigenvectors = np.linalg.eig(N_rw_GL)

	U = eigenvectors[:,:k]
	U = np.real(U)
	clustering_result, centroids = K_means(U,k)
	return clustering_result, centroids

def read_data(file_name):
	mat = scipy.io.loadmat(file_name)
	data = mat['x']
	return data

def read_image(file_name):
	data = imread(file_name)
	data = data/255
	image_shape= np.shape(data)
	data = np.reshape(data, [image_shape[0]*image_shape[1],image_shape[2]])
	return data

def plot_cluster_result(data, clustering_result):
	data_size = len(data)
	
	index0 = [v for v in range(data_size) if clustering_result[v]==0]
	index1 = [v for v in range(data_size) if clustering_result[v]==1]
	
	plt.plot(data[index0,0],data[index0,1],'ro',data[index1,0],data[index1,1],'bs')
	plt.show()
	pdb.set_trace()

def recover_image_result(original_image, clustering_result, centroids):
	image_size = np.size(original_image)
	image_result = [centroids[i,:] for i in clustering_result]
	image_result = np.reshape(image_result, [image_size[0],image_size[1],image_size[2]])
	return image_result

def plot_image(result):
	plt.imshow(result)
	plt.show()

#data = read_data('two_moons.mat')
data = read_image('tiger.jpeg')
c_result , centroids = N_Spectral_Clustering(data,3)
image_result = recover_image_result(data, c_result, centroids)
plot_image(image_result)
np.save('Image_result.npy',image_result)
#plot_cluster_result(data,c_result)









