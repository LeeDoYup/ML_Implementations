import numpy as np 
import scipy.io 
import pdb

def perms(l):
	sz = len(l)
	if sz ==1: return [l]
	return [p[:i]+[l[0]]+p[i:] for i in xrange(sz) for p in perms(l[1:])]


def LSA(term_doc,_X,U,s,V,T):
	num_word = len(U)
	num_cluster = 4
	num_doc = len(V.T)

	doc_mtc = np.dot(s,V) #[# Factors * # documents]
	doc_mtc = doc_mtc.T

	cluster_result = K_means(doc_mtc,num_cluster)
	accuracy = Check_accuracy(cluster_result, T)

	return accuracy
	#pdb.set_trace()

def K_means(input,num_cluster):
	num_doc, num_factor = np.shape(input)
	centroids = np.zeros([num_cluster,num_factor])
	temp = range(len(input))
	np.random.shuffle(temp)

	iteration = 200
	
	for i in range(num_cluster):
		centroids[i,:] = input[temp[i],:]

	for iter in range(iteration):
		temp = np.zeros([len(input),num_cluster])
		for i in range(len(input)):
			for j in range(num_cluster):
				temp[i,j] = 1-np.dot(centroids[j,:],input[i,:])/np.linalg.norm(centroids[j,:])/np.linalg.norm(input[i,:])
		cluster_result = np.argmin(temp,axis=1)

		#pdb.set_trace()
		for i in range(num_cluster):
			num = 0
			for j in range(len(cluster_result)):
				if cluster_result[j] ==  i :
					num = num + 1
					centroids[i,:] += input[j,:]
			centroids[i,:] /= num
	
	return cluster_result


def PLSA(term_doc, U, s, V, T):
	num_word = len(U)
	num_cluster = 4
	num_doc = len(V.T)
	
	#initialize variable

	p_w_z = np.random.ranf([num_word,num_cluster])
	p_w_z = p_w_z / np.reshape(np.sum(p_w_z, axis=1),[num_word,1])

	p_d_z = np.random.ranf([num_doc,num_cluster])
	p_d_z = p_d_z / np.reshape(np.sum(p_d_z, axis=1),[num_doc,1])

	p_z = np.random.ranf([num_cluster,1])
	p_z = p_z/np.sum(p_z)

	p_z_wd = np.ones([num_cluster,num_word,num_doc]) 
	p_z_wd = p_z_wd / np.reshape(np.sum(np.sum(p_z_wd,axis=2),axis=1),[num_cluster,1,1])

	print 1
	cluster_result = EM_run(p_w_z,p_d_z,p_z,p_z_wd,term_doc)
	accuracy = Check_accuracy(cluster_result, T)

	return accuracy


def E_step_PLSA(p_w_z,p_d_z,p_z):
	print 1
	num_word = len(p_w_z)
	num_doc = len(p_d_z)
	num_cluster = len(p_z)
	temp_p_z_wd = np.zeros([num_cluster, num_word, num_doc])

	for k in range(num_cluster):
		for i in range(num_word):
			for j in range(num_doc):
				temp = 0
				for l in range(num_cluster):
					temp = temp + p_w_z[i,l]*p_d_z[j,l]*p_z[l]
				temp_p_z_wd[k,i,j] = ( p_w_z[i,k] * p_d_z[j,k] * p_z[k] ) / temp

	return temp_p_z_wd

def M_step_PLSA(p_w_z,p_d_z,p_z,p_z_wd,term_doc):
	print 1

	num_word = len(p_w_z)
	num_doc = len(p_d_z)
	num_cluster = len(p_z)

	for k in range(num_cluster):
		for i in range(num_word):
			p_w_z[i,k] = np.sum(term_doc[i,:] * p_z_wd[k,i,:]) / np.sum(term_doc * p_z_wd[k,:,:])
		for j in range(num_doc):
			p_d_z[j,k] = np.sum(term_doc[:,j] * p_z_wd[k,:,j]) / np.sum(term_doc * p_z_wd[k,:,:])
		p_z[k] = np.sum(term_doc * p_z_wd[k,:,:]) / np.sum(term_doc)

	return p_w_z, p_d_z, p_z


def EM_run(p_w_z,p_d_z,p_z,p_z_wd,term_doc):
	for i in range(50):
		p_z_wd = E_step_PLSA(p_w_z,p_d_z,p_z)
		p_w_z,p_d_z,p_z = M_step_PLSA(p_w_z,p_d_z,p_z,p_z_wd,term_doc)

	num_doc = len(p_d_z)
	num_cluster = len(p_z)
	p_z_d = np.zeros([num_cluster,num_doc])
	for k in range(num_cluster):
		for j in range(num_doc):
			p_z_d[k,j] = p_d_z[j,k] * p_z[k]

	cluster_result = np.argmax(p_z_d,axis=0)
	return cluster_result

def Check_accuracy(cluster_result, T):
	#T_index = np.argmax(T,axis=0)
	accuracy = 0.0
	T_num = len(cluster_result)
	for p in perms(range(4)):
		_accuracy =0.0
		T_temp = np.zeros(np.shape(T))
		for q in range(len(p)):
			T_temp[q,:] = T[p[q],:]
		T_index = np.argmax(T_temp,axis=0)
		for i in range(T_num):
			if cluster_result[i] == T_index[i]:
				_accuracy +=1
		_accuracy = _accuracy/T_num
		if accuracy < _accuracy:
			accuracy = _accuracy
	print accuracy
	return accuracy

'''
#for num_factor in [115, 150, 200, 250, 300]:
for num_factor in [115, 150, 200, 250, 300]:
	mat = scipy.io.loadmat('cstr_revised.mat')

	term_doc = mat[mat.keys()[0]]
	T = mat[mat.keys()[3]]

	temp_U, temp_s, temp_V = np.linalg.svd(term_doc,0)

	U = temp_U[:,:num_factor]

	#print np.sum(temp_s[:num_factor+1])/np.sum(temp_s)

	s = np.diag(temp_s[:num_factor])
	V = temp_V[:num_factor,:]

	_X = np.dot(U,s)
	_X = np.dot(_X,V)
	accuracy_record=[]
	for it in range(100):
		temp = LSA(term_doc, _X, U, s, V, T)
		accuracy_record.append(temp)
	np.save('LSA_Accuracy'+str(num_factor)+'.npy',accuracy_record)
#
'''
num_factor = 150
accuracy_record=[]
for iter in range(10):
	print iter
	print "Trials"
	mat = scipy.io.loadmat('cstr_revised.mat')

	term_doc = mat[mat.keys()[0]]
	print np.shape(term_doc)
	T = mat[mat.keys()[3]]

	temp_U, temp_s, temp_V = np.linalg.svd(term_doc,0)

	U = temp_U[:,:num_factor]

	#print np.sum(temp_s[:num_factor+1])/np.sum(temp_s)

	s = np.diag(temp_s[:num_factor])
	V = temp_V[:num_factor,:]

	_X = np.dot(U,s)
	_X = np.dot(_X,V)
	
	accuracy = PLSA(term_doc, U, s, V, T)
	accuracy_record.append(accuracy)

np.save('PLSA_Accuracy.npy',accuracy_record)


