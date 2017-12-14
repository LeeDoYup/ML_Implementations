#Config for CIFAR10_data
#Check train+valid+test = 1

input_dim = 3072 
output_dim = 10
layer1_nodes = 100
layer2_nodes = 300
learning_rate = 0.0005
regularization=True
regularization_parameter = 0.0001
type_regularization_ = 'L2'
epoch = 1000
batch_size = 10000
fc1_activation = 'Sigmoid'
fc2_activation = 'ReLU'
fc_out_activation ='Softmax'
loss_type = 'cross_entropy'

W1_init = 'Xavier'
W2_init = 'Xavier'
W_out_init = 'normal'

W1_bias = True
W2_bias = True
