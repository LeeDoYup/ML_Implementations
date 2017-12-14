#Config for two_moon
#Check train+valid+test = 1

input_dim = 2
output_dim = 2
layer1_nodes = 10
layer2_nodes = 10
learning_rate = 0.0005
regularization=True
regularization_parameter = 0.001
type_regularization_ = 'L2'
train_ratio = 0.5
valid_ratio = 0.2
test_ratio = 0.3
epoch = 100000
batch_size = 40
fc1_activation = 'Sigmoid'
fc2_activation = 'Sigmoid'
fc_out_activation ='Softmax'
loss_type = 'cross_entropy'

W1_init = 'Xavier'
W2_init = 'Xavier'
W_out_init = 'normal'

W1_bias = True
W2_bias = True
