#Config for two_moon

input_dim = 10
output_dim = 1
out_ahead = 1
layer1_nodes = 100
layer2_nodes = 100
learning_rate = 0.00005
regularization=True
regularization_parameter = 0.001
type_regularization_ = 'L2'
train_ratio = 0.5
valid_ratio = 0
test_ratio = 0.5
epoch = 3000
batch_size = 5
fc1_activation = 'Sigmoid'
fc2_activation = 'Sigmoid'
fc_out_activation ='Identity'
loss_type = 'MSE'

W1_init = 'Xavier'
W2_init = 'Xavier'
W_out_init = 'normal'

W1_bias = True
W2_bias = True
