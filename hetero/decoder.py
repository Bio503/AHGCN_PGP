import numpy
import torch
import numpy as np
import torch.nn.functional as F

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        # dtype=tf.compat.v1.float32
    )

    return torch.autograd.Variable(initial, name=name)


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=torch.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        # with tf.compat.v1.variable_scope(self.name + '_vars'):
        self.vars['weights'] = weight_variable_glorot(
            input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        # with tf.compat.v1.name_scope(self.name):
        inputs = torch.dropout(inputs, 1-self.dropout)
        R = inputs[0:self.num_r, :]
        D = inputs[self.num_r:, :]
        R = torch.matmul(R, self.vars['weights'])
        D = torch.transpose(D)
        x = torch.matmul(R, D)
        x = torch.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
