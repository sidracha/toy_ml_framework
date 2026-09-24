basic toy framework works

Parts that work:
add/mult/div/sub
GEMM2D and Bias add
Sigmoid
ReLU
backward functions for all this
autograd/autodiff w/automatic deletion of intermediate nodes w/ graph.clear()
Linear layer
simple Tensor permuations/arbitrary shapes/sizes for tensor
simple FFN trains

TODO (in order of priority):

have Tensor be param by reference, dont copy tensor objects everywhere
add 3D batched tensors
add Model class custom forward methods
Xavier initialization
Add single ReLU backward
Add other activation functions
Convolution
CUDA backends for GEMM
