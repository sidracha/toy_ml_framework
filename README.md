basic toy framework works

implemented by hand in a day for fun as an exercise

Parts that work:
- add/mult/div/sub
- GEMM2D and Bias add
- Sigmoid
- ReLU
- backward functions for all this
- autograd/autodiff w/automatic deletion of intermediate nodes
- Linear layer
- simple Tensor permutations/arbitrary shapes/sizes for tensor
- simple FFN trains

TODO (in order of priority):
- (DONE) change make_operator_output_node function take a list of predecessors
- (DONE) have Tensor be param by reference, dont copy tensor objects everywhere
- add 3D batched tensors
- add Model class custom forward methods
- Xavier initialization
- Add single ReLU backward
- Add other activation functions
- Convolution
- CUDA backends for GEMM
