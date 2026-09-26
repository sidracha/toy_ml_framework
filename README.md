basic toy framework works

code implemented by hand in a weekend for fun n learning

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

Sin predictor 10000 iterations
Sigmoid except last 3 layers of 32
  
<img width="525" height="394" alt="Screenshot 2026-09-25 at 11 38 16 PM" src="https://github.com/user-attachments/assets/7aadf235-208a-4ed3-b8a3-128c73c0f3cf" />


TODO (in order of priority):
- (DONE) change make_operator_output_node function take a list of predecessors
- (DONE) have Tensor be param by reference, dont copy tensor objects everywhere
- some sort of list of params that optimizer holds that can iterate through
- (DONE) add 3D batched tensors
- work with general tensor shapes in MSELoss
- add Model class custom forward methods
- Xavier initialization
- Add single ReLU backward
- Add other activation functions
- Convolution
- CUDA backends for GEMM


