basic toy framework works

code implemented BY HAND in a few few days for fun n learning

Parts that work:
- Multi headed self attention Full Transformer
- Softmax/layernorm/MLP/qkv/concat
- add/mult/div/sub
- GEMM2D and Bias add
- Sigmoid
- ReLU
- backward functions for all this
- autograd/autodiff w/automatic deletion of intermediate nodes
- Linear layer
- simple Tensor permutations/arbitrary shapes/sizes for tensor
- simple FFN trains

DFT predictor (iterations /40 is the loss scale, see fourier_train.cc)
  
<img width="1311" height="397" alt="Screenshot 2026-09-29 at 9 53 27 PM" src="https://github.com/user-attachments/assets/f1214456-0644-4418-9d9a-b3f17e8a425a" />


TODO (in order of priority):
- GeLU
- Swish
- (DONE) change make_operator_output_node function take a list of predecessors
- (DONE) have Tensor be param by reference, dont copy tensor objects everywhere
- some sort of list of params that optimizer holds that can iterate through
- (DONE) add 3D batched tensors
- work with general tensor shapes in MSELoss
- (DONE) add Model class custom forward methods
- Xavier initialization
- Add single ReLU backward
- Add other activation functions
- Convolution
- CUDA backends for GEMM


