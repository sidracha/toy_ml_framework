basic toy framework works

implemented BY HAND from first principles, design not based on anything

Since ive "had experience" with Pytorch mostly, the workflow is the same: input, loss, optimizer, zero_grad etc,
but the underlying framework is different 

Parts that work:
- Autograd
- Optimizer
- Tensor
- Tensor shape/stride aware
- Transpose
- GEMM2D and bias add, broadcast add
- Sigmoid
- ReLU
- Softmax
- LayerNorm
- Linear layer
- FNN MLP
- Concatenate
- Self attention
- Multiheaded self attention
- Tanh
- Transformer block
- Transformer/MLP train
- Automatic graph clearing

DFT predictor (iterations /40 is the loss scale, see fourier_train.cc)
  
<img width="1301" height="392" alt="Screenshot 2026-09-29 at 10 10 39 PM" src="https://github.com/user-attachments/assets/605c2c61-f05a-4cbb-af4c-3bc13da8c1d3" />

TODO (in order of priority):
- Optimizer holds params
- GeLU
- Swish
- (DONE) change make_operator_output_node function take a list of predecessors
- (DONE) have Tensor be param by reference, dont copy tensor objects everywhere
- (DONE) add 3D batched tensors
- work with general tensor shapes in MSELoss
- (DONE) add Model class custom forward methods
- Xavier initialization
- Add single ReLU backward
- Convolution
- CUDA backends for GEMM


