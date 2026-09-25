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

Sin predictor 2500 iterations
- model.register_layer<LinearReLU>(1, 32);
- model.register_layer<LinearSigmoid>(32, 32);
- model.register_layer<Linear>(32, 1);
  
<img width="492" height="395" alt="Screenshot 2026-09-25 at 12 00 01 AM" src="https://github.com/user-attachments/assets/b4d8cf9e-3aed-4b72-a816-24cf88889cdf" />


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


