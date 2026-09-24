#include <vector>
#include "tensor.h"
#include "calc.h"


void add_backward(TensorNode* node);
void sub_backward(TensorNode* node);

void mult_backward(TensorNode* node);
void div_backward(TensorNode* node);
void MATMUL_2D_backward(TensorNode* node);
void BIAS_ADD_2D_1D_backward(TensorNode* node);
