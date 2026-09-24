#include "tensor.h"
#include "layer.h"

Tensor create_expect_sine(Tensor t);
Tensor create_validation_set(int batch_size, Graph* graph);
void train_sine(SequentialModel& model);
