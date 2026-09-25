#include "tensor.h"
#include "layer.h"

using test_fn_type = std::function<double(double)>;
Tensor create_expect_fn(Tensor t, test_fn_type fn);
//Tensor create_validation_set(int batch_size, Graph* graph);
void train_fn(SequentialModel& model);
