#include "tensor.h"

#include <vector>

class Layer {
public:
	Layer() {}

	virtual Tensor forward(Tensor t) = 0;
};
