#include "losses.h"

Tensor* MSELoss(Tensor* a, Tensor* b) {
	Tensor* diff = *a - b;
	return diff->pow(2.0);

}
