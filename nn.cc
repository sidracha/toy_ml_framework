#include "nn.h"

Tensor* ReLU(Tensor* a) {
	if (a->value <= 0) return (*a * 0.0);
	else return *a * 1.0;
}

std::vector<Tensor*>* ReLU(std::vector<Tensor*>* a) {
	std::vector<Tensor*>* ret = new std::vector<Tensor*>(a->size());
	for (int i=0; i<a->size(); i++) {
		(*ret)[i] = ReLU((*a)[i]);
	}
	return ret;
}
