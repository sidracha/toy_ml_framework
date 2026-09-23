#include <iostream>

#include "tensor.h"

int main () {
	Tensor* a = new Tensor(5);
	Tensor* b = new Tensor(10);
	Tensor* c = new Tensor(4);

	Tensor* x = *c +(*a) * b;
	x = *x * b;
	x->backward();
	std::cout << x->value << " " << b->grad << std::endl;
	


	return 0;
}
