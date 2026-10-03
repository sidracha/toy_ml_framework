#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "trainers.h"
#include "optimizer.h"
#include "attention.h"
#include "losses.h"
#include "nn.h"
#include "burgers.h"


int main () {
	
	//fft_train_loop();
	burgers_train_loop();
	return 1;
	
}
