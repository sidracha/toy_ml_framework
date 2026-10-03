#include "tensor.h"

#include <vector>
#include <utility>

class Dataset {
	// hold all the tensors... indexed by 1.... then we can batch size maxx it...
	// just store for now as [1, S, E]
	// then shuffle it easily... then just run it
	// Tensor literally easy ass copy

public:
	std::vector<std::pair<Tensor, Tensor>> data_tensors;
	std::vector<std::pair<Tensor, Tensor>> epoch_tensors;
	int index = 0;

	Dataset() {}

	std::pair<Tensor, Tensor> get_input();
	void shuffle();
	void prepare_epoch(int batch_size);
};
