#include "tensor.h"
#include "dataset.h"
#include "nn.h"
#include <random>
#include <utility>

// uh so how should this dataset look? it should hold some 
// sequence of tensors, and be able to index into it? should i told 
// one tensor or a sequence of tensors... right now it can holdsequence

// returns the input tensor with the batch size
// at the next index already previously shuffled tho...
std::pair<Tensor, Tensor> Dataset::get_input() {
	return epoch_tensors[index++];
}

void Dataset::shuffle() {
	std::random_device rd;
	std::mt19937 gen(rd());

	std::shuffle(data_tensors.begin(), data_tensors.end(), gen);
}

// run this before every one of your epochs
void Dataset::prepare_epoch(int batch_size) {
	epoch_tensors.clear();
	index = 0;
	//shuffle();
	int N = data_tensors.size();
	// thennn do the thing with the cat
	int num_batches = std::ceil((double) N / (double) batch_size);
	for (int i=0; i<num_batches; i++) {
		// now inside make the vector and keep reading it...
		std::vector<Tensor> batch_input;
		std::vector<Tensor> batch_target;
		int start = i * batch_size;
		for (int j = start; j<std::min(N, start+batch_size); j++) {
			// create the batch here.... then concatenate
			batch_input.push_back(data_tensors[j].first);
			batch_target.push_back(data_tensors[j].second);

		}
		// turn off grad for the concat since we dont want to do allat
		// backward stuff 
		Tensor input = concat(batch_input, 0, false);
		Tensor target = concat(batch_target, 0, false);
		epoch_tensors.push_back({input, target});
	}
}
