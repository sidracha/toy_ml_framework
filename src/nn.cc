#include "nn.h"
#include "tensor.h"
#include "calc.h"
#include "utils.h"

#include <cmath>
// ReLU is an operation and should return a Tensor, we also have to calculate the backward pass..
// for now its automatic
Tensor ReLU(const Tensor& t) {
	
	// we have to make sure the shape and the stride are the same... but when we create it with the shape,
	// it might be discontinuous...
	// do we create the gradient...?
	
	// i guess we can set the stride the same i guess...
	// create a tensor with the same shape and stride...
	
	Tensor new_tensor = create_tensor_clone_scalar(t, 0.0);
	// now its a clone, do the ReLU
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		int val;
		if (t.tensor_node->data[i] > 0.0) new_tensor.tensor_node->data[i] = 1.0; 
	}

	return t * new_tensor;

}

double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

Tensor Sigmoid (const Tensor& t) {
	// I guess this is just a pointwise operation
	// we have to create a new node for this... but will it be within the class?
	// we need some extendable way to do this
	std::vector<double> data(t.tensor_node->data.size());
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		// do the sigmoid operation
		double x = t.tensor_node->data[i];
		data[i] = sigmoid(x);
	}
	// create the output node
	std::shared_ptr<TensorNode> node = make_operator_output_node(
										data, 
										t.shape(),
										t.stride(),
										{t.tensor_node}, 
										Sigmoid_backward);
	
	return Tensor(node);
	
}


void Sigmoid_backward(TensorNode* node) {
	
	// this is a unary operator, so i guess we only have one predecessor...
	// but what? how do we register this as an operation...
	// just call sigmoid squared... that is the gradient... 
	// get the first parent
	TensorNode* a = node->predecessors[0].get();
	for (int i=0; i<a->data.size(); i++) {
		a->grad[i] += (node->data[i] * (1.0-node->data[i]) * node->grad[i]);
	}

}


// ugh we have to do this odometer shit where N goes across the last row,
// and we apply it independently across each row....
// kind of annoying but whatveer man

Tensor softmax(const Tensor& t) {
	
	std::vector<int> shape = t.shape();
	std::vector<int> stride = t.stride();

	// how many odometers? well the last index is the one we want to iterate over... for 
	// the soft max. so
	
	int N_index = t.dim()-1;
	int N = shape[N_index];
	std::vector<int> odometer(N_index, 0);
	int odometer_index = 0;
	
	std::vector<double> output(t.tensor_node->data.size());
	
	while (odometer_index >= 0) {
		
		// i guess we get the index offset here,
		// thennnn we do the actual softmax here... by iterating i through 
		// all the row elements and write into the output at the end
		
		// do the log sum exp trick
		double m = std::numeric_limits<double>::lowest();
		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * stride[N_index];
			m = std::max(m, t.tensor_node->data[data_index]);
		}
		
		double e_sum = 0.0;
		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * stride[N_index];
			e_sum += std::exp(t.tensor_node->data[data_index] - m);	
		}

		// ok so we got the toal one, now we should calculate the per-
		// element softmax

		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * stride[N_index];
			output[data_index] = std::exp(t.tensor_node->data[data_index] - m) / e_sum;
		}
	
		odometer_index = odometer_next(odometer, shape, stride);

	}

	// ok so then we create the output node here...
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape, {t.tensor_node}, softmax_backward);
	
	return Tensor(node);
	
}

// ok i derived it on paper its kind of annoying but whatveer
// let g be the upstream gradient row-wise vector

// and let z (i...n) be the input vector (row) into the row-wise
// softmax....
// we have L = f(y)
// y (i...n) = softmax(z)
// using the multivariate chain rule and some equation massaging on paper
//
// dL/dz_i = y_i * (g_i - sum(g_k * y_k) k0..n) 
// ok so yeah its O(n) thats basically the asnwer... first create
// the summation of g*y into a vector
// y is just the row-wise output so node->data i guess

void softmax_backward(TensorNode* node) {

	// how many odometers? well the last index is the one we want to iterate over... for 
	// the soft max. so
	
	// we only have one predecessor and its like the same
	// shape and such
	
	// so this is basically the "input z"
	TensorNode* A = node->predecessors[0].get();
	
	int N_index = node->dim()-1;
	int N = node->shape[N_index];
	std::vector<int> odometer(N_index, 0);
	int odometer_index = 0;
	
	while (odometer_index >= 0) {
		
		// just do everything per-row here
		// ok so first create the summation vector
		// with the uptream gradients
		
		double upstream_sum = 0.0;
		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * node->stride[N_index];
			upstream_sum += (node->grad[data_index] * node->data[data_index]);
		}

		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * node->stride[N_index];
			// write into the downstream z, which is A
			A->grad[data_index] += node->data[data_index] * (node->grad[data_index] - upstream_sum); 
		}

		odometer_index = odometer_next(odometer, node->shape, node->stride);

	}

}

// we want a concat operation here.
// this will take in a vector of tensors...
// how do we concat across a dim? 
// we basically write the rest of the ones in appropriately
// and then we do a loop iteration when 
// we get into the dim that we care about....
// and the odometer will be all of the other dims...
bool check_concat_dims(const Tensor& t1, const Tensor& t2, int dim) {
	std::vector<int> shape1 = t1.shape();
	std::vector<int> shape2 = t2.shape();
	if (shape1.size() != shape2.size()) return false;
	for (int i=0; i<shape1.size(); i++) {
		if (i == dim) continue;
		if (shape1[i] != shape2[i]) return false;
	}
	return true;
}


Tensor concat(const std::vector<Tensor>& array, int concat_dim) {

	// basically just reutn utself/noop if its
	// concat with 1, dont even add to ghte graph
	if (array.size() == 1) return array[0]; 
	
	//calculate the shape... and we can also calculate the stride
	// ok the dims have to match everywhere except for the dimension of the 
	// dim...
	for (int i=1; i<array.size(); i++) {
		if (!check_concat_dims(array[i], array[i-1], concat_dim)) {
			throw std::runtime_error("Cannot concatenate all the tensor shapes dont match up");
		}
	}
	// the final shape is the addition across all of those dims,
	// and the rest of the shapes is the same
	int final_dim = 0;
	for (const auto& t : array) {
		final_dim += t.shape()[concat_dim];
	}


	// create the strides and the shapes
	std::vector<int> output_shape = array[0].shape();
	int dim = output_shape.size();
	output_shape[concat_dim] = final_dim;
	std::vector<int> output_stride(dim);
	output_stride[dim-1] = 1;
	for (int i=dim-2; i>=0; i--) output_stride[i] = output_stride[i+1] * output_shape[i+1];
	
	// okk we create the output buffer here... easy peasy
	int total_values = 1;
	for (int i=0; i<dim; i++) total_values *= output_shape[i];

	std::vector<double> output(total_values);
	
	// now do the unordered map odometer BS, resolve 
	// the indivual dims for each one but
	// we verified that the shapes/dims are all equal
	// so we iterate from the first one.. over its dim size...
	// so we have the odometer... for each position of the odometer offset 
	// we want to write in all the values of that dim ok easy enough
	
	std::vector<int> odometer;
	std::vector<int> index_map;
	for (int i=0; i<dim; i++) {
		if (i == concat_dim) continue;
		odometer.push_back(0);
		index_map.push_back(i);
	}
	
	// now iterate over the odometer, then iterate over the array writing the values
	// in appropriately...
	// we have to probably keep track of the odometer offset globally since 
	// we enforce that the shape is the same...
	// ah fuck the stride ok wahtever ignore for now
	
	
	bool next_exists = true;
	while (next_exists) {
		// now iterate through it
		int output_dim_offset = 0;
		for (const auto& t : array) {
			int concat_dim_size = t.shape()[concat_dim];
			std::vector<int> input_stride = t.stride();
			
			for (int i=0; i<concat_dim_size; i++) {
					
				// find the input offset
				
				int input_index = i * input_stride[concat_dim] + calculate_offset(odometer, input_stride, index_map);
				int output_index = output_dim_offset * output_stride[concat_dim] + calculate_offset(odometer, output_stride, index_map);

				output[output_index] = t.tensor_node->data[input_index];
				
				//increase the write position after every one
				output_dim_offset++;
			}

		}
		next_exists = odometer_next_mapped(odometer, output_shape, index_map);
	}
	
	// now make the output node
	std::vector<std::shared_ptr<TensorNode>> predecessors;
	for (auto& t : array) predecessors.push_back(t.tensor_node);

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, output_shape, predecessors, concat_backward);
	node->concat_dim = concat_dim;	

	return Tensor(node);

}


void concat_backward(TensorNode* node) {
	
	// we KNOW The  predecessors
	// just recreate it....
	
	int dim = node->dim();
	int concat_dim = node->concat_dim;
	
	std::vector<int> odometer;
	std::vector<int> index_map;
	for (int i=0; i<dim; i++) {
		if (i == concat_dim) continue;
		odometer.push_back(0);
		index_map.push_back(i);
	}
	
	std::vector<int> output_shape = node->shape;
	std::vector<int> output_stride = node->stride;

	bool next_exists = true;
	while (next_exists) {
		// now iterate through it
		int output_dim_offset = 0;
		for (const auto& t: node->predecessors) {
			int concat_dim_size = t->shape[concat_dim];
			std::vector<int> input_stride = t->stride;
			
			for (int i=0; i<concat_dim_size; i++) {
					
				// find the input offset
				
				int input_index = i * input_stride[concat_dim] + calculate_offset(odometer, input_stride, index_map);
				int output_index = output_dim_offset * output_stride[concat_dim] + calculate_offset(odometer, output_stride, index_map);

				// write back in the proper value
				t->grad[input_index] += node->grad[output_index];

				//increase the write position after every one
				output_dim_offset++;
			}

		}
		next_exists = odometer_next_mapped(odometer, output_shape, index_map);
	}


}

// uhh basically just add the 2D matrix to it...
// just create another tensor and add it
// and we basically get if for free type....
// we dont even need a backward actually!
Tensor fixed_sincos_pos_embed(const Tensor& t, int embed_dim) {
	
	std::vector<int> t_shape = t.shape();
	int tdim = t.dim();
	int N = t_shape[tdim-2];
	int M = t_shape[tdim-1];
	std::vector<int> pos_embed_shape = {N, M};

	// now make the tensor node and write in the mask into it
	std::vector<double> posembed_mask(N*M);
	
	// so PE_even = sin(pos/(10000^(2i/d)))
	// where d is the embedding dimension of the block
	// i is the position in the last vector of it 
	// pos is the token index...
	// so when we iterate over N (the rows) this will be the pos
	
	for (int pos=0; pos<N; pos++) {
		for (int i=0; i<M; i++) {
			double exponent = (2 * (i / 2)) / static_cast<double>(embed_dim);
			double inner = pos / std::pow(10000, exponent);

			// basically because stride = [M, 1];
			int index = pos * M + i;
			
			if (i % 2 == 0) posembed_mask[index] = sin(inner);
			else posembed_mask[index] = cos(inner);
			
		}
	}

	// create the tensor node here we not gonna do
	// backprop on this so just leave predecessors and backward fn to 
	// null
	std::shared_ptr<TensorNode> posembed_node = make_operator_output_node(
		posembed_mask, {N, M}, {}, nullptr);
	
	// wrap this in the tensor so we can do the add
	Tensor posembed_tensor(posembed_node);
	
	// now use the broadcast add we just iplemented
	Tensor out = t + posembed_tensor;
	return out;

}


// just a pointwise so we gucci
Tensor Tanh(const Tensor& t) {
	
	int N = t.tensor_node->data.size();
	std::vector<double> output(N);

	for (int i=0; i<N; i++) {
		output[i] = std::tanh(t.tensor_node->data[i]);
	}
	
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, t.shape(), t.stride(), {t.tensor_node}, Tanh_backward);
	
	return Tensor(node);

}

// d/dx tanh(x) = 1 - tanh^2(x)

void Tanh_backward(TensorNode* node) {
	
	TensorNode* X = node->predecessors[0].get();

	for (int i=0; i<node->data.size(); i++) {
		double ddx = 1 - (node->data[i] * node->data[i]);
		X->grad[i] += node->grad[i] * ddx; 
	}

}
