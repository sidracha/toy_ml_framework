#include "modules.h"
#include "tensor.h"
#include "utils.h"
#include <unordered_set>

void layer_norm_backward(TensorNode* Y);



// soo 
// y = gamme + beta * (x - E[x]) / sqrt(var(x) + eps) 
// eq gotten from the pytorch docs lul
Tensor LayerNorm::forward(Tensor t) {
	// we want to calculate the variance and nem over the
	// entire dimensions that are mentioned...

	int dim = t.dim();
	std::vector<int> odometer(dim - normalize_dims.size(), 0);
	std::vector<int> stride = t.stride();
	std::vector<int> shape = t.shape();

	std::vector<int> gamma_stride = gamma.stride();
	std::vector<int> gamma_index_map;
	for (int i=0; i<normalize_dims.size(); i++) gamma_index_map.push_back(i);

	std::unordered_set<int> index_find(normalize_dims.begin(), normalize_dims.end());
	std::vector<int> index_map_odometer;
	for (int i=0; i<dim; i++) {
		if (index_find.find(i) == index_find.end()) index_map_odometer.push_back(i);
	}
	
	// create the output node here first...
	std::vector<double> output(t.tensor_node->data.size());

	int next_exists = true;
	while (next_exists) {
	
		std::vector<int> normalize_odometer(normalize_dims.size(), 0);
		bool normalize_next_exists = true;
	

		// calculate mean
		double mean = 0.0;
		int N = 0;
		while (normalize_next_exists) {
			int global_offset = calculate_offset(odometer, stride, index_map_odometer);
			int normalize_offset = calculate_offset(normalize_odometer, stride, normalize_dims);
			int index = global_offset + normalize_offset;

			// use the running average formula
			N++;
			mean = mean + (t.tensor_node->data[index] - mean) / N;
			normalize_next_exists = odometer_next_mapped(normalize_odometer, shape, normalize_dims);
		}		
	
		for (int i=0; i<normalize_odometer.size(); i++) normalize_odometer[i] = 0;
		normalize_next_exists = true;

		// calculate variance
		double variance = 0;
		int VN = 0;
		while (normalize_next_exists) {
			int global_offset = calculate_offset(odometer, stride, index_map_odometer);
			int normalize_offset = calculate_offset(normalize_odometer, stride, normalize_dims);
			int index = global_offset + normalize_offset;
			double diff = (t.tensor_node->data[index] - mean);
			VN++;
			variance = variance + (diff*diff - variance) / VN;
			normalize_next_exists = odometer_next_mapped(normalize_odometer, shape, normalize_dims);
		}		
		
		for (int i=0; i<normalize_odometer.size(); i++) normalize_odometer[i] = 0;
		normalize_next_exists = true;

		while (normalize_next_exists) {
			// we want to pass in the same stride
			// great! calculate the global offset and the index offset
			int global_offset = calculate_offset(odometer, stride, index_map_odometer);
			int normalize_offset = calculate_offset(normalize_odometer, stride, normalize_dims);
			int gamma_offset = calculate_offset(normalize_odometer, gamma_stride, gamma_index_map);
			int index = global_offset + normalize_offset;
			double x = t.tensor_node->data[index];
			output[index] = ((x-mean) / std::sqrt(variance + eps)) * gamma.data_at(gamma_offset) + beta.data_at(gamma_offset);

			normalize_next_exists = odometer_next_mapped(normalize_odometer, shape, normalize_dims);
		}
		next_exists = odometer_next_mapped(odometer, shape, index_map_odometer);

	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
			output, shape, stride, {t.tensor_node, gamma.tensor_node, beta.tensor_node}, layer_norm_backward);
	node->normalize_dims = normalize_dims;
	node->eps = eps;
	
	return Tensor(node);
	
}


//holyy fuckkkk this took me like 5 pages to derive
//to find the o(n) solution
// but bascially
// let hi = gi * gammai
// let a = sqrt(v + eps)
// where gi is the upstream incoming gradient
// so dL/dxi = hi/a - sum_j(hj)/(Na) - (xi-miu)*sum_j(hj*(xj-miu))/Na^3
// so term 1 = sum_j(hj) 
// so term 2 = sum_j(hj*(xj-miu))
// ok jesus
// so bascially we can precompute the summartions of terms 2 and 3
// and we can per i multiply by the constsnt hi/a and whatevers before the term 3 summation
// ok jesus
//
// then dL/dgammai = (xi-mean)/a * gi pretty simple there since multiplication
// hmm we can probably just do the tensor mult and get it for free....
// ugh but we havent implemented reduce dims yet into the mult.. which 
// we should probbaly do but for now we can just do pointwise 
// and then dL/dbeta = gi its just addition
// ok lets get implementing!
void layer_norm_backward(TensorNode* Y) {

	TensorNode* X = Y->predecessors[0].get();
	TensorNode* gamma = Y->predecessors[1].get();
	TensorNode* beta = Y->predecessors[2].get();

	std::vector<int> normalize_dims = Y->normalize_dims;
	double eps = Y->eps;

	int dim = Y->dim();
	std::vector<int> odometer(dim - normalize_dims.size(), 0);
	std::vector<int> stride = Y->stride;
	std::vector<int> shape = Y->shape;

	std::vector<int> gamma_stride = gamma->stride;
	std::vector<int> gamma_index_map;
	for (int i=0; i<normalize_dims.size(); i++) gamma_index_map.push_back(i);

	std::unordered_set<int> index_find(normalize_dims.begin(), normalize_dims.end());
	std::vector<int> index_map_odometer;
	for (int i=0; i<dim; i++) {
		if (index_find.find(i) == index_find.end()) index_map_odometer.push_back(i);
	}
	
	// create the output node here first...
	int next_exists = true;
	while (next_exists) {

		std::vector<int> normalize_odometer(normalize_dims.size(), 0);
		bool normalize_next_exists = true;
		
		// ok here calcualte the constant stuff over all j
		double term1 = 0.0;
		double term2 = 0.0;
		int N = 0;
		double mean = 0.0;
		while (normalize_next_exists) {

			int global_offset = calculate_offset(odometer, stride, index_map_odometer);
			int normalize_offset = calculate_offset(normalize_odometer, stride, normalize_dims);
			int gamma_offset = calculate_offset(normalize_odometer, gamma_stride, gamma_index_map);
			int index = global_offset + normalize_offset;

			// use the running average formula
			N++;
			mean = mean + (X->data[index] - mean) / N;
			double h = Y->grad[index] * gamma->data[gamma_offset];
			term1 += h;

			normalize_next_exists = odometer_next_mapped(normalize_odometer, shape, normalize_dims);
		}
		
		for (int i=0; i<normalize_odometer.size(); i++) normalize_odometer[i] = 0;
		normalize_next_exists = true;
		
		double variance = 0;
		term1 = 0.0;
		term2 = 0.0;
		int VN = 0;
		while (normalize_next_exists) {

			int global_offset = calculate_offset(odometer, stride, index_map_odometer);
			int normalize_offset = calculate_offset(normalize_odometer, stride, normalize_dims);
			int gamma_offset = calculate_offset(normalize_odometer, gamma_stride, gamma_index_map);
			int index = global_offset + normalize_offset;

			double diff = (X->data[index] - mean);
			VN++;
			variance = variance + (diff*diff - variance) / VN;

			double h = Y->grad[index] * gamma->data[gamma_offset];
			term1 += h;
			term2 += h * (X->data[index] - mean);
			normalize_next_exists = odometer_next_mapped(normalize_odometer, shape, normalize_dims);
		}

		for (int i=0; i<normalize_odometer.size(); i++) normalize_odometer[i] = 0;
		normalize_next_exists = true;
		
		while (normalize_next_exists) {

			int global_offset = calculate_offset(odometer, stride, index_map_odometer);
			int normalize_offset = calculate_offset(normalize_odometer, stride, normalize_dims);
			int gamma_offset = calculate_offset(normalize_odometer, gamma_stride, gamma_index_map);
			int index = global_offset + normalize_offset;

			// we have calculated the 2 terms now we can put it together
			// so dL/dxi = hi/a - sum_j(hj)/(Na) - (xi-miu)*sum_j(hj*(xj-miu))/Na^3
			// so term 1 = sum_j(hj)
			// so term 2 = sum_j(hj*(xj-miu))
			//
			// dL/dxi = hi/a - term1/(Na) - (xi-miu)/Na^3 * term2
			// a = sqrt(variance + eps)

			double a = std::sqrt(variance + eps);
			
			double h = Y->grad[index] * gamma->data[gamma_offset];
			double dLdxi = (h/a) - (term1/(N*a)) - ((X->data[index]-mean)/(N*a*a*a)) * term2;

			// now we calculate the gamma and the beta
			double dgamma = ((X->data[index] - mean) / a) * Y->grad[index];
			double dbeta = Y->grad[index];

			X->grad[index] += dLdxi;
			gamma->grad[gamma_offset] += dgamma;
			beta->grad[gamma_offset] += dbeta;
			normalize_next_exists = odometer_next_mapped(normalize_odometer, shape, normalize_dims);

		}

		next_exists = odometer_next_mapped(odometer, shape, index_map_odometer);
	}

}
