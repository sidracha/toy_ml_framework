#include <vector>
#include <stdexcept>

int linearize_index(const std::vector<int>& index, const std::vector<int>& stride) {
	int ret = 0;
	for (int i=0; i<stride.size(); i++) {
		ret += stride[i] * index[i];
	}
	return ret;
}

// GEMM kernal (Generalized MM)
// Computes C = AB
// Takes in the shapes and buffers of each one and calcualtes the rest
// THE BUFFER C HAS TO BE WRITTEN TO ZEROS, SINCE WE ADD
void GEMM_2D_ADD (
	const std::vector<double>& data_A,
	const std::vector<int>& stride_A,
	const std::vector<int>& shape_A,
	
	const std::vector<double>& data_B,
	const std::vector<int>& stride_B,
	const std::vector<int>& shape_B,
	
	std::vector<double>& data_C,
	const std::vector<int>& stride_C,
	const std::vector<int>& shape_C

	) {
	
	int AN = shape_A[0];
	int AM = shape_A[1];

	int BN = shape_B[0];
	int BM = shape_B[1];
	
	int CN = AN;
	int CM = BM;

	if (AM != BN) throw std::runtime_error("Invalid data shapes for GEMM");
	
	//iterate over the output positons...
	int index_A, index_B, index_C;
		
	for (int i=0; i<CN; i++) {
		for (int j=0; j<CM; i++) {
			

			double dot_product = 0;
			for (int k=0; k<AM; k++) {
				
				// the row of A depends on i
				// the col of A depends on k

				// the col of B depends on j
				// the row of B depends on k

				index_A = linearize_index({i, k}, stride_A);
				index_B = linearize_index({k, j}, stride_B);
				
				dot_product += data_A[index_A] * data_B[index_B];

			}
			
			index_C = linearize_index({i, j}, stride_C);
			data_C[index_C] += dot_product;

		}
	}

}
