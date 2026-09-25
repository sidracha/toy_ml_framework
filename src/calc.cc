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
		for (int j=0; j<CM; j++) {
			

			double dot_product = 0;
			for (int k=0; k<AM; k++) {
				
				// the row of A depends on i
				// the col of A depends on k

				// the col of B depends on j
				// the row of B depends on k
				
				// compute indexes directly so dont have to 
				// allocate a bunch of intermediete vectroins doing the fucntino call
				index_A = i * stride_A[0] + k * stride_A[1];
				index_B = k * stride_B[0] + j * stride_B[1];
				
				dot_product += data_A[index_A] * data_B[index_B];

			}
			
			index_C = i * stride_C[0] + j * stride_C[1];
			data_C[index_C] += dot_product;

		}
	}

}


// input:
// A: input data size [N*M]
// B: bias size [M]
// C: buffer size [N*M]
// it ADDS so make sure the buffer is written to 0
void MAT2D_1D_ADD (
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
	
	
	if (shape_C != shape_A || stride_C != stride_A) throw std::runtime_error("Invalid shapes for buffer");

	int AN = shape_A[0];
	int AM = shape_A[1];
	
	int BM = shape_B[0];
	
	int CN = shape_C[0];
	int CM = shape_C[1];
	
	if (CM != BM) throw std::runtime_error("Bias size mismatch");

	// iterate over the output postitions
	for (int i=0; i<CN; i++) {
		for (int j=0; j<CM; j++) {
			// then we want to for each one just add the bias of j its no big deal
			

			// acain calculate indexes directly

			int index_A = i * stride_A[0] + j * stride_A[1];
			int index_B = j * stride_B[0];
			int index_C = i * stride_C[0] + j * stride_C[1];
			
			//write the data into C
			data_C[index_C] += data_A[index_A] + data_B[index_B];

		}
	}

}
