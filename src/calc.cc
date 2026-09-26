#include <vector>
#include <stdexcept>
#include <format>

int linearize_index(const std::vector<int>& index, const std::vector<int>& stride) {
	int ret = 0;
	for (int i=0; i<stride.size(); i++) {
		ret += stride[i] * index[i];
	}
	return ret;
}

inline int calc_batch_size(const std::vector<int>& shape) {
	return (shape.size() == 3) ? shape[0] : 1;
}

int odometer_next(
		std::vector<int>& odometer, 
		const std::vector<int>& shape, 
		const std::vector<int>& stride) {
	

	// if ever it gets more than the actual shape, we want to modify the odometer
	// if the carry at the end we want to return -1
	
	int n = odometer.size();
	if (n == 0) return -1;
	int carry = 1;
	int index = 0;
	

	for (int i=n-1; i>=0; i--) {
		int sum = odometer[i] + carry;
		int remainder = sum % shape[i];
		carry = sum / shape[i];
		odometer[i] = remainder;
		index += (odometer[i] * stride[i]);
	}

	if (carry > 0) return -1;
	else return index;

}

// cpu GEMM kernel lol
// Computes C = AB
// Takes in the shapes and buffers of each one and calcualtes the rest
// THE BUFFER C HAS TO BE WRITTEN TO ZEROS, SINCE WE ADD

// we can probably take in 2D or 3D tensor
// iterate over the batch dimension of A
// and do MM with the B tensor, which is 2D for now


// ok we ALSO ASSUME THAT STRIDE A = STRIDE C
// and that B = dim 2 matrix
void BATCHED_GEMM_2D_ADD (
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
	
	// input and output should have the same ones, right?? yes they should
	// because it needs to be batched but B the one we are multiplying with
	// needs to be a dim=2 matrix for sure
	if (shape_A.size() != shape_C.size()) {
		throw std::runtime_error("dim mismatch in GEMM");
	}
	int dim = shape_A.size();
	
	// use the last 2 indexes as the N and M for the matmul but this is basically for A and C
	int N_index = dim-2;
	int M_index = dim-1;
	int AN = shape_A[N_index];
	int AM = shape_A[M_index];
	int BN = shape_B[0];
	int BM = shape_B[1];
	int CN = shape_C[N_index];
	int CM = shape_C[M_index];
	
	if (AM != BN) throw std::runtime_error(
		std::format("Shape mismatch ({} x {}) and ({} x {})", AN, AM, BN, BM)
	);
	if (CN != AN || CM != BM) {
		throw std::runtime_error("Shape mismatch for input and output buffer");
	}

	// ok so basically have an odometer vector
	std::vector<int> odometer_A;
	std::vector<int> odometer_C;
	for (int i=0; i<N_index; i++) {
		odometer_A.push_back(0);
		odometer_C.push_back(0);
	}

	// thennnn we can go to the next odometer sure....
	// first have the loop to iterate over the odometer
	int index_A_batch = 0;
	int index_C_batch = 0;

	// basically loops over all of the initial dimensions... then inside we want to properly go over
	// the back 2 dimensions N and M

	int index_A, index_B, index_C;
	while (index_A_batch >= 0) {

		// iterate over the output positions
		for (int i=0; i<CN; i++) {
			for (int j=0; j<CM; j++) {
				// iterate over the colums of A, which are the rows of B
				double dot_product = 0.0;
				for (int k=0; k<AM; k++) {

					// A = [i][k]
					// B = [k][j]
					// C = [i][j]
					// and we have to add up the stuff in k
					index_A = index_A_batch + i * stride_A[N_index] + k * stride_A[M_index];
					index_B = k * stride_B[0] + j * stride_B[1];

					dot_product += (data_A[index_A] * data_B[index_B]);
				}

				index_C = index_C_batch + i * stride_C[N_index] + j * stride_C[M_index];
				data_C[index_C] = dot_product;
			}


		}
		// we want to do next on what, the stride of A whcih
		// is also the stride of C.. and i think we are done
		index_A_batch = odometer_next(odometer_A, shape_A, stride_A);
		index_C_batch = odometer_next(odometer_C, shape_C, stride_C);

	}

}
	
// input:
// A: input data size [N*M]
// B: bias size [M]
// C: buffer size [N*M]
// it ADDS so make sure the buffer is written to 0


void BATCHED_MAT2D_1D_ADD (
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

	// if the shapes of C and A are not the same we cant do this
	if (shape_A != shape_C || stride_A != stride_C) {
		throw std::runtime_error("Input and output shapes do not match");
	}

	// now the same thing,, we want to iterate
	// over all the odometer positions of A 
	// but also C
	
	// eh fuck it lets make the top layer enfoce the checks cuz we basically 
	// also have to check there, right? 
	
	int index_N = shape_A.size()-2;
	int index_M = index_N + 1;

	int AN = shape_A[index_N];
	int AM = shape_A[index_M];
	int BM = shape_B[0];
	int CN = shape_C[index_N];
	int CM = shape_C[index_M];
	
	std::vector<int> odometer;
	for (int i=0; i<index_N; i++) odometer.push_back(0);
	int index_odometer = 0;
	

	// iterate over the first dimensions
	// then iterate over the output positions i guess
	// to EACH ROW WE ADD THE BIAS
	
	int index_A, index_B, index_C;
	while (index_odometer >= 0) {
		
		for (int i=0; i<CN; i++) {
			for (int j=0; j<CM; j++) {
					
				// sooo which ones get the offset? probably A and C
				index_A = index_odometer + i * stride_A[index_N] + j * stride_A[index_M];
				index_B = j * stride_B[0];
				index_C = index_A;
				
				data_C[index_C] += (data_A[index_A] + data_B[index_B]);
			}
		}

		//incremenet the odometer for each first dim
		index_odometer = odometer_next(odometer, shape_A, stride_A);
	}

}

