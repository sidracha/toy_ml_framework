#include <vector>

int odometer_next(std::vector<int>& odometer, const std::vector<int>& shape, const std::vector<int>& stride);

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
);


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
);
