#include "utils.h"

int odometer_next(
	std::vector<int>& odometer,
	const std::vector<int>& shape,
	const std::vector<int>& stride) {

	int n = odometer.size();
	if (n == 0) return -1;
	int carry = 1;
	int index = 0;

	for (int i = n-1; i >= 0; i--) {
		int sum = odometer[i] + carry;
		int remainder = sum % shape[i];
		carry = sum / shape[i];
		odometer[i] = remainder;
		index += (odometer[i] * stride[i]);
	}

	if (carry > 0) return -1;
	return index;
}

bool odometer_next_mapped(
	std::vector<int>& odometer,
	std::vector<int>& shape,
	std::vector<int>& index_map) {

	if (odometer.size() == 0) return false;

	int carry = 0;
	bool first = false;
	for (int i = odometer.size()-1; i >= 0; i--) {
		if (!first) {
			odometer[i]++;
			first = true;
		}
		odometer[i] += carry;
		int shape_index = index_map[i];
		int dig = (odometer[i] % shape[shape_index]);
		carry = (odometer[i] / shape[shape_index]);
		odometer[i] = dig;
		if (carry == 0) break;
	}
	if (carry > 0) return false;
	return true;
}

int calculate_offset(
	std::vector<int>& odometer,
	std::vector<int>& stride,
	std::vector<int>& index_map) {

	int ret_index = 0;
	for (int i = 0; i < odometer.size(); i++) {
		int stride_index = index_map[i];
		ret_index += odometer[i] * stride[stride_index];
	}
	return ret_index;
}

int calculate_offset_broadcast(
	std::vector<int>& odometer,
	std::vector<int>& stride,
	std::vector<int>& shape) {

	int ret_index = 0;
	for (int i = 0; i < odometer.size(); i++) {
		int pos = (shape[i] == 1) ? 0 : odometer[i];
		ret_index += pos * stride[i];
	}
	return ret_index;
}
