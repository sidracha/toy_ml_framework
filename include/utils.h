#pragma once

#include <vector>

int odometer_next(
	std::vector<int>& odometer,
	const std::vector<int>& shape,
	const std::vector<int>& stride);

bool odometer_next_mapped(
	std::vector<int>& odometer,
	std::vector<int>& shape,
	std::vector<int>& index_map);

int calculate_offset(
	std::vector<int>& odometer,
	std::vector<int>& stride,
	std::vector<int>& index_map);

int calculate_offset_broadcast(
	std::vector<int>& odometer,
	std::vector<int>& stride,
	std::vector<int>& shape);
