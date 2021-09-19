// Ref : https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div
#include <iostream>
#include <cmath>
#include <cutf/memory.hpp>

constexpr unsigned block_size = 1u << 8;

// Div operations
struct approx_div;
struct full_range_approx_div;
struct ieee_div;

template <class DivOp>
__device__ float div(const float a, const float b);

template <>
__device__ float div<approx_div>(const float a, const float b) {
	float r;
	asm(
			R"(
{
div.approx.f32 %0, %1, %2;
}
)": "=f"(r) : "f"(a), "f"(b)
			);
	return r;
}

template <>
__device__ float div<full_range_approx_div>(const float a, const float b) {
	float r;
	asm(
			R"(
{
div.full.f32 %0, %1, %2;
}
)": "=f"(r) : "f"(a), "f"(b)
			);
	return r;
}

template <>
__device__ float div<ieee_div>(const float a, const float b) {
	float r;
	asm(
			R"(
{
div.rn.f32 %0, %1, %2;
}
)": "=f"(r) : "f"(a), "f"(b)
			);
	return r;
}

template <class DivOp>
__global__ void div_kernel(
		double* const r_ptr,
		const float* const b_ptr,
		const unsigned array_length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= array_length) return;

	const auto a = 1. / 3;
	const auto b = b_ptr[tid];

	const auto dp_result = a / b;
	const auto sp_result = div<DivOp>(a, b);

	r_ptr[tid] = abs(dp_result - sp_result) / dp_result;
}

void test(
		const int exp_min,
		const int exp_max,
		const unsigned num_mantissa_split
		) {
	const auto array_length = (exp_max - exp_min + 1) * num_mantissa_split;
	auto b_array = cutf::memory::get_host_unique_ptr<float>(array_length);

	for (int e = exp_min; e <= exp_max; e++) {
		for (unsigned i = 0; i < num_mantissa_split; i++) {
			const auto mantissa_dp = static_cast<double>(1) * i / num_mantissa_split + 1;
			b_array.get()[(e - exp_min) * num_mantissa_split + i] = mantissa_dp * std::pow(2., static_cast<double>(e));
		}
	}

	auto approx_result = cutf::memory::get_host_unique_ptr<double>(array_length);
	div_kernel<approx_div><<<(array_length + block_size - 1) / block_size, block_size>>>(
			approx_result.get(),
			b_array.get(),
			array_length
			);
	auto full_range_approx_result = cutf::memory::get_host_unique_ptr<double>(array_length);
	div_kernel<full_range_approx_div><<<(array_length + block_size - 1) / block_size, block_size>>>(
			full_range_approx_result.get(),
			b_array.get(),
			array_length
			);
	auto ieee_result = cutf::memory::get_host_unique_ptr<double>(array_length);
	div_kernel<ieee_div><<<(array_length + block_size - 1) / block_size, block_size>>>(
			ieee_result.get(),
			b_array.get(),
			array_length
			);
	cudaDeviceSynchronize();

	std::printf("b,approx,full_range_approx,ieee\n");
	for (int e = exp_min; e <= exp_max; e++) {
		for (unsigned i = 0; i < num_mantissa_split; i++) {
			const auto array_index = (e - exp_min) * num_mantissa_split + i;
			std::printf("%e,%e,%e,%e\n",
					b_array.get()[array_index],
					approx_result.get()[array_index],
					full_range_approx_result.get()[array_index],
					ieee_result.get()[array_index]
					);
		}
	}
}

int main() {
	test(-126, 126, 10);
}
