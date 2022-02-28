#include <torch/torch.h>

#include "env_util.h"

void log_reg_main()
{
	bool cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "GPU in use." : "CPU in use") << '\n';

	//Hyper params
	constexpr int64_t input_size = 784;
	constexpr int64_t num_classes = 10;
	constexpr int64_t batch_size = 100;
	constexpr size_t num_epochs = 5;
	constexpr double learning_rate = 0.001;


}