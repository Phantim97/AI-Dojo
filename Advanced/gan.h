#pragma once

#include <torch/torch.h>

class GAN
{
public:
	torch::nn::Sequential Generator{ nullptr };
	torch::nn::Sequential Discriminator{ nullptr };

	GAN(int64_t image_size, int64_t hidden_size, int64_t latent_size)
	{
		Generator = torch::nn::Sequential
		{
			torch::nn::Linear(latent_size, hidden_size),
			torch::nn::ReLU(),
			torch::nn::Linear(hidden_size, hidden_size),
			torch::nn::ReLU(),
			torch::nn::Linear(hidden_size, image_size),
			torch::nn::Tanh()
		};

		Discriminator = torch::nn::Sequential
		{
			torch::nn::Linear(image_size, hidden_size),
			torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
			torch::nn::Linear(hidden_size, hidden_size),
			torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2),
			torch::nn::Linear(hidden_size, 1)),
			torch::nn::Sigmoid()
		};
	}
};