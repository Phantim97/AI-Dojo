#pragma once

#include <torch/torch.h>

struct ConvNet : torch::nn::Module
{
	explicit ConvNet(int64_t num_classes = 10);
	torch::Tensor forward(torch::Tensor x);

	torch::nn::Sequential layer1
	{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
		torch::nn::BatchNorm2d(16),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
	};

	torch::nn::Sequential layer2
	{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
		torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
	};

	torch::nn::Sequential layer3
	{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(16,32,3).stride(1)),
		torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
	};

	torch::nn::AdaptiveAvgPool2d pool { torch::nn::AdaptiveAvgPool2dOptions({4,4}) };

	torch::nn::Linear fc {nullptr};
};