#pragma once

#include <torch/torch.h>

class RnnNet : public torch::nn::Module
{
private:
	torch::nn::LSTM lstm {nullptr};
	torch::nn::Linear fc {nullptr};

public:
	RnnNet(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes);
	torch::Tensor forward(torch::Tensor x);
};