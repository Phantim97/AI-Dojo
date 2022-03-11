#pragma once

#include <torch/torch.h>

class BiRNN : public torch::nn::Module
{
private:
	torch::nn::LSTM lstm;
	torch::nn::Linear fc;
public:
	BiRNN(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes);
	torch::Tensor forward(torch::Tensor x);
};