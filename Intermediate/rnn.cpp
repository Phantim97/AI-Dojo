#include "rnn.h"

RnnNet::RnnNet(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes)
{
	lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)));
	fc = register_module("fc", torch::nn::Linear(hidden_size, num_classes));
}

torch::Tensor RnnNet::forward(torch::Tensor x)
{
	const torch::Tensor out = std::get<0>(lstm->forward(x)).index({ torch::indexing::Slice(), -1 });
	return fc->forward(out);
}