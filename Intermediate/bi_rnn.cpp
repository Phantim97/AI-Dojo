#include "bi_rnn.h"

BiRNN::BiRNN(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes)
{
	lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
		.num_layers(num_layers)
		.batch_first(true)
		.bidirectional(true)
	));

	fc = register_module("fc", torch::nn::Linear(hidden_size * 2, num_classes));
}

torch::Tensor BiRNN::forward(torch::Tensor x)
{
	const torch::Tensor out = std::get<0>(lstm->forward(x)); // out: tensor of shape (batch_size, sequence length, 2 * hidden_size)

	const std::vector<torch::Tensor> out_direction = out.chunk(2, 2);

	//Last hidden state of fwd direction
	const torch::Tensor out_1 = out_direction[0].index({ torch::indexing::Slice(), -1 }); //Tensor of shape batch_size x hidden_size

	//First hidden state of backward direction output
	const torch::Tensor out_2 = out_direction[1].index({ torch::indexing::Slice(), 0 }); //Tensor of shape batch_size x hidden_size

	const torch::Tensor out_cat = torch::cat({ out_1, out_2 }, 1); //out_cat: tensor of shape (batch_size, 2 * hidden_size)

	return fc->forward(out_cat);
}
