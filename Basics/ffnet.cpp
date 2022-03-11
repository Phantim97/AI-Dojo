#include <torch/torch.h>
#include "env_util.h"

class NeuralNet : public torch::nn::Module
{
private:
	torch::nn::Linear fc1 { nullptr };
	torch::nn::Linear fc2 { nullptr };

public:
	NeuralNet(int64_t input_size, int64_t hidden_size, int64_t num_classes)
	{
		register_module("fc1", torch::nn::Linear(input_size, hidden_size));
		register_module("fc2", torch::nn::Linear(hidden_size, num_classes));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::nn::functional::relu(fc1->forward(x));
		return fc2->forward(x);
	}
};

void ffnet()
{
	bool cuda_available = torch::cuda::is_available();

	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available" : "CPU Only") << '\n';

	//Hyper params
	constexpr int64_t input_size = 784;
	constexpr int64_t hidden_size = 500;
	constexpr int64_t num_classes = 10;
	constexpr int64_t batch_size = 100;
	constexpr size_t num_epochs = 5;
	constexpr double learning_rate = 0.001;

	const std::string mnist_data_path = util::get_dataset_path() + "mnist";

	const auto train_dataset = torch::data::datasets::MNIST(mnist_data_path)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	const size_t num_train_samples = train_dataset.size().value();

	const auto test_dataset = torch::data::datasets::MNIST(mnist_data_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	const size_t num_test_sampels = test_dataset.size().value();

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(train_dataset, batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(test_dataset, batch_size);

	//NN Model
	NeuralNet model(input_size, hidden_size, num_classes);

	model.to(device);

	torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";

	for (size_t epoch = 0; epoch != num_epochs; epoch++)
	{
		double running_loss = 0.0;
		size_t num_correct = 0;

		for (torch::data::Example<>& batch : *train_loader)
		{
			const torch::Tensor data = batch.data.view({ batch_size, -1 }).to(device);
			const torch::Tensor target = batch.target.to(device);

			//Forward pass
			const torch::Tensor output = model.forward(data);
			const torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

			//Running loss update
			running_loss += loss.item<double>() * data.size(0);

			//Caluclate prediction
			const torch::Tensor prediction = output.argmax(1);

			//Update correct count
			num_correct += prediction.eq(target).sum().item<int64_t>();

			//Backprop and optimize
			optimizer.zero_grad();
			loss.backward();
		}

		const double sample_mean_loss = running_loss / num_train_samples;
		const double accuracy = num_correct / num_train_samples;

		std::cout << "Epoch[" << (epoch + 1) << "/" << num_epochs << "], Trainset loss: " << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
	}

	std::cout << "Training finished\n\n";
	std::cout << "Testing...\n";

	model.eval();

	torch::NoGradGuard no_grad;

	double running_loss = 0.0;
	size_t num_correct = 0;

	for (torch::data::Example<>& batch : *test_loader)
	{
		const torch::Tensor data = batch.data.view({ batch_size, -1 }).to(device);
		const torch::Tensor target = batch.target.to(device);

		const torch::Tensor output = model.forward(data);

		const torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

		running_loss += loss.item<double>() * data.size(0);

		const torch::Tensor prediction = output.argmax(1);

		num_correct += prediction.eq(target).sum().item<int64_t>();
	}

	std::cout << "Testing finished\n";

	const double test_accuracy = num_correct / num_test_sampels;
	const double test_sample_mean_loss = running_loss / num_test_sampels;

	std::cout << "Testnet - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}

struct Options
{
	const int64_t input_size = 1;
	const int64_t output_size = 1;
	const size_t num_epochs = 20000;
	const double learning_rate = 0.001;
	torch::Device device = torch::kCPU;
};

static int zero_op(const int x)
{
	return x + 5;
}

static int one_op(const int x)
{
	return x - 1;
}

static int two_op(const int x)
{
	return 2 * x;
}

static int three_op(const int x)
{
	return 3 * x;
}

static int four_op(const int x)
{
	return x * x;
}

static int five_op(const int x)
{
	return x + 15;
}

static int six_op(const int x)
{
	return 6 * x + 7;
}

static int seven_op(const int x)
{
	return 4 * x + 3;
}

static int eight_op(const int x)
{
	return 4 * x + 4;
}

static int nine_op(const int x)
{
	return 5 * x - 4;
}

static Options options;

struct Net final : torch::nn::Module
{
	torch::nn::Linear fc1 { nullptr };
	torch::nn::Linear fc2 { nullptr };
	torch::nn::Linear fc3 { nullptr };
	torch::nn::Linear fc4 { nullptr };
	torch::nn::Linear fc5 { nullptr };
	torch::nn::Linear fc6 { nullptr };

	//Can use torch::nn::LinearOptions to set the input and output of each linear layer
	Net() 
	{
		fc1 = register_module("fc1", torch::nn::Linear(1, 10));
		fc2 = register_module("fc2", torch::nn::Linear(10, 20));
		fc3 = register_module("fc3", torch::nn::Linear(20, 50));
		fc4 = register_module("fc4", torch::nn::Linear(50, 20));
		fc5 = register_module("fc5", torch::nn::Linear(20, 10));
		fc6 = register_module("fc6", torch::nn::Linear(10, 1));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::tanh(fc1->forward(x));
		x = torch::tanh(fc2->forward(x));
		x = torch::tanh(fc3->forward(x));
		x = fc4->forward(x);
		x = fc5->forward(x);
		return fc6->forward(x);
	}
};

void proj()
{
	if (torch::cuda::is_available())
	{
		options.device = torch::kCUDA;
	}

	//Sample dataset
	torch::Tensor x_train = torch::randint(0, 9, { 100000, 1 });
	torch::Tensor y_train = x_train.clone();

	//CUDA Tensors
	x_train = x_train.to(options.device);
	y_train = y_train.to(options.device);

	const std::vector<std::function<int(int)>> ops{ zero_op, one_op, two_op, three_op, four_op, five_op, six_op, seven_op, eight_op, nine_op };

	for (int i = 0; i < y_train.sizes()[0]; i++)
	{
		y_train[i] = ops[x_train[i].item<int>() % 10](x_train[i].item<int>());
	}

	//Random Data point check
	std::cout << x_train[100] << " and " << y_train[100] << '\n';
	std::cout << x_train[200] << " and " << y_train[200] << '\n';
	std::cout << x_train[300] << " and " << y_train[300] << '\n';
	std::cout << x_train[400] << " and " << y_train[400] << '\n';
	std::cout << x_train[500] << " and " << y_train[500] << '\n';

	//Linear Regression model
	//torch::nn::Linear model(options.input_size, options.output_size);

	//Note can also utilize:
	//const std::shared_ptr<Net> net = std::shared<Net>();
	Net model;

	//Linear regression model to CUDA
	model.to(options.device);

	std::cout << "Optimizer\n";

	//Optimizer
	torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-4));

	//std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";

	//Train the model
	for (size_t epoch = 0; epoch != options.num_epochs; epoch++)
	{
		//Forward pass
		torch::Tensor output = model.forward(x_train);
		torch::Tensor loss = torch::nn::functional::mse_loss(output, y_train);
		//Backward pass and optimize
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if ((epoch + 1) % 5 == 0)
		{
			std::cout << "Epoch [" << (epoch + 1) << '/' << options.num_epochs <<
				"], Loss: " << loss.item<double>() << '\n';
		}
	}

	std::cout << "Training finished\n";
	int val = 0;

	std::vector<torch::Tensor> t;
	for (int i = 0; i < 10; i++)
	{
		torch::Tensor temp = torch::zeros({ 1,1 }, torch::kFloat32);
		temp[0] = i;
		t.emplace_back(temp.clone().to(options.device));
	}

	for (int i = 0; i < t.size(); i++)
	{
		std::cout << t[i] << '\n';
	}

	model.eval();
	torch::NoGradGuard no_grad;

	std::cout << "Zero %: " << model.forward(t[0].data()) << '\n';
	std::cout << "One %: " << model.forward(t[1].data()) << '\n';
	std::cout << "Two %: " << model.forward(t[2].data()) << '\n';
	std::cout << "Three %: " << model.forward(t[3].data()) << '\n';
	std::cout << "Four %: " << model.forward(t[4].data()) << '\n';
	std::cout << "Five %: " << model.forward(t[5].data()) << '\n';
	std::cout << "Six %: " << model.forward(t[6].data()) << '\n';
	std::cout << "Seven %: " << model.forward(t[7].data()) << '\n';
	std::cout << "Eight %: " << model.forward(t[8].data()) << '\n';
	std::cout << "Nine  %: " << model.forward(t[9].data()) << '\n';
}