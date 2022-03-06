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

	const std::string mnist_path = util::get_dataset_path() + "mnist/";

	const auto train_dataset = torch::data::datasets::MNIST(mnist_path)
	                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
	                           .map(torch::data::transforms::Stack<>());

	size_t num_train_samples = train_dataset.size().value();

	const auto test_dataset = torch::data::datasets::MNIST(mnist_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	size_t num_test_samples = test_dataset.size().value();

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(train_dataset, batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(test_dataset, batch_size);

	torch::nn::Linear model(input_size, num_classes);

	model->to(device);

	//Loss and optimizer
	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

	std::cout << std::fixed << std::setprecision(4);

	//Train
	for (size_t epoch = 0; epoch != num_epochs; epoch++)
	{
		//Running metrics
		double running_loss = 0.0;
		size_t num_correct = 0;

		for (torch::data::Example<>& batch : *train_loader)
		{
			torch::Tensor data = batch.data.view({ batch_size, -1 }).to(device);

			torch::Tensor target = batch.target.to(device);

			//Forward pass
			torch::Tensor output = model->forward(data);

			//Calc loss
			torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

			running_loss += loss.item<double>() * data.size(0);

			//Calc prediction
			torch::Tensor prediction = output.argmax(1);

			//Update num correct
			num_correct += prediction.eq(target).sum().item<int64_t>();

			//Backward pass and optimize
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		double sample_mean_loss = running_loss / num_train_samples;
		double accuracy = num_correct / num_train_samples();

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << '\n';
	}

	std::cout << "Training Complete.\nTesting...\n";

	//Test model
	model->eval();
	torch::NoGradGuard no_grad; //Guard for gradient

	double running_loss = 0;
	size_t num_correct = 0;

	for (const torch::data::Example<>& batch : *test_loader)
	{
		torch::Tensor data = batch.data.view({ batch_size, -1 }).to(device);
		torch::Tensor target = batch.target.to(device);

		torch::Tensor output = model->forward(data);

		torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

		running_loss += loss.item<double>() * data.size(0);
		torch::Tensor prediction = output.argmax(1);

		num_correct += prediction.eq(target).sum().item<int64_t>();
	}

	std::cout << "Testing finished\n";

	const double test_accuracy = num_correct / num_test_samples;
	const double test_sample_mean_loss = running_loss / num_test_samples;

	std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}