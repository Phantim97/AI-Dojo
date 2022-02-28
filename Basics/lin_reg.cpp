#include <torch/torch.h>

struct Options
{
	const int64_t input_size = 1;
	const int64_t output_size = 1;
	const size_t num_epochs = 60;
	const double learning_rate = 0.001;
	torch::Device device = torch::kCPU;
};

static Options options;

void lin_reg_main()
{
	if (torch::cuda::is_available())
	{
		options.device = torch::kCUDA;
	}

	//Sample dataset
	torch::Tensor x_train = torch::randint(0, 10, { 15, 1 });
	torch::Tensor y_train = torch::randint(0, 10, { 15, 1 });

	//CUDA Tensors
	x_train = x_train.to(options.device);
	y_train = y_train.to(options.device);

	//Linear Regression model
	torch::nn::Linear model(options.input_size, options.output_size);

	//Linear regression model to CUDA
	model->to(options.device);

	//Optimizer
	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(options.learning_rate));

	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";

	//Train the model
	for (size_t epoch = 0; epoch != options.num_epochs; epoch++)
	{
		//Forward pass
		torch::Tensor output = model(x_train);
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
}

int zero_op(const int x)
{
	return x + 5;
}

int one_op(const int x)
{
	return x - 1;
}

int two_op(const int x)
{
	return 2 * x;
}

int three_op(const int x)
{
	return 3 * x;
}

int four_op(const int x)
{
	return x * x;
}

int five_op(const int x)
{
	return x + 15;
}

int six_op(const int x)
{
	return 6 * x + 7;
}

int seven_op(const int x)
{
	return 4 * x + 3;
}

int eight_op(const int x)
{
	return 4 * x + 4;
}

int nine_op(const int x)
{
	return 5 * x - 4;
}

//Loss flatlines at 36.388 tried multiple methods
void proj1()
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
	torch::nn::Linear model(options.input_size, options.output_size);

	//Linear regression model to CUDA
	model->to(options.device);

	//Optimizer
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(5e-2));

	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training...\n";

	//Train the model
	for (size_t epoch = 0; epoch != options.num_epochs * 1000 + 40000; epoch++)
	{
		//Forward pass
		torch::Tensor output = model(x_train);
		torch::Tensor loss = torch::nn::functional::mse_loss(output, y_train);

		//Backward pass and optimize
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if ((epoch + 1) % 5 == 0)
		{
			std::cout << "Epoch [" << (epoch + 1) << '/' << options.num_epochs * 1000 + 40000<<
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

	model->eval();
	torch::NoGradGuard no_grad;

	std::cout << "Zero %: "  << model->forward(t[0].data()) << '\n';
	std::cout << "One %: "   << model->forward(t[1].data()) << '\n';
	std::cout << "Two %: "   << model->forward(t[2].data()) << '\n';
	std::cout << "Three %: " << model->forward(t[3].data()) << '\n';
	std::cout << "Four %: "  << model->forward(t[4].data()) << '\n';
	std::cout << "Five %: "  << model->forward(t[5].data()) << '\n';
	std::cout << "Six %: "   << model->forward(t[6].data()) << '\n';
	std::cout << "Seven %: " << model->forward(t[7].data()) << '\n';
	std::cout << "Eight %: " << model->forward(t[8].data()) << '\n';
	std::cout << "Nine  %: " << model->forward(t[9].data()) << '\n';
}