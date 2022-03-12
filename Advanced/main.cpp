#include "gan.h"
#include "env_util.h"

static torch::Tensor denorm(torch::Tensor tensor)
{
	return tensor.add(1).div_(2).clamp(0, 1);
};

void gan_main()
{
	const bool cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "GPU in use" : "CPU in use") << '\n';

	//Hyper params
	const int64_t latent_size = 64;
	const int64_t hidden_size = 256;
	const int64_t image_size = 28 * 28;
	const int64_t batch_size = 100;
	const size_t num_epochs = 200;
	const double learning_rate = 0.0002;

	const std::string mnist_path = util::get_dataset_path() + "mnist/";

	const std::string sample_output_dir_path = "output/";

	//Mnist
	auto dataset = torch::data::datasets::MNIST(mnist_path)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	unsigned long long num_samples = dataset.size().value();

	//Data loader
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, batch_size);

	GAN gan(image_size, hidden_size, latent_size);

	gan.Discriminator->to(device);
	gan.Generator->to(device);

	//Optimizers
	torch::optim::Adam d_optimizer(gan.Discriminator->parameters(), torch::optim::AdamOptions(learning_rate));
	torch::optim::Adam g_optimizer(gan.Generator->parameters(), torch::optim::AdamOptions(learning_rate));

	std::cout << std::fixed << std::setprecision(4);

	//Training
	std::cout << "Training...\n";

	for (size_t epoch = 0; epoch != num_epochs; epoch++)
	{
		torch::Tensor images;
		torch::Tensor fake_images;
		size_t batch_index = 0;

		for (torch::data::Example<>& batch : *dataloader)
		{
			images = batch.data.reshape({ batch_size, -1 }).to(device);

			torch::Tensor real_labels = torch::ones({ batch_size, 1 }).to(device);
			torch::Tensor fake_labels = torch::zeros({ batch_size, 1 }).to(device);

			//discrim training

			torch::Tensor outputs = gan.Discriminator->forward(images);
			torch::Tensor d_loss_real = torch::nn::functional::cross_entropy(outputs, real_labels);
			double real_score = outputs.mean().item<double>();

			torch::Tensor z = torch::randn({ batch_size, latent_size }).to(device);
			fake_images = gan.Generator->forward(z);
			outputs = gan.Discriminator->forward(fake_images);
			torch::Tensor d_loss_fake = torch::nn::functional::binary_cross_entropy(outputs, fake_labels);
			double fake_score = outputs.mean().item<double>();

			torch::Tensor d_loss = d_loss_real - d_loss_fake;

			//Backward pass and optimize
			d_optimizer.zero_grad();
			d_loss.backward();
			d_optimizer.step();

			//Train the generator

			//Compute loss with fake images
			z = torch::randn({ batch_size, latent_size }).to(device);
			fake_images = gan.Generator->forward(z);
			outputs = gan.Discriminator->forward(fake_images);

			//Generator trained to maximize log(D(G(z)) instead of minimizing log(1 - D(G(z))
			torch::Tensor g_loss = torch::nn::functional::binary_cross_entropy(outputs, real_labels);

			//Backwards pass and optimize
			g_optimizer.zero_grad();
			g_loss.backward();
			g_optimizer.step();

			if ((batch_index + 1) % 200 == 0)
			{
				std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Step [" << batch_index + 1 << "/" << num_samples / batch_size
					<< "] d_loss: " << d_loss.item<double>() << ", g_loss: " << g_loss.item<double>()
					<< ", D(x): " << real_score << ", D(G(z)): " << fake_score << "\n";
			}

			batch_index++;
		}

		//Save real images once
		if (epoch == 0)
		{
			images = denorm(images.reshape({ images.size(0), 1, 28, 28 }));
			//save_image(images, sample_output_dir_path, "real_images.png");
		}

		//Save generated fake images
		fake_images = denorm(fake_images.reshape({ fake_images.size(0), 1, 28, 28 }));
		//save_image(fake_images, sample_output_dir_path + "fake-images-" + std::to_string(epoch + 1) + ".png");
	}

	std::cout << "Training finished\n";
}

int main()
{

	return 0;
}