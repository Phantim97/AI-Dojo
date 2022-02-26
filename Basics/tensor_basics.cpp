#include <torch/torch.h>
#include <torch/script.h>

void tensor_gradients1()
{
	std::cout << std::fixed << std::setprecision(4);

	//Gradient Calculation

	//By having torch::requires_grad() we make these tensors act as graph leaves
	const torch::Tensor x = torch::tensor(1.0, torch::requires_grad()); //Note torch::tensor vs torch::Tensor
	const torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
	const torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

	//Build Computational Graph
	const torch::Tensor y = w * x + b;

	y.backward(); //Computes the gradients of all graph leaves in this graph (y)

	//.grad() is a printout of the gradient
	std::cout << "X Gradient: " << x.grad() << '\n';
	std::cout << "W Gradient: " << w.grad() << '\n';
	std::cout << "B Gradient: " << b.grad() << '\n';
}

void tensor_gradients2()
{
	//Create 2D tensors
	torch::Tensor x = torch::randn({ 10, 3 });
	torch::Tensor y = torch::randn({10, 2});

	//Fully connected layer
	torch::nn::Linear linear(3, 2); //3x2 linear
	std::cout << "W: " << linear->weight << '\n'; //3x2
	std::cout << "B: " << linear->bias << "\n\n"; //Bias is 2x1

	//Loss function and optimizer
	torch::nn::MSELoss criteria;
	torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.001)); //learning rate is option param

	//Forward pass our input prediction (x)
	torch::Tensor prediction = linear->forward(x);

	//Calculate loss against output (y)
	torch::Tensor loss = criteria(prediction, y);
	std::cout << "Loss: " << loss.item<double>() << "\n\n";

	loss.backward();

	//Gradients:
	std::cout << "dL / dw: " << linear->weight.grad() << '\n';
	std::cout << "dL / db: " << linear->bias.grad() << "\n\n";

	//1 step of gradient descent
	optimizer.step();

	prediction = linear->forward(x);
	loss = criteria(prediction, y);
	std::cout << "Loss after one step of gd: " << loss.item<double>() << '\n';
}

void tensor_existing_data()
{
	//torch::from_blob(ptr_to_data, ...) does not allow the tensor stored to own the data
	//to fix this do from_blob(...).clone()

	//tensor from float array
	float data_array[] = { 1.0f, 2.0f, 3.0f, 4.0f };
	const torch::Tensor t = torch::from_blob(data_array, { 2, 2 }); //clone for ownership, second param determines the shape
	std::cout << "Tensor from array: " << t << '\n';

	//This check only works if you did not clone (they point to the same place)
	TORCH_CHECK(data_array == t.data_ptr<float>())

	//Tensor from vector
	std::vector<float> v = { 1.0f, 2.0f, 3.0f, 4.0f };
	const torch::Tensor t2 = torch::from_blob(v.data(), {2, 2});
	std::cout << "Tensor from vector: " << t2 <<"\n\n";

	TORCH_CHECK(v.data() == t2.data_ptr<float>())

	//Tensor from vector
	const torch::Tensor t3 = torch::from_blob(v.data(), { 2, 2 }).clone();
	std::cout << "Tensor from vector cloned: " << t3 << '\n';
	std::cout << "Checking if v.data != t3.data_ptr<float>()\n";

	TORCH_CHECK(v.data() != t3.data_ptr<float>())

	std::cout << "Check shows these two do not point to the same place due to .clone()";
}

namespace tensor_ops
{
	typedef torch::indexing::Slice Slice;
	using torch::indexing::Ellipsis;
	int None = static_cast<int>(torch::indexing::TensorIndexType::None);
};

void tensor_slicing()
{
	std::vector<int> test_data = { 1,2,3,4,5,6,7,8,9 };
	torch::Tensor t = torch::from_blob(test_data.data(), { 3, 3 }, torch::kInt64);

	std::cout << "Base tensor: " << t << '\n';

	// Slice Tensor along a dimension at a given index
	std::cout << "t[:,2]" << t.index({ tensor_ops::Slice(), 2 }) << '\n';
	/* 3
	 * 6
	 * 9
	 */

	// Slice a tensor along a dimension at given indices from
	// a start-index up to - but not including - an end-index using a given step size.
	std::cout << "\"s[:2,:]\":\n" << t.index({ tensor_ops::Slice(tensor_ops::None , 2), tensor_ops::Slice()}) << '\n';
	// Output:
	// 1 2 3
	// 4 5 6
	std::cout << "\"s[:,1:]\":\n" << t.index({ tensor_ops::Slice(), tensor_ops::Slice(1, tensor_ops::None) }) << '\n';
	// Output:
	// 2 3
	// 5 6
	// 8 9
	std::cout << "\"t[:,::2]\":\n" << t.index({ tensor_ops::Slice(), tensor_ops::Slice(tensor_ops::None, tensor_ops::None, 2) }) << '\n';
	// Output:
	// 1 3
	// 4 6
	// 7 9

	// Combination.
	std::cout << "\"t[:2,1]\":\n" << t.index({ tensor_ops::Slice(tensor_ops::None, 2), 1 }) << '\n';
	// Output:
	// 2
	// 5

	 // Ellipsis (...).
	std::cout << "\"t[..., :2]\":\n" << t.index({tensor_ops::Ellipsis, tensor_ops::Slice(tensor_ops::None, 2)}) << "\n\n";
	// Output:
	// 1 2
	// 4 5
	// 7 8
}

static void mnist_loader_example()
{
	// Construct MNIST dataset
	const std::string mnist_data_path = "../../../../data/mnist/";

	torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<
		                                  torch::data::datasets::MNIST, torch::data::transforms::Normalize<>>,
	                                  torch::data::transforms::Stack<>> dataset = torch::data::datasets::MNIST(
			mnist_data_path)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	// Fetch one data pair
	const torch::data::datasets::detail::optional_if_t<
		false, torch::data::datasets::MapDataset<
			torch::data::datasets::MapDataset<torch::data::datasets::MNIST, torch::data::transforms::Normalize<>>,
			torch::data::transforms::Stack<>>::OutputBatchType> example = dataset.get_batch(0);
	std::cout << "Sample data size: ";
	std::cout << example.data.sizes() << "\n";
	std::cout << "Sample target: " << example.target.item<int>() << "\n";

	// Construct data loader
	const std::unique_ptr<torch::data::StatelessDataLoader<
			torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<
				                                  torch::data::datasets::MNIST, torch::data::transforms::Normalize<>>,
			                                  torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>>
		dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			dataset, 64);

	// Fetch a mini-batch
	const torch::data::Example<> example_batch = *dataloader->begin();
	std::cout << "Sample batch - data size: ";
	std::cout << example_batch.data.sizes() << "\n";
	std::cout << "Sample batch - target size: ";
	std::cout << example_batch.target.sizes() << "\n\n";
}

static void print_script_module(const torch::jit::script::Module& module, const size_t spaces)
{
	for (const torch::jit::Named<torch::jit::Module>& sub_module : module.named_children())
	{
		if (!sub_module.name.empty())
		{
			std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name()
				<< " " << sub_module.name << "\n";
		}

		print_script_module(sub_module.value, spaces + 2);
	}
}

//Sample function
void pretrained_model()
{
	// Loading a pretrained model using the C++ API is done
	// in the following way:
	// In Python:
	// (1) Create the (pretrained) pytorch model.
	// (2) Convert the pytorch model to a torch.jit.ScriptModule (via tracing or by using annotations)
	// (3) Serialize the scriptmodule to a file.
	// In C++:
	// (4) Load the scriptmodule form the file using torch::jit::load()
	// See https://pytorch.org/tutorials/advanced/cpp_export.html for more infos.

	// Path to serialized ScriptModule of pretrained resnet18 model,
	// created in Python.
	// You can use the provided Python-script "create_resnet18_scriptmodule.py" in
	// tutorials/basics/pytorch-basics/model to create the necessary file.

	const std::string pretrained_model_path = "../../../../tutorials/basics/pytorch_basics/model/"
		"resnet18_scriptmodule.pt";

	torch::jit::script::Module resnet;

	try
	{
		resnet = torch::jit::load(pretrained_model_path);
	}
	catch (const torch::Error& error) 
	{
		std::cerr << "Could not load scriptmodule from file " << pretrained_model_path << ".\n"
			<< "You can create this file using the provided Python script 'create_resnet18_scriptmodule.py' "
			"in tutorials/basics/pytorch-basics/model/." << error.what() <<"\n";
		return;
	}

	std::cout << "Resnet18 model:\n";

	print_script_module(resnet, 2);

	std::cout << "\n";

	const auto fc_weight = resnet.attr("fc").toModule().attr("weight").toTensor();

	const int in_features = fc_weight.size(1);
	const int out_features = fc_weight.size(0);

	std::cout << "Fully connected layer: in_features=" << in_features << ", out_features=" << out_features << "\n";

	// Input sample
	const torch::Tensor sample_input = torch::randn({ 1, 3, 224, 224 });
	const std::vector<torch::jit::IValue> inputs{ sample_input };

	// Forward pass
	std::cout << "Input size: ";
	std::cout << sample_input.sizes() << "\n";
	const torch::Tensor output = resnet.forward(inputs).toTensor();
	std::cout << "Output size: ";
	std::cout << output.sizes() << "\n\n";

	// =============================================================== //
	//                      SAVE AND LOAD A MODEL                      //
	// =============================================================== //

	// Simple example model
	torch::nn::Sequential model
	{
		//ConvTranspose2dOptions allows us to use padding
		torch::nn::Conv2d(torch::nn::ConvTranspose2dOptions(1, 16, 3).stride(2).padding(1)),
		torch::nn::ReLU()
	};

	// Path to the model output file (all folders must exist!).
	const std::string model_save_path = "output/model.pt";

	// Save the model
	torch::save(model, model_save_path);

	std::cout << "Saved model:\n" << model << "\n";

	// Load the model
	torch::load(model, model_save_path);

	std::cout << "Loaded model:\n" << model;
}