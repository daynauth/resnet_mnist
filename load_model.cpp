#include <torch/torch.h>
#include <ATen/Context.h>
#include <iostream>
#include <memory>
#include <assert.h>
#include "NvInfer.h"


class Logger : public nvinfer1::ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;



struct Net : torch::nn::Module{
	Net(){
		fc1 = register_module("fc1", torch::nn::Linear(784, 64));
		fc2 = register_module("fc2", torch::nn::Linear(64, 32));
		fc3 = register_module("fc3", torch::nn::Linear(32, 10));
	}

	torch::Tensor forward(torch::Tensor x){
		x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
		x = torch::dropout(x, 0.5, is_training());
		x = torch::relu(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), 1);

		return x;
	}

	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};


//got this one from nvidia, change this in the future
//see here https://en.cppreference.com/w/cpp/memory/unique_ptr
//and here https://marcoarena.wordpress.com/2014/04/12/ponder-the-use-of-unique_ptr-to-enforce-the-rule-of-zero/
struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};


//TODO use IHostMemory class for this later :)
std::vector<std::shared_ptr<float[]>> ptr_memory;


nvinfer1::Weights load_weights_from_tensors(at::Tensor &tensor){
	std::cout << "Tensor size: " << tensor.sizes() << std::endl;
	int dims = tensor.sizes().size();

	std::cout << "dims: " << dims << std::endl;

	//create weight variable
	nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};



	if(dims == 1){
		int size = tensor.sizes()[0];
		wt.count = size;
		auto tensor_f = tensor.accessor<float, 1>();

		auto val = std::shared_ptr<float[]>(new float[size]);
		ptr_memory.emplace_back(val);

		for(int i = 0; i < size; i++){
			val[i] = tensor_f[i];
		}

		wt.values = val.get();

		//std::cout << ((float *)wt.values)[0] << std::endl;

	}
	else{
		int out_size = tensor.sizes()[0];
		int in_size = tensor.sizes()[1];
		wt.count = in_size * out_size;

		//auto tensor_f = tensor.packed_accessor64<float, 2>();
		//assume that the tensor is a CPU tensor for now
		auto tensor_f = tensor.accessor<float, 2>();



		
		//shared pointer array 
		auto val = std::shared_ptr<float[]>(new float[out_size * in_size]);
		ptr_memory.emplace_back(val);
		
		for(int i = 0; i < out_size; i++){
			for(int j = 0; j < in_size; j++){
				val[i * in_size + j] = tensor_f[i][j];
			}
		}

		wt.values = val.get();

		//std::cout << ((float *)wt.values)[0] << std::endl;
	}




	return wt;
}


int main(){
    auto net = std::make_shared<Net>();
    torch::load(net, "net.pt");
	

	net->eval();


    std::cout << *net << std::endl;
	std::vector<at::Tensor> tensors;


	for(auto m : net->modules()){
		std::cout << m->name() << std::endl;
		//std::cout << "Keys " << std::endl;
		//std::cout << m->named_parameters().keys() << std::endl;
		//auto weights = m->named_parameters()["weight"];
		//std::cout << weights.sizes() << std::endl;
		
	}






    for (auto layers : net->children()){
		//std::cout << *layers << std::endl;
		
		auto parameters = layers->named_parameters();
		
		for(auto param: parameters.keys()){
			tensors.push_back(parameters[param]);
		}

    }

	
/*

	//lets create a simple network for tensorRT
	auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(gLogger));
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
	auto data = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, 28, 28});

	auto fc1_weights = load_weights_from_tensors(tensors[0]);
	auto fc1_bias = load_weights_from_tensors(tensors[1]);

	auto fc2_weights = load_weights_from_tensors(tensors[2]);
	auto fc2_bias = load_weights_from_tensors(tensors[3]);

	auto fc3_weights = load_weights_from_tensors(tensors[4]);
	auto fc3_bias = load_weights_from_tensors(tensors[5]);

	//first linear layer
	auto ip1 = network->addFullyConnected(*data, 64, fc1_weights, fc1_bias);

	//relu
	auto relu1 = network->addActivation(*ip1->getOutput(0), nvinfer1::ActivationType::kRELU);

	//dropout layer not needed
	//second linear layer
	auto ip2 = network->addFullyConnected(*relu1->getOutput(0), 32, fc2_weights, fc2_bias);
	auto relu2 = network->addActivation(*ip2->getOutput(0), nvinfer1::ActivationType::kRELU);

	//find linear layer
	auto ip3 = network->addFullyConnected(*relu2->getOutput(0), 10, fc3_weights, fc3_bias);
	std::cout << "debug " << std::endl;
	auto softmax = network->addSoftMax(*ip3->getOutput(0));
	std::cout << "debug " << std::endl;

	//std::cout << ((float *)fc1_weights.values)[0] << std::endl;
	//std::cout << ((float *)fc1_bias.values)[0] << std::endl;

*/
	return 1;
	
}