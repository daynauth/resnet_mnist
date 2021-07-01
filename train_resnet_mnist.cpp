#include <iostream>
#include <torch/torch.h>
#include <ATen/Context.h>

#include "Resnet.h"

int main(){
    const int64_t kNumberOfEpochs = 10;
    torch::Device device(torch::kCPU);

	if(torch::cuda::is_available()){
		device = torch::kCUDA;
	}
    
    auto resnet = Resnet18(10);
    resnet->_conv1 = resnet->replace_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3).bias(false)
    ));
    
    resnet->to(device);


    auto loss_function = torch::nn::CrossEntropyLoss();

	//create a multi-threaded data loader for the MNIST dataset.
	auto data_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST("../data").map(
			torch::data::transforms::Stack<>()), 64
	);

    torch::optim::SGD optimizer(resnet->parameters(), 0.01);

	for(size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch){
		size_t batch_index = 0;

        //set the model to train
        resnet->train();

		//iterate the data loader to yield batches from the dataset
		for(auto& batch : *data_loader){
			auto data = batch.data.to(device);
			auto targets = batch.target.to(device);

            //reset gradients
			optimizer.zero_grad();
			//execute the mode on the input data
			torch::Tensor prediction = resnet->forward(data);

			//compute a loss value to judge the prediction of our model
            torch::Tensor loss = loss_function->forward(prediction, targets);

			//compute gadients of the loss w.r.t the parameters of our model.
			loss.backward();
			//update the parameters based on the caluated gradients
			optimizer.step();

			if(++batch_index % 100 == 0){
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
						  << " | Loss: " << loss.item<float>() << std::endl;
				torch::save(resnet, "resnet18_mnist.pt");
			}
        }
    }    

    return 0;
}