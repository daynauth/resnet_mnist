#include <iostream>
#include <torch/torch.h>
#include <ATen/Context.h>
#include <chrono>
#include "Resnet.h"

int main(){
    auto resnet = Resnet18(10);

    torch::Device device(torch::kCPU);

	if(torch::cuda::is_available()){
		device = torch::kCUDA;
	}

    resnet->_conv1 = resnet->replace_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3).bias(false)
    ));


    torch::load(resnet, "resnet18_mnist.pt"); 
    resnet->to(device);

	auto data_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest).map(
			torch::data::transforms::Stack<>()), 1
	);

    resnet->eval();
    resnet->zero_grad();

    auto my_batch = data_loader->begin();
    auto my_data = my_batch->data.to(device);
    auto my_target = my_batch->target.to(device);
    auto my_pred = resnet->forward(my_data);

    int count = 0;
    float correct = 0;

    auto start = std::chrono::high_resolution_clock::now();

    
    for(auto & batch : *data_loader){
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);  

        torch::Tensor prediction = resnet->forward(data);   
        prediction -= prediction.max();
        auto sum = prediction.exp().sum();
        auto softmax = prediction.exp()/sum;

        if(targets.equal(softmax.argmax().flatten())){
            correct++;
        }

        count++;
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    if(count != 0){
        std::cout << "Total Time: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Accuracy: " << correct/count * 100 << "%" << std::endl;
    }

    return EXIT_SUCCESS;
}