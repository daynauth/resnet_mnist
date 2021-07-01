#include <iostream>
#include <torch/torch.h>
#include <ATen/Context.h>
#include <vector>

#include "NvInfer.h"
#include "Resnet.h"
#include "utils.h"

std::vector<nvinfer1::Weights> trt_weights;


class Logger : public nvinfer1::ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;


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


nvinfer1::IConvolutionLayer* add_conv2d(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::Module>  _module, nvinfer1::ITensor *previous,
const std::string name = "conv_layer"
){
    if(_module->name() != "torch::nn::Conv2dImpl"){
        std::cout << "This needs to be a conv layer" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto conv_layer = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(_module);

    std::cout << conv_layer->weight.sizes() << std::endl;

    auto featuremap_size = conv_layer->weight.sizes()[0];
    int kernel_height = conv_layer->weight.sizes()[2];
    int kernel_width = conv_layer->weight.sizes()[3];
    auto conv_weights = conv_layer->weight.flatten();
    auto wt_size =  conv_weights.sizes()[0];


    if(wt_size != (featuremap_size * kernel_height * kernel_height * conv_layer->weight.sizes()[1])){
        std::cout << "size doesn't match" << std::endl;
        exit(EXIT_FAILURE);
    }


    auto stride1 = conv_layer->options.stride()->at(0);
    auto stride2 = conv_layer->options.stride()->at(0);
    auto padding1 = conv_layer->options.padding()->at(0);
    auto padding2 = conv_layer->options.padding()->at(1);
    auto groups = conv_layer->options.groups();
    auto dilation1 = conv_layer->options.dilation()->at(0);
    auto dilation2 = conv_layer->options.dilation()->at(1);

    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    trt_weights.push_back(bias);

    float *weights = (float *)malloc(sizeof(float) * wt_size);
    mem_check(weights)

    tensor_accessor(conv_weights, weights);

    wt.values = weights;
    wt.count = wt_size;
    trt_weights.push_back(wt);




    auto conv = network->addConvolutionNd(*previous, featuremap_size, nvinfer1::DimsHW{kernel_height, kernel_width},  wt, bias);
    conv->setName(name.c_str());
    conv->setStrideNd(nvinfer1::DimsHW(stride1, stride2));
    conv->setPaddingNd(nvinfer1::DimsHW(padding1, padding2));
    conv->setNbGroups(groups);
    conv->setDilationNd(nvinfer1::DimsHW(dilation1, dilation2));

    return conv;
}


nvinfer1::ILayer * addFullyConnected(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::Module>  _module, nvinfer1::ITensor *previous, const std::string name = "full_connected"){
    if(_module->name() != "torch::nn::LinearImpl"){
        std::cout << "This needs to be a fully connected layer" << std::endl;
        exit(EXIT_FAILURE);
    }    

    auto fully_connected = std::dynamic_pointer_cast<torch::nn::LinearImpl>(_module);

    int outputs = 10;
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights bias_t{nvinfer1::DataType::kFLOAT, nullptr, 0};


    auto fc_weights = fully_connected->weight.flatten();
    auto fc_weight_size = fc_weights.sizes()[0];

    float * weights = (float *)malloc(sizeof(float) * fc_weight_size);
    mem_check(weights);

    tensor_accessor(fc_weights, weights);
    wt.count = fc_weight_size;
    wt.values = weights;
    trt_weights.push_back(wt);


    auto fc_bias = fully_connected->bias.flatten();
    auto fc_bias_size = fc_bias.sizes()[0];

    float * bias = (float *)malloc(sizeof(float) * fc_bias_size);
    mem_check(bias)

    tensor_accessor(fc_bias, bias);
    bias_t.count = fc_bias_size;
    bias_t.values = bias;
    trt_weights.push_back(bias_t);

    auto fc = network->addFullyConnected(*previous, outputs, wt, bias_t);
    return fc;
}

nvinfer1::IScaleLayer* add_batchnorm(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::Module> _module, nvinfer1::ITensor *previous, const std::string name = "BatchNorm"){
    if(_module->name() != "torch::nn::BatchNorm2dImpl"){
        std::cout << "This needs to be a batchnorm layer" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(_module);
    auto mean = bn->running_mean;
    auto var = bn->running_var;
    auto gamma = bn->weight;
    auto beta= bn->bias;
    auto eps = 1.0e-5;
    

    auto size = gamma.sizes()[0];
    if(gamma.sizes().size() != 1){
        std::cout << "Tensor should have a dim of 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, size};
    

    //calculate scale
    var += eps;
    var = var.sqrt();
    
    auto scaleT = gamma/var;

    //convert torch tensor to tensorrt tensor
    float *scaleWt = (float *)malloc(sizeof(float) * size);
    mem_check(scaleWt)

    tensor_accessor(scaleT, scaleWt);
    scale.values = scaleWt;
    trt_weights.push_back(scale);

    //calculate shift
    auto shiftT =  beta - ((mean * gamma) / var);
    float *shiftWt = (float *)malloc(sizeof(float) * size);
    mem_check(shiftWt)
    
    tensor_accessor(shiftT, shiftWt);
    shift.values = shiftWt;
    trt_weights.push_back(shift);


    //power needs to be one
    float *powerWt = (float *)malloc(sizeof(float) * size);
    mem_check(powerWt)

    for(int i = 0; i < size; i++) powerWt[i] = 1.0;
    
    power.values = powerWt;
    trt_weights.push_back(power);

    auto bnn = network->addScale(*previous, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    return bnn;
}

nvinfer1::IPoolingLayer * add_max_pool(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::MaxPool2dImpl> _maxpool, nvinfer1::ITensor *previous, const std::string name = "maxpool"){
    auto height = _maxpool->options.kernel_size()->at(0);
    auto width = _maxpool->options.kernel_size()->at(1);

    auto stride1 = _maxpool->options.stride()->at(0);
    auto stride2 = _maxpool->options.stride()->at(1);

    auto padding1 = _maxpool->options.padding()->at(0);
    auto padding2 = _maxpool->options.padding()->at(1);

    auto maxpool = network->addPoolingNd(*previous, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(height, width));
    maxpool->setStrideNd(nvinfer1::DimsHW(stride1, stride2));
    maxpool->setPaddingNd(nvinfer1::DimsHW(padding1, padding2));
    maxpool->setName(name.c_str());

    return maxpool;
}

nvinfer1::IPoolingLayer * add_ave_pool(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::AvgPool2dImpl> _avgpool, nvinfer1::ITensor *previous){
    auto kernel = _avgpool->options.kernel_size()->at(0);
    auto stride = _avgpool->options.stride()->at(0);

    auto avgpool =  network->addPoolingNd(*previous, nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW(kernel, kernel));
    
    avgpool->setStrideNd(nvinfer1::DimsHW(stride, stride));
    avgpool->setName("Average_Pool");
    return avgpool;
}


nvinfer1::ITensor * build_downsample(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::Module> _sequential, nvinfer1::ITensor *output, const std::string name = "downsample"){
    if(_sequential->children().size() != 2){
        std::cout << "Incorrect downsample layer" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto conv_layer = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(_sequential->children()[0]);
    
    auto conv = add_conv2d(network, _sequential->children()[0], output, name + "_conv1");
    output = conv->getOutput(0);

    auto bn = add_batchnorm(network, _sequential->children()[1], output, name + "_bn1");
    output = bn->getOutput(0);

    return output;
}

nvinfer1::ITensor * build_block(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, std::shared_ptr<torch::nn::Module> Block, nvinfer1::ITensor *previous, const std::string name = "block"){
    if(Block->name() != "BasicBlock"){
        std::cout << "Either basic block of bottleneck block required" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto identity = previous;

    auto child = Block->children()[0];
    auto output0 = add_conv2d(network, child, previous, name + "_conv1");
    previous = output0->getOutput(0);

    child = Block->children()[1];
    auto output1 = add_batchnorm(network, child, previous, name + "_bn1");
    previous = output1->getOutput(0);

    child = Block->children()[2];
    auto output2 = network->addActivation(*previous, nvinfer1::ActivationType::kRELU);
    output2->setName((name + "_relu1").c_str());
    previous = output2->getOutput(0);

    child = Block->children()[3];
    auto output3 = add_conv2d(network, child, previous, name + "_conv2");
    previous = output3->getOutput(0);

    child = Block->children()[4];
    auto output4 = add_batchnorm(network, child, previous, name + "_bn2");
    previous = output4->getOutput(0);

    if(Block->children().size() == 6){
        child = Block->children()[5];
        identity = build_downsample(network, child, identity, name + "_ds1"); 
    }

    //handle the skip connection
    auto skip = network->addElementWise(*identity, *previous, nvinfer1::ElementWiseOperation::kSUM);
    previous = skip->getOutput(0);

    auto output = network->addActivation(*previous, nvinfer1::ActivationType::kRELU);
    previous = output->getOutput(0);
    
    return previous;
}


nvinfer1::ITensor * build_layer(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> &network, torch::nn::Sequential _sequential, nvinfer1::ITensor *previous, const std::string name = "layer"){
    int i = 0;
    for(auto block : _sequential->children()){
        previous = build_block(network, block, previous, name + "_block" + std::to_string(++i));
    }

    return previous;
}

int main(){
    auto resnet = Resnet18(10);

    resnet->_conv1 = resnet->replace_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3).bias(false)
    ));

    //load the pretrained libtorch file    
    torch::load(resnet, "resnet18_mnist.pt");

    //we will always assume that we're using resnet18 for now.
	auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(gLogger));
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(builder->createNetworkV2(0U));

    //add the input layer
    auto data = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW{1, 28, 28});

    //conv1 layer
    auto conv1 = add_conv2d(network, resnet->_conv1, data, "conv1");
    
    //bn1
    auto bn1 = add_batchnorm(network, resnet->_bn1, conv1->getOutput(0), "bn1");

    //relu
    auto relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    relu1->setName("relu1");

    //max pool
    auto maxpool = add_max_pool(network, resnet->_maxpool, relu1->getOutput(0), "maxpool1");

    //deal with layer 1
    auto output = build_layer(network, resnet->_layer1, maxpool->getOutput(0), "layer1");

    output = build_layer(network, resnet->_layer2, output, "layer2");
    output = build_layer(network, resnet->_layer3, output, "layer3");
    output = build_layer(network, resnet->_layer4, output, "layer4");

    auto avgpool = add_ave_pool(network, resnet->_avgpool, output);

    auto fc = addFullyConnected(network, resnet->_fc, avgpool->getOutput(0));
    fc->setName("fullyconnected");


    fc->getOutput(0)->setName("output");
    network->markOutput(*fc->getOutput(0));


    //single batch size for now
    builder->setMaxBatchSize(1);

    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);

    //set floating point 16
    if(builder->platformHasFastFp16()){
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }


    std::cout << "building engine" << std::endl;

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter>(builder->buildEngineWithConfig(*network, *config));
    //auto engine = builder->buildEngineWithConfig(*network, *config);

    if(!engine){
        std::cout << "engine could not be built" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "building complete" << std::endl;

    std::cout << "Serializing engine" << std::endl;
    auto serializedModel = engine->serialize();

    if(!serializedModel){
        std::cout << "Model could not be serialized" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::stringstream outstream;
    outstream.seekg(0, outstream.beg);
    outstream.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());

    std::ofstream outfile;
    outfile.open("resnet_fp16.rt", std::ios::binary | std::ios::out);
    outfile << outstream.rdbuf();
    outfile.close();


    if(builder->platformHasTf32() == true){
        std::cout << "TF 32 supported" << std::endl;
    }

    for(auto wt : trt_weights){
        if(wt.count > 0)
            free(const_cast<void *>(wt.values));
    }



    return 0;
}