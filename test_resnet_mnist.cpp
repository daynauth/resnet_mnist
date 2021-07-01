#include <iostream>
#include <iterator>
#include <algorithm>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <vector>
#include <math.h>
#include <chrono>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "utils.h"

class Logger : public nvinfer1::ILogger           
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;


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


typedef unsigned char uchar;

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readInt(std::ifstream &stream, int &number){
    stream.read((char *)&number, sizeof(number));
    number = reverseInt(number);
}


std::vector<std::shared_ptr<uchar[]>> load_mnist_images(const std::string file_path, int &number_of_images, int &image_size){
    std::ifstream input(file_path, std::ios::binary);

    if(input.is_open()){
        int magic_number = 0;
        readInt(input, magic_number);

        if(magic_number != 2051){
            std::cout << "Incorrect file!" << std::endl;
            exit(EXIT_FAILURE);
        }

        readInt(input, number_of_images);

        std::vector<std::shared_ptr<uchar[]>> dataset;

        int rows = 0;
        readInt(input, rows);

        int columns = 0;
        readInt(input, columns);
        image_size = rows * columns;

        for(int i = 0; i < number_of_images; i++){
            auto image = std::shared_ptr<uchar[]>(new uchar[image_size]);
            input.read((char *)image.get(), image_size);
            dataset.push_back(image);
        }

        return dataset;
    }
    else{
        std::cout << "Could not open file" << std::endl;
        exit(EXIT_FAILURE);
    }
}


std::unique_ptr<uchar[]> load_mnist_labels(const std::string file_path, int & number_of_labels){
    std::ifstream input(file_path, std::ios::binary);


    if(!input){
        std::cout << "Cannot Open File" << std::endl;
        exit(EXIT_FAILURE);        
    }

    if(input.is_open()){
        int magic_number = 0;
        readInt(input, magic_number);

        if(magic_number != 2049){
            std::cout << "Incorrect file!" << std::endl;
            exit(EXIT_FAILURE);
        }

        readInt(input, number_of_labels);

        auto labels = std::unique_ptr<uchar[]>(new uchar[number_of_labels]);

        for(int i = 0; i < number_of_labels; i++){
            input.read((char *)&labels.get()[i], 1);
        }

        return labels;
    }
    else{
        std::cout << "Cannot Open File" << std::endl;
        exit(EXIT_FAILURE);
    }
}


std::unique_ptr<float[]> softmax_cpu(float * output, size_t number_of_classes){
    auto max = std::max_element(output, output + number_of_classes);

    auto shifted_output = std::unique_ptr<float[]>(new float[number_of_classes]);
    std::transform(output, output + number_of_classes, shifted_output.get(), [&](auto a){
        return a - *max;
    });

    auto sum = std::accumulate(shifted_output.get(), shifted_output.get() + number_of_classes, 0.0, [](auto a, auto b){
        return a + exp(b);
    });
    
    auto softmax_output = std::unique_ptr<float []>(new float[number_of_classes]);
    std::transform(shifted_output.get(), shifted_output.get() + number_of_classes, softmax_output.get(), [&](auto a){
        return exp(a)/sum;
    });

    return softmax_output;
}



class ResnetTRT{
public:
    ResnetTRT(const std::string name){
        if(file_exist(name)){
            load_serialized_file(name);
            buildNetwork();
        }
        else{
            std::cout << "file cannot be found" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    bool file_exist(const std::string name){
        std::ifstream file(name.c_str());
        return file.good();
    }

    void load_serialized_file(const std::string name)
    {
        std::ifstream file(name, std::ios::binary);
        file.seekg(0, file.end);
        fileSize = file.tellg();
        file.seekg(0, file.beg);
        modelStream = new char[fileSize];
        file.read(modelStream, fileSize);
        file.close();
    }

    void buildNetwork(){
        //create infer runtime
        runtime = std::unique_ptr<nvinfer1::IRuntime, InferDeleter>(nvinfer1::createInferRuntime(gLogger));


        //deserialize the file into the TensorRT engine
        engine = std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter>(runtime->deserializeCudaEngine(modelStream, fileSize, nullptr));

        //Create some space to store intermediate activation values. Since the engine holds the network definition and trained parameters, 
        //additional space is necessary. These are held in an execution context
        context = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter>(engine->createExecutionContext());

        //get the binding for input
        auto input_idx = engine->getBindingIndex("input");

        if(engine->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT){
            std::cout << "Datatype mismatch" << std::endl;
            exit(EXIT_FAILURE);
        }

        auto dim = engine->getBindingDimensions(input_idx);
        auto C = dim.d[0];
        auto H = dim.d[1];
        auto W = dim.d[2]; 
        image_size = C * H * W;

        auto input_dims = nvinfer1::DimsCHW{C, H, W};

        //set context bindings. apparently you can have multiple context
        context->setBindingDimensions(input_idx, input_dims);

        //handle the output bindings
        auto output_idx = engine->getBindingIndex("output");
        checkType(engine->getBindingDataType(output_idx), nvinfer1::DataType::kFLOAT);

        //create input buffer
        cuda_check(cudaMalloc(&input_mem, sizeof(float) * C * H * W));

        //create output buffer
        size_t output_size = sizeof(float) * number_of_classes;
        cuda_check(cudaMalloc(&output_mem, sizeof(float) * number_of_classes));

        //all buffers
        buffers[0] = input_mem;
        buffers[1] = output_mem;

        cuda_check(cudaStreamCreate(&stream));
 
    }

    void cleanupNetwork(){
        cuda_check(cudaFree(input_mem));
        cuda_check(cudaFree(output_mem));
    }

    std::unique_ptr<float[]> infer(float * image){
        auto output = std::unique_ptr<float[]>{new float[number_of_classes]};
        size_t output_size = sizeof(float) * number_of_classes;

        size_t input_size = sizeof(float) * image_size;
        cuda_check(cudaMemcpyAsync(input_mem, image, input_size, cudaMemcpyHostToDevice, stream));
        context->enqueue(1, buffers, stream, nullptr);
        cuda_check(cudaMemcpyAsync(output.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream));

        return output;
    }

private:
    size_t number_of_classes = 10;
    size_t fileSize;
    size_t image_size;
    cudaStream_t stream;
    void * input_mem{nullptr};
    void * output_mem{nullptr};
    char * modelStream{nullptr};

    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context;
    void *buffers[2];

    void checkType(nvinfer1::DataType type1, nvinfer1::DataType type2){
        if(!compareType(type1, type2)){
            std::cout << "output device mismatch" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    bool compareType(nvinfer1::DataType type1, nvinfer1::DataType type2){
        return type1 == type2;
    }
};


void test_demo(){

}


int main(){
    srand(time(NULL));

    int number_of_images = 0;
    int image_size = 0;
    int number_of_labels = 0;
     
     
    auto dataset = load_mnist_images("data/t10k-images-idx3-ubyte", number_of_images, image_size);
    auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", number_of_labels);

    if(number_of_labels != number_of_images){
        std::cout << "label/image size mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }


    size_t number_of_classes = 10;
    auto resnet = ResnetTRT("resnet_fp16.rt");
    
    float correct = 0.0;
    int iterations = number_of_images;

    //copy char dataset to float dataset
    std::vector<std::shared_ptr<float []>> test_dataset;
    for(int i = 0; i < iterations; i++){
        test_dataset.push_back(std::shared_ptr<float []>(new float[image_size]));
        std::transform(dataset[i].get(), dataset[i].get() + image_size, test_dataset[i].get(), [](auto a){return (float)a/255.0;});
    }

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < iterations; i++){
        auto output = resnet.infer(test_dataset[i].get());
        auto softmax = softmax_cpu(output.get(), 10);
        auto pred = std::max_element(softmax.get(), softmax.get() + number_of_classes);
        auto pred_idx = std::distance(softmax.get(), pred);

        if((int)pred_idx == (int)labels[i]){
            correct++;
        }
    }


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    if(iterations > 0){
        std::cout << "Total Time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "accuracy " << (correct/iterations) * 100 << "%" << std::endl;
    }
    
    
    resnet.cleanupNetwork();
    return 0;
}