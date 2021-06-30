#ifndef UTILS_H
#define UTILS_H
#include <torch/torch.h>
#include <ATen/Context.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime_api.h"

#define cuda_check(err){if(err != cudaSuccess){\
std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;\
exit(EXIT_FAILURE);}}\


#define mem_check(result){if(!result){\
std::cout << "memory cannot be allocated in" << __FILE__ << " at line " << __LINE__ << std::endl;\
exit(EXIT_FAILURE);}}\


void tensor_accessor(at::Tensor at_weights, float * ft_weights);

#endif