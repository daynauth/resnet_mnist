#include <math.h>
#include "utils.h"

__global__ 
void packed_accessor_kernel(torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> weights_a, float *weights, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) weights[i] = weights_a[i];
}

void tensor_accessor(at::Tensor at_weights, float * ft_weights){
    int n = at_weights.sizes()[0];
    int size = n * sizeof(float);
    float * device_weights;
    
    cuda_check(cudaMalloc((void **) &device_weights, size));

    auto weights_a = at_weights.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
    packed_accessor_kernel<<<ceil(n/256.0), 256>>>(weights_a, device_weights, n);

    cudaMemcpy(ft_weights, device_weights, size, cudaMemcpyDeviceToHost);
    cudaFree(device_weights);
}
