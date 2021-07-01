# Libtorch and TensorRT Implementation of Resnet

## Requirements
Libtorch
TensorRT 7+
Cuda 11.1


## Resnet 18 - MNIST Benchmark Comparison
| Implementation | Time (ms) | Accuracy |
| --- | ---| ---|
| Pytorch  |  26424 |   99.4 |
| Libtorch |  18676  | 99.22|
| TRT 32:  |  15960  | 99.22|
| TRT 16:  |  2408  | 99.22|
