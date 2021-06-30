#include <iostream>
#include <torch/torch.h>
#include <ATen/Context.h>

template<typename Base, typename T>
inline bool instanceof(const T*){
    return std::is_base_of<Base, T>::value;
}

torch::nn::Conv2d conv3x3(int in_inplanes, int out_planes, int stride = 1, int groups = 1, int dilation = 1){
    return torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_inplanes, out_planes, 3).stride(stride).padding(dilation).groups(groups).bias(false).dilation(dilation)
    );
}

torch::nn::Conv2d conv1x1(int in_planes, int out_planes, int stride = 1){
    return torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_planes, out_planes, 1).stride(stride).bias(false)
    );
}


struct BasicBlock : torch::nn::Module{
    static const int expansion = 1;
    std::shared_ptr<torch::nn::Conv2dImpl> _conv1;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> _bn1;
    std::shared_ptr<torch::nn::ReLUImpl> _relu;
    std::shared_ptr<torch::nn::Conv2dImpl> _conv2;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> _bn2;
    std::shared_ptr<torch::nn::SequentialImpl> _downsample;
    at::Tensor out;

    BasicBlock(int inplanes, int planes, int stride = 1, torch::nn::Sequential downsample = nullptr, int groups = 1, int base_width = 64, int dilation = 1){
        if(groups != 1 || base_width != 64){
            std::cerr << "BasicBlock only supports groups = 1 and base_width = 64" << std::endl;
            exit(EXIT_FAILURE);
        }

        if(dilation > 1){
            std::cerr << "Dilation > 1 not supported in BasicBlock" << std::endl;
            exit(EXIT_FAILURE);
        }

        _conv1 = register_module("conv1", conv3x3(inplanes, planes, stride));

        _bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
        _relu = register_module("relu", torch::nn::ReLU(
            torch::nn::ReLUOptions().inplace(true)
        ));

        _conv2 = register_module("conv2", conv3x3(planes, planes));
        _bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));

        if(downsample){
            _downsample = register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x){
        auto identity = x;
        
        out = _conv1->forward(x);
        out = _bn1->forward(out);
        out = _relu->forward(out);

        out = _conv2->forward(out);
        out = _bn2->forward(out);

        
        if(_downsample){
            identity = _downsample->forward(x);
        }

        out += identity;
        out = _relu->forward(out);

        return out;
    }
};


struct Bottleneck : torch::nn::Module{
    static const int expansion = 4;
    int _stride;
    std::shared_ptr<torch::nn::Conv2dImpl> _conv1;
    std::shared_ptr<torch::nn::Conv2dImpl> _conv2;
    std::shared_ptr<torch::nn::Conv2dImpl> _conv3;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> _bn1;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> _bn2;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> _bn3;
    std::shared_ptr<torch::nn::ReLUImpl> _relu;
    std::shared_ptr<torch::nn::SequentialImpl> _downsample;
    
    at::Tensor out;


    Bottleneck(int inplanes, int planes, int stride = 1, torch::nn::Sequential downsample = nullptr, int groups = 1, int base_width = 64, int dilation = 1){
        int width = (int)(planes * (base_width / 64)) * groups;
        
        _conv1 = register_module("conv1", conv1x1(inplanes, width));
        _bn1 = register_module("bn1", torch::nn::BatchNorm2d(width));
        _conv2 = register_module("conv2", conv3x3(width, width, stride, groups, dilation));
        _bn2 = register_module("bn2", torch::nn::BatchNorm2d(width));
        _conv3 = register_module("conv3", conv1x1(width, planes * expansion));
        _bn3 = register_module("bn3", torch::nn::BatchNorm2d(planes * expansion));
        _relu = register_module("relu", torch::nn::ReLU(
            torch::nn::ReLUOptions().inplace(true)
        ));

        if(downsample)
            _downsample = register_module("downsample", downsample);

        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x){
        auto identity = x;
        
        out = _conv1->forward(x);
        out = _bn1->forward(out);
        out = _relu->forward(out);

        out = _conv2->forward(out);
        out = _bn2->forward(out);
        out = _relu->forward(out);

        out = _conv3->forward(out);
        out = _bn3->forward(out);

        if(_downsample){
            identity = _downsample->forward(x);
        }

        out += identity;
        out = _relu->forward(out);

        return out;
    }

};

template<class block>
struct Resnet : torch::nn::Module{
    int _inplanes = 64;
    int _dilation = 1;
    int _groups = 1;
    int _base_width;

    std::shared_ptr<torch::nn::Conv2dImpl> _conv1;
    std::shared_ptr<torch::nn::BatchNorm2dImpl> _bn1;
    std::shared_ptr<torch::nn::ReLUImpl> _relu;
    std::shared_ptr<torch::nn::MaxPool2dImpl> _maxpool;
    torch::nn::Sequential _layer1;
    torch::nn::Sequential _layer2;
    torch::nn::Sequential _layer3;
    torch::nn::Sequential _layer4;
    std::shared_ptr<torch::nn::AvgPool2dImpl> _avgpool;
    std::shared_ptr<torch::nn::LinearImpl> _fc;

    Resnet(int layers[], int num_classes=1000, bool zero_init_residual=false, int groups = 1, int width_per_group = 64){
        _groups = groups;
        _base_width = width_per_group;

        _conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, _inplanes, 7).stride(2).padding(3).bias(false)
        ));

        _bn1 = register_module("bn1", torch::nn::BatchNorm2d(_inplanes));
        _relu = register_module("relu", torch::nn::ReLU(
            torch::nn::ReLUOptions().inplace(true)
        ));
        _maxpool = register_module("maxpool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(3).stride(2).padding(1)
        ));

        _layer1 = register_module("layer1", _make_layer(64, layers[0]));
        _layer2 = register_module("layer2", _make_layer(128, layers[1], 2));
        _layer3 = register_module("layer3", _make_layer(256, layers[2], 2));
        _layer4 = register_module("layer4", _make_layer(512, layers[3], 2));
       
        _avgpool = register_module("avgpool", torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions({1,1}).stride(1)
       ));

        _fc = register_module("fc", torch::nn::Linear(512 * block::expansion, num_classes));        
    }

    torch::nn::Sequential _make_layer(int planes, int blocks, int stride = 1, bool dilate = false){
        torch::nn::Sequential downsample = nullptr;
        int previous_dilation = _dilation;

        if(dilate){
            _dilation *= stride;
            stride = 1;
        }

        if(stride != 1 || _inplanes != (planes * block::expansion)){
            downsample = torch::nn::Sequential(
                conv1x1(_inplanes, planes * block::expansion, stride),
                torch::nn::BatchNorm2d(planes * block::expansion)        
            );
        }

        auto sequential = torch::nn::Sequential();
        sequential->push_back(std::make_shared<block>(_inplanes, planes, stride, downsample, _groups, _base_width, previous_dilation));

        _inplanes = planes * block::expansion;

        for(int i = 1; i < blocks; i++){
            sequential->push_back(std::make_shared<block>(_inplanes, planes, 1, nullptr, _groups, _base_width, _dilation));
        }

        return sequential;
    }

    torch::Tensor forward(torch::Tensor x){
        x = _conv1->forward(x);
        x = _bn1->forward(x);
        x = _relu->forward(x);
        x = _maxpool->forward(x);

        x = _layer1->forward(x);
        x = _layer2->forward(x);
        x = _layer3->forward(x);
        x = _layer4->forward(x);

        x = _avgpool->forward(x);
        x = torch::nn::Flatten(torch::nn::FlattenOptions())->forward(x);
        x = _fc->forward(x);

        return x;
    }
};


template <typename block>
std::shared_ptr<Resnet<block>> _resnet(int layers[], int num_classes = 1000){
    return std::make_shared<Resnet<block>>(layers, num_classes);
}


std::shared_ptr<Resnet<BasicBlock>> Resnet18(int num_classes = 1000){
    int layers[] = {2, 2, 2, 2};
    return _resnet<BasicBlock>(layers, num_classes);
}


std::shared_ptr<Resnet<BasicBlock>> Resnet34(int num_classes = 1000){
    int layers[] = {3, 4, 6, 3};
    return _resnet<BasicBlock>(layers, num_classes);
}

std::shared_ptr<Resnet<Bottleneck>> Resnet50(int num_classes = 1000){
    int layers[] = {3, 4, 6, 3};
    return _resnet<Bottleneck>(layers, num_classes);
}