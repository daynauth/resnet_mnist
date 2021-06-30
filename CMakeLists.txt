CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT(Hello LANGUAGES CXX CUDA)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_PREFIX_PATH /opt/pytorch/libtorch)

FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})


list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
set( CUDA_SEPARABLE_COMPILATION ON)



#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--expt-relaxed-constexpr")

#--------------------
# Copied from tkdnn
#----------------------
find_cuda_helper_libs(nvinfer)
set(NVINFER_LIBRARY ${CUDA_nvinfer_LIBRARY} CACHE FILEPATH "location of the nvinfer library")
unset(CUDA_nvinfer_LIBRARY CACHE)


find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

if(NVINFER_LIBRARY)
  find_path(NVINFER_INCLUDE_DIR
    NvInfer.h
    PATHS ${CUDA_TOOLKIT_INCLUDE}
    DOC "location of NvInfer.h"
    NO_DEFAULT_PATH
  )

  if(NOT NVINFER_INCLUDE_DIR)
    find_path(NVINFER_INCLUDE_DIR
        NvInfer.h
        DOC "location of NvInfer.h"
    )
  endif()

  message("-- Found NVINFER: " ${NVINFER_LIBRARY})
  message("-- Found NVINFER include: " ${NVINFER_INCLUDE_DIR})
endif()


include_directories(${NVINFER_INCLUDES} ${OPENCV_INCLUDE_DIRS})


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr")
add_library(utils utils.cu)
set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(utils PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr")
target_link_libraries(utils ${CUDA_CUBLAS_LIBRARIES})


ADD_EXECUTABLE(Hello main.cpp)
TARGET_LINK_LIBRARIES(Hello "${TORCH_LIBRARIES}")

ADD_EXECUTABLE(load_model load_model.cpp)
TARGET_LINK_LIBRARIES(load_model "${TORCH_LIBRARIES}")
TARGET_LINK_LIBRARIES(load_model "${NVINFER_LIBRARY}")


ADD_EXECUTABLE(resnet Resnet.cpp)
TARGET_LINK_LIBRARIES(resnet "${TORCH_LIBRARIES}")
TARGET_LINK_LIBRARIES(resnet "${NVINFER_LIBRARY}")
TARGET_LINK_LIBRARIES(resnet utils)

ADD_EXECUTABLE(train_resnet_mnist train_resnet_mnist.cpp)
TARGET_LINK_LIBRARIES(train_resnet_mnist "${TORCH_LIBRARIES}")

ADD_EXECUTABLE(test_resnet_mnist test_resnet_mnist.cpp)
TARGET_LINK_LIBRARIES(test_resnet_mnist "${NVINFER_LIBRARY}")
TARGET_LINK_LIBRARIES(test_resnet_mnist "${TORCH_LIBRARIES}")
target_link_libraries(test_resnet_mnist ${OpenCV_LIBS} )

ADD_EXECUTABLE(torch_test_resnet_mnist torch_test_resnet_mnist.cpp)
TARGET_LINK_LIBRARIES(torch_test_resnet_mnist "${TORCH_LIBRARIES}")

