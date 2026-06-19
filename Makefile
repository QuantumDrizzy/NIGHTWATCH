# NIGHTWATCH — Linux / WSL2 build (synthetic pipeline, no sensor required).
#
# Requires: nvcc (CUDA toolkit) and OpenCV (pkg-config opencv4).
#   sudo apt install nvidia-cuda-toolkit libopencv-dev
#
# To wire a live ToF/IR sensor, append its SDK flags and -DNIGHTWATCH_USE_SENSOR
# (e.g. a Kinect v2 via libfreenect: add `-lfreenect`).
# To run real neural inference instead of the classical CFAR-centroid fallback,
# build with TensorRT: add -DNIGHTWATCH_USE_TENSORRT and link -lnvinfer.

NVCC   ?= nvcc
SRC     = main.cpp vision_kernel.cu synth_tof.cu trackformer_trt.cpp
CVFLAGS = $(shell pkg-config --cflags --libs opencv4)
TARGET  = nightwatch_vision

$(TARGET): $(SRC)
	$(NVCC) -std=c++17 -o $(TARGET) $(SRC) $(CVFLAGS) -lcudart

clean:
	rm -f $(TARGET)

.PHONY: clean
