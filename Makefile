kħaos_vision: main.cpp vision_kernel.cu
	nvcc -o kħaos_vision main.cpp vision_kernel.cu `pkg-config --cflags --libs opencv4` -lfreenect -lcudart
