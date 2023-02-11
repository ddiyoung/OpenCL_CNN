#pragma warning(disable : 4996)
#include <CL/cl.h>
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <direct.h>

extern const char* CLASS_NAME[];

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {
	FILE* file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
}

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_command_queue read_queue;
size_t kernel_source_size;
char* kernel_source;
cl_program program;
cl_kernel kernel_convolution, kernel_max_pooling, kernel_fc_layer, kernel_convolution2;
cl_context context;
cl_command_queue queue;
cl_mem buffer1, buffer2, buffer3;
int order;

void cnn_init(float* network) {
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("kernel.cl", &kernel_source_size);

	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	build_error(program, device, err);
	CHECK_ERROR(err);

	kernel_convolution = clCreateKernel(program, "convolution", &err);
	CHECK_ERROR(err);

	kernel_max_pooling = clCreateKernel(program, "max_pooling", &err);
	CHECK_ERROR(err);

	kernel_fc_layer = clCreateKernel(program, "fc_layer", &err);
	CHECK_ERROR(err);

	kernel_convolution2 = clCreateKernel(program, "convolution2", &err);
	CHECK_ERROR(err);

	buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * 750, NULL, &err);
	CHECK_ERROR(err);

	buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * 750, NULL, &err);
	CHECK_ERROR(err);

	buffer3 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * (60980520 / 4), network, &err);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution, 0, sizeof(cl_mem), &buffer1);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution, 1, sizeof(cl_mem), &buffer2);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution, 2, sizeof(cl_mem), &buffer3);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_max_pooling, 0, sizeof(cl_mem), &buffer1);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_max_pooling, 1, sizeof(cl_mem), &buffer2);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_fc_layer, 0, sizeof(cl_mem), &buffer1);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_fc_layer, 1, sizeof(cl_mem), &buffer2);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_fc_layer, 2, sizeof(cl_mem), &buffer3);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_fc_layer, 3, sizeof(float) * 512, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution2, 2, sizeof(cl_mem), &buffer3);
	CHECK_ERROR(err);
}

static void convolution2(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn, int num_of_image, int f_index, int b_index) {
	size_t local_size = inDim;
	size_t global_size = nbyn * nbyn * inDim * outDim * num_of_image;

	if (order % 2 == 1) {
		err = clSetKernelArg(kernel_convolution2, 0, sizeof(cl_mem), &buffer2);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel_convolution2, 1, sizeof(cl_mem), &buffer1);
		CHECK_ERROR(err);
	}
	else {
		err = clSetKernelArg(kernel_convolution2, 0, sizeof(cl_mem), &buffer1);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel_convolution2, 1, sizeof(cl_mem), &buffer2);
		CHECK_ERROR(err);
	}

	err = clSetKernelArg(kernel_convolution2, 8, sizeof(float) * inDim, NULL);
	CHECK_ERROR(err);

	int in_Dim = inDim;
	int out_Dim = outDim;
	int n = nbyn;
	int bias_index = b_index;

	err = clSetKernelArg(kernel_convolution2, 6, sizeof(int), &bias_index);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution2, 3, sizeof(int), &in_Dim);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution2, 4, sizeof(int), &out_Dim);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution2, 5, sizeof(int), &n);
	CHECK_ERROR(err);

	int filter_index = f_index;
	err = clSetKernelArg(kernel_convolution2, 7, sizeof(int), &filter_index);
	CHECK_ERROR(err);


	clEnqueueNDRangeKernel(
		queue, kernel_convolution2, 1, NULL,
		&global_size, &local_size,
		0, NULL, NULL
	);

	if (order == 14) {
		err = clEnqueueReadBuffer(queue, buffer2, CL_TRUE, 0, sizeof(float) * nbyn * nbyn * outDim * num_of_image, outputs, 0, NULL, NULL);
		CHECK_ERROR(err);
	}
}

static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn, int num_of_image, int f_index, int b_index) {

	size_t local_size = nbyn * nbyn;
	size_t global_size = nbyn * nbyn * outDim * num_of_image;

	if (order == 0) {
		err = clEnqueueWriteBuffer(queue, buffer1, CL_TRUE, 0, sizeof(float) * nbyn * nbyn * inDim * num_of_image, inputs, 0, NULL, NULL);
		CHECK_ERROR(err);
	}
	else {
		if (order % 2 == 1) {
			err = clSetKernelArg(kernel_convolution, 0, sizeof(cl_mem), &buffer2);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_convolution, 1, sizeof(cl_mem), &buffer1);
			CHECK_ERROR(err);
		}
		else {
			err = clSetKernelArg(kernel_convolution, 0, sizeof(cl_mem), &buffer1);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_convolution, 1, sizeof(cl_mem), &buffer2);
			CHECK_ERROR(err);
		}
	}
	

	int in_Dim = inDim;
	int out_Dim = outDim;
	int n = nbyn;
	int bias_index = b_index;

	err = clSetKernelArg(kernel_convolution, 6, sizeof(int), &bias_index);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution, 3, sizeof(int), &in_Dim);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution, 4, sizeof(int), &out_Dim);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_convolution, 5, sizeof(int), &n);
	CHECK_ERROR(err);

	int filter_index = f_index;
	err = clSetKernelArg(kernel_convolution, 7, sizeof(int), &filter_index);
	CHECK_ERROR(err);


	clEnqueueNDRangeKernel(
		queue, kernel_convolution, 1, NULL,
		&global_size, &local_size,
		0, NULL, NULL
	);
	if (order == 14) {
		err = clEnqueueReadBuffer(queue, buffer2, CL_TRUE, 0, sizeof(float) * nbyn * nbyn * outDim * num_of_image, outputs, 0, NULL, NULL);
		CHECK_ERROR(err);
	}
}

static void max_pooling(float* input, float* output, int DIM, int nbyn, int num_of_image) {
	if (order % 2 == 1) {
		err = clSetKernelArg(kernel_max_pooling, 0, sizeof(cl_mem), &buffer2);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel_max_pooling, 1, sizeof(cl_mem), &buffer1);
		CHECK_ERROR(err);
	}
	else {
		err = clSetKernelArg(kernel_max_pooling, 0, sizeof(cl_mem), &buffer1);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel_max_pooling, 1, sizeof(cl_mem), &buffer2);
		CHECK_ERROR(err);
	}

	int in_Dim = DIM;
	int n = nbyn;

	err = clSetKernelArg(kernel_max_pooling, 2, sizeof(int), &in_Dim);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_max_pooling, 3, sizeof(int), &n);
	CHECK_ERROR(err);


	size_t local_size = nbyn * nbyn;
	size_t global_size = nbyn * nbyn * DIM * num_of_image;

	clEnqueueNDRangeKernel(
		queue, kernel_max_pooling, 1, NULL,
		&global_size, &local_size,
		0, NULL, NULL
	);
}


void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim, int num_of_image, int f_index, int b_index) {
	if (order % 2 == 1) {
		err = clSetKernelArg(kernel_fc_layer, 0, sizeof(cl_mem), &buffer2);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel_fc_layer, 1, sizeof(cl_mem), &buffer1);
		CHECK_ERROR(err);
	}
	else {
		err = clSetKernelArg(kernel_fc_layer, 0, sizeof(cl_mem), &buffer1);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel_fc_layer, 1, sizeof(cl_mem), &buffer2);
		CHECK_ERROR(err);
	}

	int in_Dim = inDim;
	int out_Dim = outDim;

	err = clSetKernelArg(kernel_fc_layer, 4, sizeof(int), &in_Dim);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_fc_layer, 5, sizeof(int), &out_Dim);
	CHECK_ERROR(err);

	int filter_index = f_index;
	int bias_index = b_index;

	err = clSetKernelArg(kernel_fc_layer, 6, sizeof(int), &filter_index);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_fc_layer, 7, sizeof(int), &bias_index);
	CHECK_ERROR(err);

	size_t local_size = inDim;
	size_t global_size = inDim * outDim * num_of_image;

	clEnqueueNDRangeKernel(
		queue, kernel_fc_layer, 1, NULL,
		&global_size, &local_size,
		0, NULL, NULL
	);

	if (order == 20) {
		err = clEnqueueReadBuffer(queue, buffer2, CL_TRUE, 0, sizeof(float) * outDim * num_of_image, output, 0, NULL, NULL);
		CHECK_ERROR(err);
	}
}

static void softmax(float* input, int N, int j) {
	int i;
	float max = input[0];
	for (i = 1; i < N; i++) {
		if (max < input[i + j * N]) max = input[i + j * N];
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(input[i + j * N] - max);
	}
	for (i = 0; i < N; i++) {
		input[i + j * N] = exp(input[i + j * N] - max) / (sum + 1e-7);
	}
}

static int find_max(float* input, int classNum, int j) {
	int i;
	int maxIndex = 0;
	float max = 0;
	for (i = 0; i < classNum; i++) {
		if (max < input[i + j * classNum]) {
			max = input[i + j * classNum];
			maxIndex = i + j * classNum;
		}
	}
	return maxIndex;
}


const int INPUT_DIM[] = {
	3, 64,
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10
};

const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};

int network_index = 0;

const int NETWORK[] = {
0,
1728,
1792,
38656,
38720,
112448,
112576,
260032,
260160,
555072,
555328,
1145152,
1145408,
1735232,
1735488,
2915136,
2915648,
5274944,
5275456,
7634752,
7635264,
9994560,
9995072,
12354368,
12354880,
14714176,
14714688,
14976832,
14977344,
15239488,
15240000,
15245120
};

void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {

	float* w[21];
	float* b[21];
	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}

	// allocate memory for layer
	float* layer[21];
	
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i] * 750);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}


	for (int loop = 0; loop < 4; ++loop) {
		order = 0;
		network_index = 0;

		convolution(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], 750, NETWORK[network_index], NETWORK[network_index+1]); order++; network_index += 2;
		convolution(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		max_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2, 750); order++;
		
		convolution(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		max_pooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2, 750); order++;
		
		convolution(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		max_pooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2, 750); order++;
		
		convolution(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		max_pooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2, 750); order++;
		
		convolution(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		convolution(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		max_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2, 750); order++;

		fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;
		fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20], 750, NETWORK[network_index], NETWORK[network_index + 1]); order++; network_index += 2;


		// run network
		for (int i = 0; i < 750; ++i) {
			softmax(layer[20], 10, i);

			labels[i + 750 * loop] = find_max(layer[20], 10, i) - 10 * i;
			confidences[i + 750 * loop] = layer[20][labels[i + 750 * loop] + 10 * i];
		}

		images += 32 * 32 * 3 * 750;
	}

	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}
}
