// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //include open cl helper file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define MAT_SIZE 1000
#define DATA_SIZE 10
#define MEM_SIZE MAT_SIZE * sizeof(float)
#define NUM_RUNS 2


float **alloc_mat(int row, int col)
{
    float **A1, *A2;

	A1 = (float **)calloc(row, sizeof(float *));		// pointer on rows
	A2 = (float *)calloc(row*col, sizeof(float));    // all matrix elements
    for (int i = 0; i < row; i++)
        A1[i] = A2 + i*col;

    return A1;
}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float *A, int row, int col)
{
	for (int i = 0; i < row*col; i++)
		A[i] = (float)(rand() % 10);
}



/** kernel  string definieren **/
const char *KernelSource =
"#define DIM 1000  // Size of matrix												\n"
"__kernel void matmult(__global float *A, __global float *B, __global float *C) {	\n"
" int i, j, k;																		\n"
" float sum = 0.0;																	\n"
" j = get_global_id(0);																\n"
" i = get_global_id(1);																\n"
" for (k = 0; k < DIM; k++)															\n"
" sum += A[i*DIM+k] * B[k*DIM+j];													\n"
" C[i*DIM+j] = sum;																	\n"
"}"

/** Beginn der main methode **/
int main (void)
{
    int D1 = MAT_SIZE;
	int D2 = MAT_SIZE;
	int D3 = MAT_SIZE;
	cl_int				err;                      // integer für error erstellen
	cl_platform_id*		platforms = NULL;         // plattform ID
	char			    platform_name[1024];      // plattform name
	cl_device_id	    device_id = NULL;         // gerät ID/ device ID
	cl_uint			    num_of_platforms = 0,     // anzahl der plattformen
					    num_of_devices = 0;       // anzahl der devices
	cl_context 			context;                  // initialisieren eines kontext
	cl_kernel 			kernel;                   // kernel initialisieren
	cl_command_queue	command_queue;            // commandqueue initialisieren
	cl_program 			program;                  // programm initialisieren
	cl_mem				input, output;            // input/output speicher initialisieren
	float				data[DATA_SIZE] =         // data array erstellen
							{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	size_t				global[2] = {D2,D3};  // größe der Objekte
	float				results[DATA_SIZE] = {0}; // ergebniarray erstellen

	/* kernel initialiseren - testen, ob alles richtig initialisert werden kann */

	// plattformen suchen
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// alle verfügbaren plattformen sammeln
	platforms = (cl_platform_id *)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// für alle vorhandenen plattformen: 
		for (unsigned int i=0; i<num_of_platforms; i++)
		{
			//speziefiche informationen über die OpenCL plattform sammeln
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name,	NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			
			// ist die plattfornm eine nvidia plattform?
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// does the list of the available devices are available on the platform?
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// ein kontext wird erstellt um objekte, command-queues, speicher usw zu managen
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	// erstellt eine command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// initialisiert ein programm für den kontext
	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

  // baut/erstellt das vorher initialisierte programm
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	// erstellt den kernel im programm
	kernel = clCreateKernel(program, "test", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) Das eigentliche programm --> speicher deklarieren?*/

    //alloc matrices
	float * A = (float*)malloc(D1*D2*sizeof(float*));
	float * B = (float*)malloc(D2*D3*sizeof(float*));
	float * C = (float*)malloc(D1*D3*sizeof(float*));
	//init matrices

	printf("initialize matrices \n");
	printf("\n");
	init_mat(A, D1,D2);
	init_mat(B, D2,D3);

	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, D1*D2*sizeof(float), NULL, NULL);
	cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, D2*D3*sizeof(float), NULL, NULL);
	cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_ONLY, D1*D3*sizeof(float), NULL, NULL);

    // input in den speicher buffer einreihen?
	clEnqueueWriteBuffer(command_queue, bufA, CL_TRUE, 0, D1*D2*sizeof(float), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, bufB, CL_TRUE, 0, D2*D3*sizeof(float), B, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, bufC, CL_TRUE, 0, D1*D3*sizeof(float), C, 0, NULL, NULL);
	


	// spezifische kernel argumente setzen 
	clSetKernelArg(kernel, 0, sizeof(int), (void*)&D1);
	clSetKernelArg(kernel, 1, sizeof(int), (void*)&D2);
	clSetKernelArg(kernel, 2, sizeof(int), (void*)&D3);
	
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufC);
	


	//clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	printf(">>> Starting %d myGEMM runs...\n", NUM_RUNS);
	gettimeofday(&Tvalue, &dummy);
	double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

    for(int i=0;i<NUM_RUNS;i++){
		
		const size_t local[2] = {32,32};
		const size_t global[2] = {D1,D2};
		
		clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

		// Wait for calculations to be finished
		clWaitForEvents(1, &event);
	}

	gettimeofday(&Tvalue, &dummy);
	double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
	double runtime = (endtime - starttime) / (double)NUM_RUNS;
	double gflop = ((long)K * (long)M * (long)N * 2) / (1000 * 1000 * 1000);
	printf(">>> Done: took %.3lf seconds per run, %.1lf GFLOPS\n", runtime, gflop / runtime);



	// blockiert, bis alle eingereihten openCL befehle in der command queue ausgeführt sind
	clFinish(command_queue);

	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	free(A);
	free(B);
	free(C);

	return 0;
}