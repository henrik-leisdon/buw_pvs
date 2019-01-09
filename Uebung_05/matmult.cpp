// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //include open cl helper file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define DATA_SIZE   10                          // 
#define MEM_SIZE    DATA_SIZE * sizeof(float)   //
#define MAT_SIZE 100

#define NUM_RUNS 2

/** kernel als string setzen **/ 
const char *KernelSource =
	"																					\n"
	"__kernel void matmult(const int D1, const int D2, const int D3,					 "
	"   const	__global float *A, const __global float *B, const __global float *C)  	\n"
	"{																					\n"
	"	const int globalRow = get_global_id(0);											\n"
	"	const int globalCol = get_global_id(1);											\n"
	"	float acc = 0.0f;																  "	
	"	for(int k=0;k<D3;k++){															\n"
	"		acc +=A[k*D1 + globalRow] * B[globalCol*D2+k];								  "
	"	}																				\n"
	"	C[globalCol*D1+ globalRow] = acc;												\n"
	"\n";


	// ---------------------------------------------------------------------------
	// random initialisation of matrix with values [0..9]
	
	void init_mat(float *A, int row, int col)
	{
		for (int i = 0; i < row*col; i++)
			A[i] = (float)(rand() % 10);
	}


/** Beginn der main methode **/
int main (void)
{
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
	size_t				global[1] = {DATA_SIZE};  // größe der Objekte
	float				results[DATA_SIZE] = {0}; // ergebniarray erstellen

	/* 1)  --> Errors? */

	// gibt es eine plattform zum ausführen?
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
	kernel = clCreateKernel(program, "matmult", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}
	clGetDeviceInfo(device_id,CL_DEVICE_NAME,1024, platform_name, NULL);
	cl_event event = NULL;


	/* 2) Das eigentliche programm --> speicher deklarieren?*/

	    // Timers
		struct timeval Tval;
		struct timezone timez;

	// größe der matrizen festlegen
	int D1 = MAT_SIZE;
	int D2 = MAT_SIZE;
	int D3 = MAT_SIZE;
	//alloc matrices
	float * A = (float*)malloc(D1*D2*sizeof(float*));
	float * B = (float*)malloc(D2*D3*sizeof(float*));
	float * C = (float*)malloc(D1*D3*sizeof(float*));
	//init matrices

	printf("initialize matrices \n");
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
	

	/* 3) Execute program?  */


	printf("start matmult...");
	gettimeofday(&Tval,&timez);
	double starttime = (double)Tval.tv_sec + 1.0e-6*((double)Tval.tv_usec);

	for(int i=0;i<NUM_RUNS;i++){
		
		const size_t local[2] = {32,32};
		const size_t global[2] = {D1,D2};
		clEnqueueNDRangeKernel(command_queue,kernel,2, NULL, global, local, 0, NULL, &event);

		clWaitForEvents(1, &event);
	}

	gettimeofday(&Tval, &timez);
	double endtime = (double)Tval.tv_sec + 1.0e-6*((double)Tval.tv_usec);
	double runtime = (endtime - starttime) / (double)NUM_RUNS;
	printf(">>> Done: took %.3lf seconds per run \n", runtime);

	clEnqueueReadBuffer(command_queue,bufC,CL_TRUE, 0, D1*D2*sizeof(float),C, 0, NULL, NULL);


	/* 4) speicher/buffer wieder frei geben*/
	

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