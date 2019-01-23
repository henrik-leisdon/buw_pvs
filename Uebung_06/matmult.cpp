// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //include open cl helper file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define DATA_SIZE   10                          // 
#define MEM_SIZE    DATA_SIZE * sizeof(float)   //
#define MAT_SIZE 2000
#define DIM 1000 
#define NUM_RUNS 2

/** kernel  string definieren **/ 
const char *KernelSource =

" 																		  "  // Groesse der Matrix
"__kernel void matmult(__global float *A, __global float *B, __global float *C) { 		\n"
"int i, j, k;																			\n"
"float sum; // Private Variable für Zwischenergebnisse									\n"	
"i = get_global_id(0);																	\n"	
"for (j = 0; j < DIM; j++) {															\n"			
"sum = 0.0;																				\n"		
"for (k = 0; k < DIM; k++)																\n"
"sum += A[i*DIM+k] * B[k*DIM+j];														\n"	
"C[i*DIM+j] = sum;																		\n"
"}																						\n"				
"}																						\n";


/*"																					\n"
"__kernel void matmult(const int D1, const int D2, const int D3,					\n"
"   const	__global float *A, const __global float *B, __global float *C)  		\n"
"{																					\n"
"	const int globalRow = get_global_id(0);											\n"
"	const int globalCol = get_global_id(1);											\n"
"	float Al[DIM], acc;																\n"
"	for(int k=0;k<DIM;l++)															\n"
"		Al[k] = A[k*D1 + globalRow];												\n"
"	acc = 0.0f;																  		\n"	
"	for(k=0;k<DIM;k++){																\n"
"		acc += Al[k] * B[k*DIM+k];													\n"
"	C[globalCol*D1+ globalRow] = acc; 												\n"
"	}																				\n"
" } \n";
*/

/* "#define DIM 1000 // Size of matrix												\n"
"__kernel void matmult(__global float *A, __global float *B, __global float *C) {	\n"
" int i, j, k;																		\n"
" float sum = 0.0;																	\n"
" j = get_global_id(0);																\n"
" i = get_global_id(1);																\n"
" for (k = 0; k < DIM; k++)															\n"
" sum += A[i*DIM+k] * B[k*DIM+j];													\n"
" C[i*DIM+j] = sum;																	\n"
"}" 
*/

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
	size_t				global[1] = {DIM};  // größe der Objekte
	float				results[DATA_SIZE] = {0}; // ergebniarray erstellen

	/* 1)  Erstellen des programms */

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

		for (unsigned int i=0; i<num_of_platforms; i++)
		{
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name,	NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

			err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

		context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}


	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

  	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	
	kernel = clCreateKernel(program, "matmult", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}
	//event erstellen um Zeit zu messen
	cl_event event;
	

	/* 2) Das eigentliche programm --> speicher deklarieren?*/

	    // Timers
		struct timeval Tval;
		struct timezone timez;

	// größe der matrizen festlegen
	
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
	

	/* 3) Execute program?  */


	printf("start matmult \n");

	cl_ulong time_start;
	cl_ulong time_end;

	//double startomp, endomp; 

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	clEnqueueNDRangeKernel(command_queue,kernel,1, NULL, global,NULL, 0, NULL, NULL);
	//startomp = omp_get_wtime();

	for(int i=0;i<NUM_RUNS;i++){
		
		const size_t local[2] = {32,32};
		const size_t global[2] = {D1,D2};
		

		clWaitForEvents(1, &event);
		
	}
	clFinish(command_queue);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end) , &time_end, NULL);
	
	printf("Done \n");

	//endomp = omp_get_wtime();
	printf(" Done took %0.3f milliseconds \n", ((time_end-time_start)/1000000.0));
	//printf("Done, took %f seconds \n", endomp-startomp);
	
	clEnqueueReadBuffer(command_queue,bufC,CL_TRUE, 0, D1*D2*sizeof(float),C, 0, NULL, NULL);




	//--------serielle version-----------------------------------------------------------
/*
	float **Aser, **Bser, *Cser, **Ctemp;	// matrices
    int d1, d2, d3;         // dimensions of matrices
    int i, j, k;			// loop variables


    // prepare matrices 
    Aser = alloc_mat(d1, d2);
    init_mat(A, d1, d2); 
    Bser = alloc_mat(d2, d3);
    init_mat(B, d2, d3);
    Cser = (float *)calloc(d1*d3, sizeof(float));	// no initialisation of C, because it gets filled by matmult
	Ctemp = alloc_mat(d1, d3);
    //serial version of matmult without speedup --> Vergleichswert
    printf("Perform serial matrix multiplication...\n");

    
    for (i = 0; i < D1; i++)
       for (j = 0; j < D3; j++)
          for (k = 0; k < D2; k++){
             Ctemp[i][j] += Aser[i][k] * Bser[k][j];


          }

	for (int q = 0; q < d1; q++){
    	for (int t = 0; t < d3; t++){
        	Cser[q * d1 + t] = Ctemp[q][t];
    	}
	}
    printf ("\nDone.\n");

	//-----Test ----------------------------
	/*
	for(i=0;i<Cser.length;i++){
		if(Cser[i]!=C[i]){
			printf("An error has been detected. Matrices are not equal.");
			break;
		}
	}
*/
	
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