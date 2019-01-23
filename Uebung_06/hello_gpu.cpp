// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              //include open cl helper file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE   10                          // die datengröße ist 10
#define MEM_SIZE    DATA_SIZE * sizeof(float)   // speichergröße ist für DATA_SIZE


/** kernel  string definieren **/
const char *KernelSource =
	"#define DATA_SIZE 10												\n"
	"__kernel void test(__global float *input, __global float *output)  \n"
	"{																	\n"
	"	size_t i = get_global_id(0);									\n"
	"	output[i] = input[i] * input[i];								\n"
	"}																	\n"
	"\n";

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

	// reserviert speicher für input und output
	input  = clCreateBuffer (context, CL_MEM_READ_ONLY,	 MEM_SIZE, NULL, &err);
	output = clCreateBuffer (context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	// input speicher in der commandqueue berechnen
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	// spezifische kernel argumente setzen 
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);


	/* 3) Execute program?  */

	// reiht befehl zum ausführen in den kernel bzw. commandqueue ein
	clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	// blockiert, bis alle eingereihten openCL befehle in der command queue ausgeführt sind
	clFinish(command_queue);

	// einreihen des buffers in die command queue zur ausgabe von output
	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);

  // ausgabe von hello world array
  for (unsigned int i=0; i < DATA_SIZE; i++)
    printf("%f\n", results[i]);


	/* 4) speicher/buffer wieder frei geben*/
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}