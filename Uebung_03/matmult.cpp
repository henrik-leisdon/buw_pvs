#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// ---------------------------------------------------------------------------
// allocate space for empty matrix A[row][col]
// access to matrix elements possible with:
// - A[row][col]
// - A[0][row*col]

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

void init_mat(float **A, int row, int col)
{
    for (int i = 0; i < row*col; i++)
		A[0][i] = (float)(rand() % 10);
}

// ---------------------------------------------------------------------------
// DEBUG FUNCTION: printout of all matrix elements

void print_mat(float **A, int row, int col, char *tag)
{
    int i, j;

    printf("Matrix %s:\n", tag);
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++) 
            printf("%6.1f   ", A[i][j]);
        printf("\n"); 
    }
}

// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
	float **A, **B, **C, **Cref;	// matrices
    int d1, d2, d3;         // dimensions of matrices
    int i, j, k;       		// loop variables
     int numtasks,          //number of tasks in partition
     taskid,                // task identifier -> taskID
     numworkers,            //number of worker task
     source,                //task ID of message source
     dest,                  //task ID of message destination
     rows,                  //rows of Matrix A sent to the workers
     averow, extra, offset; //helps to determine the rows, sent to the workers
   

    /* print user instruction */
    if (argc != 4)
    {
        printf ("Matrix multiplication: C = A x B\n");
        printf ("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]); 
        return 0;
    }

    /* read user input */
    d1 = atoi(argv[1]);		// rows of A and C
    d2 = atoi(argv[2]);     // cols of A and rows of B
    d3 = atoi(argv[3]);     // cols of B and C

    printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2, d2, d3);

    /* prepare matrices */
    A = alloc_mat(d1, d2);
    init_mat(A, d1, d2); 
    B = alloc_mat(d2, d3);
    init_mat(B, d2, d3);
    C = alloc_mat(d1, d3);	// no initialisation of C, because it gets filled by matmult
    Cref = alloc_mat(d1, d3);

   
    float start, end;
     MPI_Status status;

    MPI_Init(&argc,&argv);                  //Initaite MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks-1;

    //--------------- master ----------------------
    //Master thread -> the thread with task ID 0 is master
    // --> coordinates the tasks to the worker - threads
    if(taskid == 0)
    {
        //Send matrix data to worker tasks
        start = MPI_Wtime(); //start time tracking
        averow = d1/numworkers;
        extra = d1%numworkers;
        offset = 0;
        for(dest=1; dest<=numworkers;dest++)
        { 
            rows = (dest <= extra) ? averow+1 : averow;
            printf("send %d rows to task %d ->offset=%d \n", rows, dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&A[offset][0], rows*d2, MPI_FLOAT, dest, 1, MPI_COMM_WORLD); //Send value of Matrix A 
            MPI_Send(B[0], d2*d3, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);          //and B to worker 
            offset = offset + rows;
        }

        // recieve results from worker tasks

            for(i=1;i<=numworkers;i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            //Recieve results for multiplication from A and B
            MPI_Recv(&C[offset][0], rows*d3, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            printf("recieved results from task %d \n", source);
        }

        end = MPI_Wtime();
        printf("Done in %f seconds.\n", end-start);
        printf("finished");

    }
    // ------------------ Worker --------------------
    if(taskid > 0){ //if task ID is > 0,  the thread is a worker
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        //Worker recieves matrix A and B
        MPI_Recv(&A, rows*d2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(B, d2*d3, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    
        //Serial matrix multiplication
       printf("Perform matrix multiplication...\n");
        for (i = 0; i < d1; i++)
            for (j = 0; j < d3; j++)
                for (k = 0; k < d2; k++)
                    C[i][j] += A[i][k]*B[k][j];
                    


        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        //send the result for C back to master
        MPI_Send(&C, rows*d3, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize(); //finishes mpi
    
    /* test output */
    //print_mat(A, d1, d2, "A"); 
    //print_mat(B, d2, d3, "B"); 
    //print_mat(C, d1, d3, "C"); 

    //------- test ---------------------------------------------

            //Serial matrix multiplication
       printf("Perform reference matrix multiplication...\n");
        for (i = 0; i < d1; i++)
            for (j = 0; j < d3; j++)
                for (k = 0; k < d2; k++)
                    Cref[i][j] += A[i][k]*B[k][j];
//test if matrix C is equal of matrix C-reference matrix
        for(int i=0; i< d1; i++)
            for(j=0;j<d3; j++)
                if(C[i][j]!=Cref[i][j])
                printf("error!\n");

        printf("ok.");

    return 0;
}

/*Aufgrund einer nicht identifizierbaren Fehlermeldung beim Ausführen des Programms, 
ist es uns leider nicht möglich, die Matrixmultiplikation auf ihre Richtigkeit und Geschwindigkeits-
verbesserung zu testen.

Der Fehler "Signal: Segmentation fault (11)" geht vermutlich von einem Pointer aus, der nirgendwo hin zeigt.
Dieser wird bei der Matrixmultiplikation gebraucht, wodurch zwar die Daten vom Master zum Worker gesendet werden, aber 
dem worker es nicht möglich ist, die fertig Berechnung dem Master zu senden.
Auch mit der Hilfe eines Debuggers konnten wir den fehlerhaften pointer nicht identifizieren.

Aus diesem Grund können wir auch nicht nachprüfen, ob sich die Beschleunigung verbessert, je mehr Worker wir zur verfügung
stellen. Rein logisch würde dies natürlich Sinn machen, da sich die benötigte Zeit zur Bearbeitung der Aufgabe 
verringert, je mehr Arbeiter vorhanden sind.

*/