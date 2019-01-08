#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

#define MASTER 0 // taskid von erstem task
#define FROM_MASTER 1 // Nachrichtentypen
#define FROM_WORKER 2
//#define D1 1000 // Zeilenanzahl von A und C
//#define D2 1000 // Spaltenanzahl von A und Zeilenanzahl von B
//#define D3 1000 // Spaltenanzahl von B und C

int main (int argc, char *argv[])
{
    int numtasks, // Anzahl an Tasks
    taskid, // Task ID
    numworkers, i, // Anzahl an Arbeitern
    bsize, bpos, // Zeilenabschnitt von Matrix A
    averow, extra; // Berechnung von Zeilenabschnitten
    int D1, D2, D3;
    float **sendA, **sendB, **finalC , **recvA , **recvC ;
    MPI_Status status; // Statusvariable
    int *sendcounts;
    int *displs;
    


    /* print user instruction */
    if (argc != 4)
    {
        printf ("Matrix multiplication: C = A x B\n");
        printf ("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]); 
        return 0;
    }

    /* read user input */
    D1 = atoi(argv[1]);		// rows of A and C
    D2 = atoi(argv[2]);     // cols of A and rows of B
    D3 = atoi(argv[3]);     // cols of B and C

    printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", D1, D3, D1, D2, D2, D3);

        sendA = alloc_mat(D1, D2); init_mat(sendA, D1, D2); // Speicher für Matrizen holen
        sendB = alloc_mat(D2, D3); init_mat(sendB, D2, D3); // und initialisieren
        recvC = alloc_mat(D1, D3);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    if (numtasks < 2 ) {
        printf("Need at least two tasks!\n");
        MPI_Abort(MPI_COMM_WORLD, 0); exit(1);
    }


    MPI_Scatter( sendA , D1 * D2/numworkers , MPI_INT ,recvA , D1 * D2/numworkers , MPI_INT , 0, MPI_COMM_WORLD ) ;
        //now p0 broadcast sendB to all others
        MPI_Bcast ( sendB, D2*D3 , MPI_INT , 0 , MPI_COMM_WORLD ) ;
		
            for (i = 0; i < bsize; i++)
            for (int j = 0; j < D3; j++)
                for (int k = 0; k < D2; k++)
                    recvC[i][j] += sendA[i][k] * sendB[k][j];                    
	
		
	// nopw p0 will gather all result data from all prcesses
        MPI_Gather( recvC , D1*D3/numworkers , MPI_INT , finalC , D1*D3/numworkers , MPI_INT , 0, MPI_COMM_WORLD ) ;


	// here is the last point of calculating the time		
	MPI_Barrier(MPI_COMM_WORLD); /* barrier is needed if no necessary synchronization for the timing is ensured yet */
	double end = MPI_Wtime(); 


    
    MPI_Finalize();
}
