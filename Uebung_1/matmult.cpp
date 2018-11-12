#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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
    
    float start2, end2;

    for (int i = 0; i < row; i++)
        A1[i] = A2 + i*col;

    return A1;

}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float **A, int row, int col)
{
    float start2, end2;
    start2 = omp_get_wtime();
    //#pragma omp parallel for
    for (int i = 0; i < row*col; i++)
		A[0][i] = (float)(rand() % 10);
    end2 = omp_get_wtime();
    printf ("Benoetigte Zeit matrix erstellen: %f Sekunden\n", end2 - start2);
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
	float **A, **B, **C;	// matrices
    int d1, d2, d3;         // dimensions of matrices
    int i, j, k;			// loop variables

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

    /* serial version of matmult without speedup --> Vergleichswert*/
    printf("Perform parallel matrix multiplication...\n");
    double start, end;
    start = omp_get_wtime();    
    // #pragma omp parallel for collapse(3)
    for (i = 0; i < d1; i++)
       for (j = 0; j < d3; j++)
          for (k = 0; k < d2; k++){
             C[i][j] += A[i][k] * B[k][j];

          }
    end = omp_get_wtime();
    printf ("Benoetigte Zeit ohne Parallel: %f Sekunden\n", end - start);
    /* test output */
    /*  printf ("Ausgabe der matrix ohne Beschleunigung");
    print_mat(A, d1, d2, "A"); 
    print_mat(B, d2, d3, "B"); 
    print_mat(C, d1, d3, "C"); 
    */
    
    /* serial version of matmult with speedup*/
    double start1, end1;
    start1 = omp_get_wtime();

    #pragma omp parallel for collapse(3)
    
    for (i = 0; i < d1; i++)
       for (j = 0; j < d3; j++)
          for (k = 0; k < d2; k++){
           #pragma omp atomic //parallelisieren der Rechnung atomic
            C[i][j] += A[i][k] * B[k][j];

          }
    end1 = omp_get_wtime();
    printf ("Benoetigte Zeit mit Parallel: %f Sekunden\n", end1 - start1);
    //printf ("Ausgabe der matrix mit Beschleunigung");
    /* print_mat(A, d1, d2, "A"); 
    print_mat(B, d2, d3, "B"); 
    print_mat(C, d1, d3, "C"); 
    */

    printf ("\nDone.\n");

    return 0;
}
