#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0 // taskid von erstem task
#define FROM_MASTER 1 // Nachrichtentypen
#define FROM_WORKER 2

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




int main (int argc, char *argv[])
{
    int numtasks, // Anzahl an Tasks
    taskid, // Task ID
    numworkers, i, // Anzahl an Arbeitern
    bsize, bpos, // Zeilenabschnitt von Matrix A
    averow, extra; // Berechnung von Zeilenabschnitten
    int D1, D2, D3;
    float **A, **B, **C, **D; // Matrizen
    MPI_Status status; // Statusvariable
    


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


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    if (numtasks < 2 ) {
        printf("Need at least two tasks!\n");
        MPI_Abort(MPI_COMM_WORLD, 0); exit(1);
    }

//****************************** Master Task ************************************
    if (taskid == MASTER) {
        printf("MatMult started with %d tasks.\n", numtasks);
        A = alloc_mat(D1, D2); init_mat(A, D1, D2); // Speicher für Matrizen holen
        B = alloc_mat(D2, D3); init_mat(B, D2, D3); // und initialisieren
        C = alloc_mat(D1, D3);
        
        float start, end;
        start = MPI_Wtime(); // Zeitmessung starten
        numworkers = numtasks-1; // Anzahl der Arbeiter
        averow = D1 / numworkers; // Mittlere Blockgröße
        extra = D1 % numworkers; // Restzeilen
        
        for (i = 1, bpos = 0; i <= numworkers; i++, bpos += bsize) {
            if (i > extra) { // Restzeilen aufteilen
                bsize = averow;
            } else {
            bsize = averow+1;
        } // Senden der Matrixblöcke an die Arbeiter
            printf("Sending %d rows to task %d\n", bsize, i);
            MPI_Send(&bpos, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&bsize, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(A[bpos], bsize*D2, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
            MPI_Send(B[0], D2*D3, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        }


        for (i = 1; i <= numworkers; i++) { // Empfangen der Ergebnisse von den Arbeitern
            MPI_Recv(&bpos, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&bsize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(C[bpos], bsize*D3, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n", i);
        }
        printf("\nUsed %f seconds.\n", MPI_Wtime() - start); // Zeitmessung anhalten


}
//****************************** Worker Task ************************************
    if (taskid > MASTER) {
        
        MPI_Recv(&bpos, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&bsize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        
        A = alloc_mat(bsize, D2); // Speicher für die Matrixblöcke holen
        B = alloc_mat(D2, D3);
        C = alloc_mat(bsize, D3);
        
        MPI_Recv(A[0], bsize*D2, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(B[0], D2*D3, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        
        for (i = 0; i < bsize; i++)
            for (int j = 0; j < D3; j++)
                for (int k = 0; k < D2; k++)
                    C[i][j] += A[i][k] * B[k][j];
        
        MPI_Send(&bpos, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&bsize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(C[0], bsize*D3, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);


    }


//********VARIANTE 2 nicht blockierend***********************************************************************

    float **Av2, **Bv2, **Cv2, **Dv2; // Matrizen
    MPI_Request send_request, recv_request;



//****************************** Master Task ************************************
    if (taskid == MASTER) {
        printf("MatMult - NICHT BLOCKIEREND - started with %d tasks.\n", numtasks);
        Av2 = alloc_mat(D1, D2); init_mat(Av2, D1, D2); // Speicher für Matrizen holen
        Bv2 = alloc_mat(D2, D3); init_mat(Bv2, D2, D3); // und initialisieren
        Cv2 = alloc_mat(D1, D3);
        
        float start, end;
        start = MPI_Wtime(); // Zeitmessung starten
        numworkers = numtasks-1; // Anzahl der Arbeiter
        averow = D1 / numworkers; // Mittlere Blockgröße
        extra = D1 % numworkers; // Restzeilen
        
        for (i = 1, bpos = 0; i <= numworkers; i++, bpos += bsize) {
            if (i > extra) { // Restzeilen aufteilen
                bsize = averow;
            } else {
            bsize = averow+1;
        } // Senden der Matrixblöcke an die Arbeiter
            printf("Sending %d rows to task %d\n", bsize, i);
            MPI_Isend(&bpos, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, &status);
            MPI_Isend(&bsize, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, &status);
            MPI_Isend(Av2[bpos], bsize*D2, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, &status);
            MPI_Isend(Bv2[0], D2*D3, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, &status);
        }


        for (i = 1; i <= numworkers; i++) { // Empfangen der Ergebnisse von den Arbeitern
            MPI_Irecv(&bpos, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &status);
            MPI_Irecv(&bsize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &status);
            MPI_Irecv(Cv2[bpos], bsize*D3, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &status);

            MPI_Wait(&recv_request, &status);
            printf("Received results from task %d\n", i);
        }
        printf("\nUsed %f seconds.\n", MPI_Wtime() - start); // Zeitmessung anhalten
}


//****************************** Worker Task ************************************
    if (taskid > MASTER) {
        
        MPI_Irecv(&bpos, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &recv_request);
        MPI_Wait(&recv_request, &status);
        MPI_Irecv(&bsize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &recv_request);
        MPI_Wait(&recv_request, &status);

        Av2 = alloc_mat(bsize, D2); // Speicher für die Matrixblöcke holen
        Bv2 = alloc_mat(D2, D3);
        Cv2 = alloc_mat(bsize, D3);
        

        MPI_Recv(Av2[0], bsize*D2, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Wait(&recv_request, &status);
        MPI_Recv(Bv2[0], D2*D3, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Wait(&recv_request, &status);

        for (i = 0; i < bsize; i++)
            for (int j = 0; j < D3; j++)
                for (int k = 0; k < D2; k++)
                    C[i][j] += A[i][k] * B[k][j];
        
        MPI_Isend(&bpos, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, &status);
        MPI_Isend(&bsize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, &status);
        MPI_Isend(Cv2[0], bsize*D3, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &send_request);


    }

//********VARIANTE 3 scatter ***********************************************************************
/*
    float **sendA, **sendB, **recvC, **recvA, **finalC; // Matrizen

        printf("MatMult - NICHT BLOCKIEREND - started with %d tasks.\n", numtasks);
        sendA = alloc_mat(D1, D2); init_mat(sendA, D1, D2); // Speicher für Matrizen holen
        sendB = alloc_mat(D2, D3); init_mat(sendB, D2, D3); // und initialisieren
        recvC = alloc_mat(D1, D3);

    double start = MPI_Wtime(); 
        MPI_Scatter( sendA,D1*D2/numworkers,MPI_INT,recvA, D1*D2/numworkers, MPI_INT, 0, MPI_COMM_WORLD); //Split A into rows
        //send B to all A rows
        MPI_Bcast (sendB, D2*D3, MPI_INT, 0, MPI_COMM_WORLD) ;
		//matrix multiplication
            for (i = 0; i < bsize; i++)
            for (int j = 0; j < D3; j++)
                for (int k = 0; k < D2; k++)
                    recvC[i][j] += sendA[i][k] * sendB[k][j];                    
	
		
	// sammle alle Resultate von den buffern wieder ein und füge diese zusammen
        MPI_Gather( recvC , D1*D3/numworkers , MPI_INT , finalC , D1*D3/numworkers , MPI_INT , 0, MPI_COMM_WORLD ) ;
		
	MPI_Barrier(MPI_COMM_WORLD); 
     
    
    printf("\nUsed %f seconds.\n", MPI_Wtime() - start); // Zeitmessung anhalten

*/


    MPI_Finalize();

}

/*
blockierend:
es wird erst gewartet, bis der gesendete Prozess angekommen ist.

nicht-blockierend:
Nach dem Senden können weitere Arbeiten erledigt werden . Später kann die erfolgreiche Übermittlung geprüft
oder auf deren Abschluss gewartet werden


Scatter:
Teilt die Elementen aus einem Datenpuffer an verschiedene Prozesse auf. Gather sammelt diese dann wieder
ein und fügt diese zusammen.

Laufzeitverhalten:
Bei kleinen Matrizen ist die blockierende variante schneller, während bei größeren die nicht bockierende schneller ist.
Bei kleineren Matrizen dauert die wait funktion bzw. das organisieren dafür deutlich länger in Relation zu der eigentlichen
Berechnung. Erst bei größeren (ca 2000x2000) Matrizen wird die blockierende Variante langsamer als die nicht-blockierende.

Scatter teilt die Berechnungen auf die verschiedenen Prozesse auf, d.h. bei kleinere Matrizen ist diese vermutlich ebenfalls langsamer
als die blockierende. Aufgund der Pointer-Exception können wir die scatter funktion leider nicht testen. Wahrscheinlich
verhält sich die laufzeit ähnlich wie die von der nicht-blockierenden.

Beim Test mit der unterschiedlichen Anzahl von Workern ist das Resultat ziemlich vorhersehbar.
Bei einem Worker ist die Laufzeit deutlich langsamer als bei 2 oder 4 Workern, da mit mehr Workern die Matrix in 
kleinere Teile aufgeteilt werden kann und jeder Worker weniger zu rechnen hat. 






*/
