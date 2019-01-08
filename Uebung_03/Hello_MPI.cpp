#include<stdio.h>
#include<mpi.h> // includiert alle MPI-Funktionen 
 int main(int argc, char** argv) {
	int nodeID, numNodes;
	
		 /* Hauptprozess: comm -> MPI-interne Datenstruktur zur Verwaltung einer Menge von Prozessen */
		 MPI_Init(&argc, &argv); // Initialisiert das MPI-Laufzeitsystem
		 MPI_Comm_size(MPI_COMM_WORLD, &numNodes); // Anzahl aller Prozesse im Kommunikator
		 MPI_Comm_rank(MPI_COMM_WORLD, &nodeID); // eigene Nummer
	
		 /* bei Ausfuehrung des Programms: Hello world from process 0 of 0 (Zählung beginnt bei null */
		 printf("Hello world from process %d of %d\n", nodeID, numNodes);
	
		 /* Meldet Prozess beim MPI-Laufzeitsystem ab. */
		 MPI_Finalize();
	
		 return 0;
	
}