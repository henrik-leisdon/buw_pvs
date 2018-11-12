
# include <stdio.h> // includen der Standard-Library
# include <omp.h> // includen der open mp bibliothek
int main(int argc, char* argv[])
{
 int numThreads; // varibale für die anzahl der threads
 int threadID; // varibale für die nummer/ID des threads
 float start, end; // Start und Ende float-wert um die Zeit zu messen
 start = omp_get_wtime(); //Beginn der Zeitmessung
 /* höchste priorität ist der eingetragene Wert */
 #pragma omp parallel num_threads(2)
 {
 threadID = omp_get_thread_num(); //thread ID --> anwählen des threads
 printf("Hello from thread %d\n", threadID);

 /* der nte thread wird ausgegeben */
 if (threadID == 0)
 {
 numThreads = omp_get_num_threads(); //abfragen der thread Anzahl
 printf("Number of threads: %d\n", numThreads);
 }
 }
 end = omp_get_wtime(); //Ende der Zeitmessung
 printf("This task took %f seconds\n", end-start);
 return 0;
}
