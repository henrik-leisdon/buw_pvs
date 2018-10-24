
# include <stdio.h> // includen des
# include <omp.h> // includen der open mp bibliothek
int main(int argc, char* argv[])
{
 int numThreads; // anzahl der threads
 int threadID; // nummer/ID des threads
 float start, end; // Start und Ende float wert um die Zeit zu messen
 start = omp_get_wtime(); //startenb der get Time funktion
 /* ? */
 #pragma omp parallel num_threads(1)
 {
 threadID = omp_get_thread_num(); //thread ID --> anwählen des n-ten thread
 printf("Hello from thread %d\n", threadID);

 /* der nte thread wird ausgegeben */
 if (threadID == 0)
 {
 numThreads = omp_get_num_threads(); //?
 printf("Number of threads: %d\n", numThreads);
 }
 }
 end = omp_get_wtime(); //ende der Zeitmessung
 printf("This task took %f seconds\n", end-start);
 return 0;
}
