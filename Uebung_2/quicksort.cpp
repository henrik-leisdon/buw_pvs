#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>

#define NUM 32767                                             // Elementanzahl

// ---------------------------------------------------------------------------
// Vertausche zwei Zahlen im Feld v an der Position i und j

void swap(float *v, int i, int j)
{
    float t = v[i]; 
    v[i] = v[j];
    v[j] = t;
}

// ---------------------------------------------------------------------------
// Serielle Version von Quicksort (Wirth) 

void quicksort(float *v, int start, int end) 
{
    int i = start, j = end;
    float pivot;

    
    pivot = v[(start + end) / 2];                         // mittleres Element
    do {
        while (v[i] < pivot)
            i++;
        while (pivot < v[j])
            j--;
        if (i <= j) {               // wenn sich beide Indizes nicht beruehren
            
            swap(v, i, j);
            i++;
            j--;
        }
   } while (i <= j);
   if (start < j)  
                                                 // Teile und herrsche
       quicksort(v, start, j);                      // Linkes Segment zerlegen
       
   if (i < end)
        
       quicksort(v, i, end);                       // Rechtes Segment zerlegen
       
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

void quicksort_01(float *v, int start, int end) 
{
    int i = start, j = end;
    float pivot;

    
    pivot = v[(start + end) / 2];                         // mittleres Element
    do {
        while (v[i] < pivot)
            i++;
        while (pivot < v[j])
            j--;
        if (i <= j) {               // wenn sich beide Indizes nicht beruehren
            
            swap(v, i, j);
            i++;
            j--;
        }
   } while (i <= j);
   #pragma omp task                            //nur die linke seite vom Pivot wird als task ausgeführt
   {   
   if (start < j)                                  //Wenn ein thread nichts zu tun hat, nimmt sich  dieser eine Task und bearbeitet diese
                                                   // Teile und herrsche
         quicksort_01(v, start, j);                      // Linkes Segment zerlegen
   }
    if (i < end)
       quicksort_01(v, i, end);                       // Rechtes Segment zerlegen
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

void quicksort_02(float *v, int start, int end) 
{
    int i = start, j = end;
    float pivot;

    
    pivot = v[(start + end) / 2];                         // mittleres Element
    do {
        while (v[i] < pivot)
            i++;
        while (pivot < v[j])
            j--;
        if (i <= j) {               // wenn sich beide Indizes nicht beruehren
            
            swap(v, i, j);
            i++;
            j--;
        }
   } while (i <= j);
   #pragma omp task
   {
   if (start < j)                                    //selbe prozedur, wie bei version 1 nur, dass dieses mal sowohl die linke,
                                                     //als auch die rechte seite als Tasks zur verfügung stehen
        quicksort_02(v, start, j);                      // Linkes Segment zerlegen
   }
   #pragma omp task
   {
   if (i < end)
       quicksort_02(v, i, end);                       // Rechtes Segment zerlegen
    }
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

void quicksort_03(float *v, int start, int end) 
{
    int i = start, j = end;
    float pivot;
    

    
    pivot = v[(start + end) / 2];                         // mittleres Element
    do {
        while (v[i] < pivot)
            i++;
        while (pivot < v[j])
            j--;
        if (i <= j) {               // wenn sich beide Indizes nicht beruehren
            
            swap(v, i, j);
            i++;
            j--;
        }
   } while (i <= j);
   #pragma omp parallel
   {
   #pragma omp  sections                            //aufteilen in Sections, jede seite ist eine Section und wird auf threads aufgeteilt
   {
       #pragma omp section 
       {
    if (start < j)                                   // Teile und herrsche
        quicksort_03(v, start, j);                      // Linkes Segment zerlegen
       }
    #pragma omp section
            {
    if (i < end)
        quicksort_03(v, i, end);                       // Rechtes Segment zerlegen
        }
   }
}
}
// ---------------------------------------------------------------------------



// Hauptprogramm

int main(int argc, char *argv[])
{
    float *v;                                                         // Feld                               
    int iter;                                                // Wiederholungen             

    if (argc != 2) {                                      // Benutzungshinweis
        printf ("Vector sorting\nUsage: %s <NumIter>\n", argv[0]); 
        return 0;
    }
    iter = atoi(argv[1]);                               
    v = (float *) calloc(NUM, sizeof(float));        // Speicher reservieren

    //v0

    printf ("ohne beschleunigen:");
    double start, end;
    start = omp_get_wtime(); 
    
    
    printf("Perform vector sorting %d times...\n", iter);
    for (int i = 0; i < iter; i++) {               // Wiederhole das Sortieren
        for (int j = 0; j < NUM; j++)      // Mit Zufallszahlen initialisieren
            v[j] = (float)rand();
        quicksort(v, 0, NUM-1);                           
    }

    end = omp_get_wtime(); 

    printf ("Benoetigte Zeit: %f Sekunden\n", end - start);

//v1

    printf ("mit einem task:");
    double start1, end1;
    start1 = omp_get_wtime(); 
    
    
    printf("Perform vector sorting %d times...\n", iter);
    for (int i = 0; i < iter; i++) {               // Wiederhole das Sortieren
        for (int j = 0; j < NUM; j++)      // Mit Zufallszahlen initialisieren
            v[j] = (float)rand();
        #pragma omp parallel 
        {
        #pragma omp single
        quicksort_01(v, 0, NUM-1);      
        }                      
    }

    end1 = omp_get_wtime(); 

    printf ("Benoetigte Zeit: %f Sekunden\n", end1 - start1);
    
// v2

printf ("2x task beschleunigen:");
    double start2, end2;
    start2 = omp_get_wtime(); 
    
    
    printf("Perform vector sorting %d times...\n", iter);
    for (int i = 0; i < iter; i++) {               // Wiederhole das Sortieren
        for (int j = 0; j < NUM; j++)      // Mit Zufallszahlen initialisieren
            v[j] = (float)rand();
        #pragma omp parallel 
        {
        #pragma omp single
        quicksort_02(v, 0, NUM-1);      
        }                      
    }

    end2 = omp_get_wtime(); 

    printf ("Benoetigte Zeit: %f Sekunden\n", end2 - start2);

//v3

printf ("sections beschleunigen:");
    double start3, end3;
    start3 = omp_get_wtime(); 
    
    
    printf("Perform vector sorting %d times...\n", iter);
    for (int i = 0; i < iter; i++) {               // Wiederhole das Sortieren
        for (int j = 0; j < NUM; j++)      // Mit Zufallszahlen initialisieren
            v[j] = (float)rand();
            
        #pragma omp parallel num_threads(4)
        {
        #pragma omp single
        quicksort_03(v, 0, NUM-1);      
                             
    }

    end3 = omp_get_wtime(); 

    printf ("Benoetigte Zeit: %f Sekunden\n", end3 - start3);

    //test funktion
    //--> erstellen von 3 weiteren Vektoren, die jeweils von quicksort 1, 2 und 3 sortiert werden sollen
    
    float *v01, *v02, *v03;
    v01 = (float *) calloc(NUM, sizeof(float));
    v02 = (float *) calloc(NUM, sizeof(float));
    v03 = (float *) calloc(NUM, sizeof(float));
    
    //füllen der vier vektoren mit gleichen zufallszahlen
    for(int i = 0; i< NUM; i++)
    {
        float value = rand();
        v[i] = value;
        v01[i] = value;
        v02[i] = value;
        v03[i] = value;
    }


//sortieren der einzelnen sortiealgorithmen
    quicksort(v, 0, NUM-1);
    quicksort_01(v01, 0, NUM-1);
    quicksort_02(v02, 0, NUM-1);
    quicksort_03(v03, 0, NUM-1);
//testen ob die jeweiligen versionen korrekt sortieren.
    bool correct1 = true;
    bool correct2 = true;
    bool correct3 = true;
    for(int i = 0; i< NUM; i++)
    {
        if(v[i]!=v01[i])
            printf("Error - Version 1");
            correct1 = false;
            //break;

        if(v[i]!=v02[i])
            printf("Error - Version 1");
            correct2 = false;
            //break;
        if(v[i]!=v03[i])
            printf("Error - Version 1");
            correct3 = false;
            //break;
    } 

    //output, welche version korrekt / nicht korrekt sortiert.
    if(correct1 == false)
        printf("\n the results in version 1 were not correct");
        else
        printf("\n the results were correct");

    if(correct2 == false)
        printf("\n the results in version 2 were not correct");
        else
        printf("\n the results were correct");

    if(correct3 == false)
        printf("\n the results in version 3 were not correct");
        else
        printf("\n the results were correct");

//laut Output ist version 3 die schnellste. Diese sortiert zwar offensichtlich richtig, aber da die normale version 7 sekunden
//braucht und diese weniger als 1 sekunde ist es wahrscheinlich, dass irgendwo ein fehler ist.
//
//version 1 sortiert langsamer als das standard quicksort. Aber dort sortiert der algorithmus offensichtlich nicht richtig, den Fehler haben wir leider nicht gefunden\\ 
//
//version 2 ist deutlich langsamer als 1 und der standard quicksort. Dafür sortiert dieser richtig. Nützen tut dies aber auch wenig, da das standard quicksort immernoch die schnellste und 
//zuverlässigste sortierung ist. (Version 3 ist wie gesagt vermutlich fehlerhaft oder sehr effizient).

    printf ("\nDone.\n");
    return 0;
}
}

