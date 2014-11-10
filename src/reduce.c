#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const size_t NofS = 1<<20;

int main(int arg1, char ** arg2)
{
	clock_t time_start;
    clock_t time_end;

    int n=0;

    double *Array;

    Array=(double *)malloc(NofS*sizeof(double));

    printf("NofS = %d\n", NofS);

    // inhabitate the array

    for(n=0; n<NofS; n++)
    {
    	Array[n]=n*0.1;
    	//printf("%e ", Array[n]);
    }
    printf("\n");

    // reduce

    time_start=clock();
    double result=0.0;

    for(n=0;n<NofS;n++)
    {
    	result+=Array[n];
    }

	time_end=clock();

	printf("Reduced to %e\n", result);

	double elapsed_time = (time_end-time_start)/(double)CLOCKS_PER_SEC ;
	printf("Time elapsed = %f seconds\n", elapsed_time);

	free(Array);

	return 0;
}