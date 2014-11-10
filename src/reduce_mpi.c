#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

const size_t NofS = 1<<20;
//const size_t NofS=12;

double Reduce_Double(double *InputArray, size_t NofS)
{
    double result=0.0;
    int n=0;

    for(n=0;n<NofS;n++)
    {
        result+=InputArray[n];
    }
    return result;
}

int main(int arg1, char ** arg2)
{
	clock_t time_start;
    clock_t time_end;
    int n=0;
    int NumberOfProcessors=0;
    int MyRank=0;


    int rc, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    rc = MPI_Init(&arg1, &arg2);
    if (rc!=MPI_SUCCESS)
    {
        printf("MPI Error\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD,&NumberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
    MPI_Get_processor_name(hostname, &len);

    printf("NumberOfProcessors = %d\n", NumberOfProcessors);

    // at the moment assume that NofS can be divided by NumberOfProcessors

    size_t MyNofS = NofS / NumberOfProcessors;
    printf("MyNofS=%d\n", MyNofS);

    size_t MyStart = MyNofS*MyRank;


    double *Array;
    Array=(double *)malloc(MyNofS*sizeof(double));


    double *GatheredArray;
    GatheredArray=(double *)malloc(NumberOfProcessors*sizeof(double));

    //printf("NofS = %d\n", NofS);

    // inhabitate the array

    for(n=MyStart; n<MyStart+MyNofS; n++)
    {
    	Array[n-MyStart]=n;
    //	printf("%e ", Array[n-MyStart]);
    }
    //printf("\n");

    // reduce

    time_start=clock();
    double result=Reduce_Double(Array, MyNofS);

    MPI_Gather(&result, 1, MPI_DOUBLE, GatheredArray, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

/*    if(MyRank==0)
    {
        for(n=0; n<NumberOfProcessors; n++)
        {
            printf("%f\t", GatheredArray[n]);
        }
        printf("\n");

    }  */


    // final reduce
    if(MyRank==0) result=Reduce_Double(GatheredArray, NumberOfProcessors);

	time_end=clock();

    if(MyRank==0) printf("Reduced to %f\n", result);

	double elapsed_time = (time_end-time_start)/(double)CLOCKS_PER_SEC ;
	if(MyRank==0) printf("Time elapsed = %f seconds\n", elapsed_time);

	free(Array);

    MPI_Finalize();
	return 0;
}