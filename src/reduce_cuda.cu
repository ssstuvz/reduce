#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef USE_DOUBLES
typedef double my_float;
#else
typedef float my_float;
#endif

// convention   - any array without d_* is located on CPU
//              - any array with d_ is on device (GPU)

const size_t NofS = 1<<20;
const size_t NofThreads = 1024;

//const size_t NofS=12;

__global__ void MyReduce(float *d_Array, float *d_ReducedArray, int NofS, int NofThreads)
{
    int my_x=threadIdx.x;
    int MyNofS = NofS/NofThreads; // assume this is correct
    int MyStart = MyNofS * my_x;
    int n=0;

    my_float result=0.0;
    for (int n=0; n<MyNofS; n++)
    {
        result+=d_Array[n+MyStart];
    }
    d_ReducedArray[my_x]=result;
}

// serial part
my_float Reduce_Double(my_float *InputArray, size_t NofS)
{
    my_float result=0.0;
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
    cudaError_t err = cudaSuccess;
    
    my_float *Array;
    Array=(my_float *)malloc(NofS*sizeof(my_float));

    // inhabitate the array

    for(n=0; n<NofS; n++)
    {
    	Array[n]=n;
    }

    // allocate memory on device
    my_float *d_Array = NULL;
    err = cudaMalloc((void **)&d_Array, NofS);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy the array to device
    err = cudaMemcpy(d_Array, Array, NofS, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    my_float *d_ReducedArray;
    err = cudaMalloc((void **)&d_ReducedArray, NofThreads);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    my_float *ReducedArray;
    ReducedArray=(my_float *)malloc(NofThreads*sizeof(my_float));

    

    // reduce

    time_start=clock();
    MyReduce<<<1,1024>>>(d_Array, d_ReducedArray, NofS, NofThreads);
    err = cudaMemcpy(ReducedArray, d_ReducedArray, NofThreads, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



	time_end=clock();


	my_float elapsed_time = (time_end-time_start)/(my_float)CLOCKS_PER_SEC ;
    printf("Time elapsed = %f seconds\n", elapsed_time);

	free(Array);
    cudaFree(d_Array);
    cudaFree(d_ReducedArray);
    free(ReducedArray);

	return 0;
}