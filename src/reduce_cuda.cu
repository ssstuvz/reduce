#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#define USE_DOUBLES

#ifdef USE_DOUBLES
typedef double my_float;
#else
typedef long long int my_float;
#endif

// convention   - any array without d_* is located on CPU
//              - any array with d_ is on device (GPU)

//const size_t NofS = 1<<20;
const size_t NofS = 1048576;
const size_t NofThreads = 1024;

//const size_t NofS=12;

my_float reduce_cpu(my_float *Array,int N)
{
   my_float result=0.0;
   for (int i=0;i<N;i++)
      result+=Array[i];
   return result;
         
}

__global__ void ReduceRalf(my_float *d_Array, my_float *d_ReducedArray, int N,int current)
{
    int my_x = threadIdx.x+blockIdx.x*blockDim.x+current;
    int tx=threadIdx.x;
    
    __shared__ my_float sm[1024];
    my_float cur=0.0;
    if (my_x<N)
        cur=d_Array[my_x];
    if (my_x+blockDim.x*gridDim.x<N)
        cur+=d_Array[my_x+blockDim.x*gridDim.x];
 
    sm[tx]=cur;
    for (int i=blockDim.x/2;i>0;i/=2)
    {
        __syncthreads();
        if (tx<i)
           sm[tx]=sm[tx]+sm[tx+i];
    }
    if (tx==0) d_ReducedArray[blockIdx.x]=sm[0];
    
}

__global__ void MyReduce(my_float *d_Array, my_float *d_ReducedArray, int NofS, int NofThreads)
{
    int my_x=threadIdx.x;
    size_t MyNofS = NofS/NofThreads; // assume this is correct
    size_t MyStart = MyNofS * my_x;
    size_t n=0;
	/*my_float Dummy=549755289600.0f;
	printf("%f\n", Dummy);*/
    my_float result=0.0f;
    for ( n=0; n<MyNofS; n++)
    {
		if((n+MyStart)>=NofS) printf("pretty bad");
        result+=d_Array[n+MyStart];
    }
	//printf("%f\n", result);
	if(my_x==1023) printf("1023 result = %f", result);    
	d_ReducedArray[my_x]=result;
}

my_float Last_Reduce(my_float *Array, int NofS)
{
    my_float result=(my_float)0.0;
	int n=0;
    for (n=0; n<NofS; n++)
    {
        result+=Array[n];
    }
    return result;
}


// serial part
my_float Reduce_Double(my_float *InputArray, size_t NofS)
{
    my_float result=(my_float)0.0;
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
    	Array[n]=(my_float)n;
    }

    // allocate memory on device
    my_float *d_Array = NULL;
    err = cudaMalloc((void **)&d_Array, NofS*sizeof(my_float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy the array to device
    err = cudaMemcpy(d_Array, Array, NofS*sizeof(my_float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    my_float *d_ReducedArray;
    err = cudaMalloc((void **)&d_ReducedArray, NofThreads*sizeof(my_float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    my_float *ReducedArray;
    ReducedArray=(my_float *)malloc(NofThreads*sizeof(my_float));

    cudaEvent_t start,end; 
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    

    // reduce

//    time_start=clock();
    cudaEventRecord(start);
//    MyReduce<<<1,NofThreads>>>(d_Array, d_ReducedArray, NofS, NofThreads);
    ReduceRalf<<<1024,NofThreads>>>(d_Array,d_ReducedArray,NofS,0);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    err = cudaMemcpy(ReducedArray, d_ReducedArray, NofThreads*sizeof(my_float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	my_float result=Last_Reduce(ReducedArray, NofThreads);

	time_end=clock();

	FILE * File2Save = fopen("data.dat", "wb");

	fwrite(ReducedArray, NofThreads, sizeof(my_float), File2Save);
	fclose(File2Save);

	float elapsed_time;//(time_end-time_start)/(my_float)CLOCKS_PER_SEC ;
        cudaEventElapsedTime(&elapsed_time,start,end);
    printf("Time elapsed = %f mseconds\n", elapsed_time);
	printf("Temp value = %f\n", (float)ReducedArray[3]);
	printf("Reduced to %f\n", (float)result);

	my_float Dummy=reduce_cpu(Array,NofS);
	printf("%f\n", (float)Dummy);

	free(Array);
    cudaFree(d_Array);
    cudaFree(d_ReducedArray);
    free(ReducedArray);

	return 0;
}
