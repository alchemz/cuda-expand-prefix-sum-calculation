// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
    // Load a segment of the input vector into shared memory
    __shared__ float scan_array[BLOCK_SIZE << 1];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       scan_array[t] = input[start + t];
    else
       scan_array[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       scan_array[BLOCK_SIZE + t] = 0;

    // Reduction
    int stride;
    for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
       int index = (t + 1) * stride * 2 - 1;
       if (index < 2 * BLOCK_SIZE)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
       int index = (t + 1) * stride * 2 - 1;
       if (index + stride < 2 * BLOCK_SIZE)
          scan_array[index + stride] += scan_array[index];
       __syncthreads();
    }

    if (start + t < len)
       output[start + t] = scan_array[t];
    if (start + BLOCK_SIZE + t < len)
       output[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    cudaHostAlloc(&hostOutput, numElements * sizeof(float), cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int numBlocks = ceil((float)numElements/(BLOCK_SIZE<<1));
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    wbLog(TRACE, "The number of blocks is ", numBlocks);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    cudaFreeHost(hostOutput);

    return 0;
}