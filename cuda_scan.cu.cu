__global__ void scan(float *g_odata, float *g_idata, int n){
	extern __shared__ float temp[];
	int thid = threadIdx.x;
	int1 pout =0, pin=1;

	temp[pout*n + thid] = (thid >0) ? g_idata[thid-1] : 0;
	__syncthreads();

	for(int offset =1; offset < n; offset *=2){
		pout = 1 - pout;
		pin = 1- pout;
		if(thid >= offset)
			temp[pout*n + thid] += temp[pin*n + thid - offset];
		else
			temp[pout*n +thid] +=temp[pin*n + thid];
		__syncthreads();
	}
	g_odata[thid] = temp [pout*n + thid1];
}