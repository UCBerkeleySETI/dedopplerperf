//-------------------------------------------------------------------------------------------------------------------------------------
//First taylor tree tests on CUDA  (C) F.Antonio 2/15/2022
//   I tested on Windows/VisualStudio2022/CUDA 11.6/RTX A4000 card.  Should compile on Linux, as I avoided Windows specific APIs.
// ------------------------------------------------------------------------------------------------------------------------------------
// 2/15/22 - fa - started
// 2/19/22 - fa - got the unrolled kernels to work
// 2/21/22 - fa - removed cooperative group sync, changed to layer-by-layer launch.  Multiple launch cost is well hidden by the huge parallelism
//                due to large value of Nfreq.
//                fastest kernel is now 230ms for 2^10 x 2*20 floats on RTX A4000 6144 cores 1560 MHz.
// 2/23/22 - fa - cleanup prior to sending copy to Andrew Siemion
// 3/8/22  - fa - work on the "shoulder" issues.  Code works, but I need to redo indexing so shoulder is inside Nfreq, and works both directions.
// 3/23/22 - fa - change definition of Nfreq, so that it includes the shoulder.  I note the unrolled kernels are no longer faster, unsure why.
// 3/25/22 - fa - fork demo (simlified) version to send to Danny Price
//                speed today on home computer: 2^9 times x 2^20 freq = 9.0 sec on CPU (W-1290P), 0.120 sec on GPU (RTX A4000)
// 
// todo...
// 
// 2. Making this handle negative drift slopes would require...
//   a. duplicate kernels with - instead of + and other small indexing changes
//   b. rework shoulder logic, as negative slopes require the shoulder on the left
//   c. need to understand better how this would be used.  do we copy again from CPU, or retain a copy of raw data in GPU, or what?
// 
// 3. Test time impact of making Ntime, Nfreq variables rather than constants.  Requires adding parameters to kernels, etc.  It is possible that added indexing arithmetic
//   would be hidden by memory queueing, or it might slow down.  Seems like a requirement to be part of a flexible program.
// 
// 4. 
// 
//------------------------------------------------------------------------------------------------------------------------------------------------
// The Taylor algorithm can compute drift sums correctly only for drifts that begin AND end within the range of frequencies given to the algorithm.
// In other words, given frequencies [0,M-1], we cannot compute the drift that begins at M-1 and ends at M+37, simply because the algorithm does not
// have the data.  Therefore, when computing positive drift rates, you should consider the right M columns of the array to be a "shoulder" in which
// correct drift rates cannot be computed.  As a result, subsequent calls to cover additional frequencies must be overlapped in frequency.
// 
// It is easy to process M=1 million frequencies on standard GPUs, but it is not necessary to process all frequencies at once.  You can process the frequency
// dimension in chunks, as long as you overlap the chunks by N samples (ie the shoulder).  These facts are no surprise to anyone who has used the Taylor
// algorithm.
//  
// In all cases, the time frequency data is stored in a one dimensional array of float.  The frequency dimension increments fastest.
// The addressing equation is time*N+freq.  My code uses a C macro to do this calculation and index the array, but this is optional for the caller.
// 
// I've also included a CPU_taylor() routine, which performs the same algorithm on the CPU.  It produces identical results within the expectations of IEEE
// float precision but is 75x to 100x slower.
// 
// The time dimension N should be a power of 2.  Any power of 2 will work for CPU_taylor() or d_taylor4 kernel.  When using d_taylor6, N should be >=8.
// The frequency dimension can be anything, but >10^5 is appropriate to minimize kernel launch overhead.  
// 
// Memory requirements on the GPU are for 2 buffers, so a 512 times x 1,048,576 frequencies requires about 4.3 GB of GPU memory.  I used a two buffer
// implementation rather than in-place, because it is faster.  The 2nd buffer is contained on the GPU side, with no implications on the CPU side
// except that the 2nd buffer on the GPU side must be allocated.
// 
//-------------------------------------------------------------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cuda_fp16.h>                                              //experimental feature study
#include "device_launch_parameters.h"                               //not essential.  helps reduce some visual studio meaningless error messages
#include <math.h>
#include <stdlib.h>                                                 //gets me rand(),srand()
#include <stdio.h>
#include <time.h>                                                   //gets me clock()
#include <string>                                                   //std string library
using namespace std;


#define Atype float                                                 //type of the big array -- float or double


void setAtestvals(Atype* A);
void setAtestvals3(Atype* A);
void setAtestvals4(Atype* A);
void set_A_test_vals_matching_python(Atype* A);
void compareA(Atype* A1, Atype* A2, int Nfreqlimit);
void CPU_taylor(Atype* A);
Atype* do_taylor_tree_on_GPU(string kname, Atype* d_A, Atype* d_B);


__global__ void d_taylor4shoulder(Atype* A, Atype* B, int kmin, int kmax, int setsize);
__global__ void d_nothing();

__host__ __device__ void bufswap(Atype** A, Atype** B);
__host__ __device__ void showA(Atype* A);
__host__ __device__ void showA(Atype* A, int t0, int f0);

void findstepsA(Atype* A);

#define Ce(x)  {    \
    cudaError cerrno = (x); \
    if (cerrno != cudaSuccess) {   \
            fprintf(stderr, "*** %s  %s line %d  \n", cudaGetErrorName(cerrno), __FILE__, __LINE__);      \
            exit(EXIT_FAILURE);   \
        }    \
    }


//Dimensions of the Time/Frequency array.
//It is important that these are constants, as that allows a lot of index arithmetic to be done at compile time

// Set Ntime, Nfreq, to match Python
const int Ntime = 256;
const int Nfreq = 1 << 19;

// Create test values the same way that the "value" function in turboseti_wrapper.py
// does it.
Atype get_test_value(int time, int freq) {
  return ((time * freq) % 1337) * 0.123;
}

/*
const int Ntime = 512;             // in this version, must be a power of TWO
const int Nfreq = (1 << 20) + 512 + 8; // not constrained to power of 2, minimum is Ntime, max depends on your memory
*/

//For the CPU version of the kernel, Nfreq should NOT be a power of 2, because this makes the kernel run 6x slower due to cache contention
//in columns of the array.  It should be at least "4" away from a power of 2, or maybe a bit more on some CPU models.  Anything above "4" seems
//to produce results within 20% of best.  For My XEON W-1290P, 4,8,16,32,128 all seem to be optimal.  512 is a bit slower, but 512+8 is fast again.
//In practice you need a "shoulder" inside Nfreq anyway, so it is natural to add some to Nfreq, rather than use the power-of-2 #frequencies that come
//out of an FFT.  
//The GPU version is unaffected by this issue.

// Number of elements in the time/frequency array.  the "shoulder" is contained within Nfreq
const int Asize = Ntime * Nfreq;                                          //

// define how 2D indexing works
#define A(j,k) (A[(j)*(Nfreq)+(k)])                                       //frequency index fast, time index slow
//#define A(j,k) (A[(j)+(k)*(Ntime)])
#define B(j,k) (B[(j)*(Nfreq)+(k)])
//#define B(j,k) (B[(j)+(k)*(Ntime)])


const bool debugshowA = true;
const bool debugtimers = true;

Atype TestVals[10][10] = {
    {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f},                             //simple test values
    {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f},                             //diagonal sum should be 1.2345678
    {0.01f,0.02f,0.03f,0.04f,0.05f,0.06f,0.07f,0.08f,0.09f},
    {0.001f,0.002f,0.003f,0.004f,0.005f,0.006f,0.007f,0.008f,0.009f},
    {0.0001f,0.0002f,0.0003f,0.0004f,0.0005f,0.0006f,0.0007f,0.0008f,0.0009f},
    {0.00001f,0.00002f,0.00003f,0.00004f,0.00005f,0.00006f,0.00007f,0.00008f,0.00009f},
    {0.000001f,0.000002f,0.000003f,0.000004f,0.000005f,0.000006f,0.000007f,0.000008f,0.000009f},
    {0.0000001f,0.0000002f,0.0000003f,0.0000004f,0.0000005f,0.0000006f,0.0000007f,0.0000008f,0.0000009f}
    };

//These are properties of each kernel that depend on GPU capabilities.  Global for now.  Do something better later.
int GLOBALblocksize_d_taylor4shoulder = 0;



//-----------------------------------------------------------------------------------------------------------------------------------
// Taylor tree
// HOST (ie CPU) REFERENCE VERSION for testing/comparison
// This version is IN PLACE, without bit-reversal, which is easy on host because the freq dimension is computed serially rather than in parallel.
// Bit reversal is avoided by running the inner loop index in reverse, after which only one temporary variable is required.
// This  version of algorithm with no source code unrolling or static versions of sets, could be made faster, but its just a reference version.
// This version can work near the shoulder
// the "if" in the inner loop looks like it would slow this down, but measurement shows it adds no time.
//------------------------------------------------------------------------------------------------------------------------------------
void CPU_taylorshoulder(Atype* A) {
    for (int setsize = 2; setsize <= Ntime; setsize *= 2) {
        clock_t ct0 = clock();
        for (int j = 0; j < Ntime; j += setsize) {
            for (int k = 0; k < Nfreq; k++) {                 //+Ntime
                Atype tempx = A(j + setsize / 2, k + 0);
                for (int j0 = setsize - 1; j0 > 0; j0--) {
                    int j1 = j0 / 2;
                    int j2 = j1 + setsize / 2;
                    int j3 = (j0 + 1) / 2;
                    if (k + j3 < Nfreq) A(j + j0, k) = A(j + j1, k) + A(j + j2, k + j3);
                    }
                A(j + 0, k) = A(j + 0, k) + tempx;
                }
            }
        }
    }






//--------------------------------------------------------------------------------------------------------------------------------------------------
// Simple CUDA Kernel for Taylor Tree - no unrolls, all index calc's explicit
// DEVICE (ie GPU) REFERENCE VERSION 
// 2/21/22 updated for layer-by-layer launch
// This version can handle working within the shoulder
// The "if" stmt is completely hidden by memory latency or maybe branch prediction.  Adds no  time.
//---------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void d_taylor4shoulder(Atype* A, Atype* B, int kmin, int kmax, int setsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                     // Calculate a global thread ID
    int k = kmin + tid;                                                   // for (int k = kmin; k < kmax - 1; k++) in parallel across threads
    bool worker = (k >= kmin) && (k < kmax) && setsize <= Ntime;
//    for (int setsize = 2; setsize <= Ntime; setsize *= 2) {            //loop over layers
    if (worker) {
        for (int j = 0; j < Ntime; j += setsize) {                       //loop over sets
            for (int j0 = setsize - 1; j0 >= 0; j0--) {
                int j1 = j0 / 2;
                int j2 = j1 + setsize / 2;
                int j3 = (j0 + 1) / 2;
                if (k + j3 < kmax) B(j + j0, k) = A(j + j1, k) + A(j + j2, k + j3);
                }
            }
        }
//        }
    }



// do nothing test kernel
__global__ void d_nothing() {
    //literally do nothing 
    }


// swap the pointers to the two buffers
__host__ __device__ __forceinline__ void bufswap(Atype** A, Atype** B) {
    Atype* temp = *A;
    *A = *B;
    *B = temp;
    return;
    }





//------------------------------------------------------------------------------------------------------------------
// main test program
//------------------------------------------------------------------------------------------------------------------

int main() {
    clock_t ct0, ct1, ct2, ct3, cta, ctb;

    int dev = 0;
    cudaDeviceProp deviceProp;
    Ce(cudaGetDeviceProperties(&deviceProp, dev));                          //display what kind of GPU we're testing
    printf("Device %d: %s  clock %.0f MHz\n", dev, deviceProp.name, deviceProp.clockRate * 1e-3f);

// ask CUDA how many threads per block this GPU will support for my kernels...
    int mingridsize = 0;

    Ce(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &GLOBALblocksize_d_taylor4shoulder, d_taylor4shoulder, 0, 0));
    printf("max threads/block this GPU supports for d_taylor4sh=%i\n", GLOBALblocksize_d_taylor4shoulder);




// create the time/frequency array

    printf("Ntime=%i  Nfreq=%i  Asize=%i  GPU mem used=%f GB\n", Ntime, Nfreq, Asize, (float)Asize * sizeof(Atype) * 2 / (1 << 30));
    Atype* A = new Atype[Asize];                                            //on host (so we can create some test data)                                
    Atype* d_A = 0;                                                         //double buffer on GPU
    Atype* d_B = 0;                                                         //

    memset(A, 0, Asize * sizeof(Atype));                                    //zero the big array, 

    Ce(cudaMalloc((void**)&d_A, Asize * sizeof(Atype)));                    //allocate A on GPU
    Ce(cudaMalloc((void**)&d_B, Asize * sizeof(Atype)));                    //allocate B on GPU

    setAtestvals(A);                                                        //stuff in some test values in the time/frequency array
    Ce(cudaMemcpy(d_A, A, Asize * sizeof(Atype), cudaMemcpyHostToDevice));  //copy A to device

//  test slope=1...

    ct0 = clock();
    Atype* d_return1 = do_taylor_tree_on_GPU("d_taylor4shoulder", d_A, d_B);         //launch kernel
    Ce(cudaDeviceSynchronize());                                            //wait for it to finish
    ct1 = clock();
    printf("GPU taylor tree time = %f sec\n", double(ct1 - ct0) / CLOCKS_PER_SEC);

    printf("\nTesting the slope=1 diagonal...\n");
    Ce(cudaMemcpy(A, d_return1, Asize * sizeof(Atype), cudaMemcpyDeviceToHost));
//    showA(A);
    printf("A(%i, % i) = %f  expect 1.111111\n", 0, 0, A(0, 0));
    printf("A(%i, % i) = %f  expect 1.234567\n", Ntime - 1, 0, A(Ntime - 1, 0));
    printf("but accept 1.11111x and 1.23456x (theoretical limit of float mantissa is 6.2 digits)\n\n");

    
    printf("\nTesting benchmark data\n");
    memset(A, 0, Asize * sizeof(Atype));                                    //zero the A array  
    set_A_test_vals_matching_python(A);
    printf("initial values:\n");
    showA(A);
    Ce(cudaMemcpy(d_A, A, Asize * sizeof(Atype), cudaMemcpyHostToDevice));  //copy A to device
    ct0 = clock();
    Atype* d_return2 = do_taylor_tree_on_GPU("d_taylor4shoulder", d_A, d_B);        //execute taylor tree algorithm
    Ce(cudaDeviceSynchronize());                                            //let it to finish so we get a good time measurement
    ct1 = clock();
    printf("GPU taylor tree time = %f sec\n", double(ct1 - ct0) / CLOCKS_PER_SEC);
    Ce(cudaMemcpy(A, d_return2, Asize * sizeof(Atype), cudaMemcpyDeviceToHost));
    showA(A);
    printf("\n");

    /*
    
//--------------------------------------------------------
// 3rd test.  random array, compare CPU vs GPU results.
//--------------------------------------------------------
    printf("\n\nrandom values CPU vs GPU test...\n");
    setAtestvals4(A);                                                       //fill A with random values
    showA(A);

    ct0 = clock();
    CPU_taylorshoulder(A);                                                  //taylor on CPU
    ct1 = clock();
    printf("\n CPU implementation of taylor %f seconds\n", float(ct1 - ct0) / CLOCKS_PER_SEC);

    Atype* A2 = new Atype[Asize];
    memcpy(A2, A, Asize * sizeof(Atype));                                   //put the CPU result in A2.
    printf("showing upper left corner of result array\n");
    showA(A2);
    printf("\n");

    setAtestvals4(A);                                                       //fill A with same random values
    Ce(cudaMemcpy(d_A, A, Asize * sizeof(Atype), cudaMemcpyHostToDevice));  //copy A to device

    ct0 = clock();
    Atype* d_return3 = do_taylor_tree_on_GPU("d_taylor4shoulder", d_A, d_B);    //taylor on GPU
    Ce(cudaDeviceSynchronize())                                             //let it finish so we get a good time measurement
        ct1 = clock();
    printf("GPU taylor tree time = %f sec\n", float(ct1 - ct0) / CLOCKS_PER_SEC);

    Ce(cudaMemcpy(A, d_return3, Asize * sizeof(Atype), cudaMemcpyDeviceToHost));

    printf("showing upper left corner of result array\n");
    showA(A);
    printf("\n");

    printf("\n");
    compareA(A, A2, Nfreq - Ntime);                                        //Don't compare in the shoulder
    printf("\n");

    // more carefully time the kernel
    ct0 = clock();
    for (int k = 0; k < 20; k++) {
        Atype* d_return3 = do_taylor_tree_on_GPU("d_taylor4shoulder", d_A, d_B);     //taylor on GPU
        }
    Ce(cudaDeviceSynchronize());
    //let it finish so we get a good time measurement
    ct1 = clock();
    printf("GPU taylor tree time (more carefully measured, in a loop) = %f sec\n", float(ct1 - ct0) / CLOCKS_PER_SEC / 20.0);

    */

    /*
    ct0 = clock();
    for (int k = 0; k < 1000; k++) {
        d_nothing << <1000000 / 768, 768 >> > ();
        }
    Ce(cudaDeviceSynchronize());
    ct1 = clock();
    printf("empty kernel time =%f\n", float(ct1 - ct0) / CLOCKS_PER_SEC / 1000);
    */

    //printf("find steps A2\n");
    //findstepsA(A2);
    //printf("find steps A\n");
    //findstepsA(A);


    Ce(cudaFree(d_A));
    Ce(cudaFree(d_B));
    free(A);
    // free(A2);

    return 0;
    } //end of main()



// ----------------------------------------------------------------------------------------------
// launch a taylor-tree kernel.
// ----------------------------------------------------------------------------------------------
Atype* do_taylor_tree_on_GPU(string kernelname, Atype* d_A, Atype* d_B) {

// positive vs negative frequency rate require different indexing here, shoulder on other side, etc  ****FUTURE****
    
    if (kernelname == "d_taylor4shoulder") {
        for (int ss = 2; ss <= Ntime; ss *= 2) {
            int threads = GLOBALblocksize_d_taylor4shoulder;
            d_taylor4shoulder << <(Nfreq + threads - 1) / threads, threads >> > (d_A, d_B, 0, Nfreq, ss);
            bufswap(&d_A, &d_B);
            }
        }
   
// I removed the other test versions, to make demo code simpler.

    else {
        printf("unknown kernel name in launch_taylor_tree\n");
        exit(EXIT_FAILURE);
        }

    //variable number of swaps occur, depending on kernel and Ntime, but because of the swaps above the "answer" always ends up in the
    //buffer known here as d_A.  We return this address, so the caller knows where to find the result. (It could be either buffer.)
    return d_A;
    }






// display the upper left corner of the array, for debugging.
__host__ __device__ void showA(Atype* A) {
    int jlimit = min(16, Ntime);
    for (int j = 0; j < jlimit; j++) {
        for (int k = 0; k < 8; k++) printf("%.8f  ", A(j, k));
        printf("\n");
        }
    }

// display a block from an arbitrary location in the big array, for debugging.
__host__ __device__ void showA(Atype* A, int t0, int f0) {
    printf("block starting at %i %i ...\n", t0, f0);
    int jlimit = min(16, Ntime);
    for (int j = 0; j < jlimit; j++) {
        for (int k = 0; k < 8; k++) printf("%.8f  ", A(t0 + j, f0 + k));
        printf("\n");
        }

    }


// set A from a testvals array...
void setAtestvals(Atype* A) {
    int N = min(10, Ntime);
    for (int j = 0; j < N; j++)                            //put test data into the big array
        for (int k = 0; k < N; k++)
            A(j, k) = TestVals[j][k];
    }



    // load up one particular diagonal, and see if the tree finds it
void setAtestvals3(Atype* A) {
    float f1 = 4;
    int deltaf = 5;
    double s = (double)deltaf;
    for (int time = 0; time < Ntime; time++) {
        double freq = f1 + (double)time / (Ntime - 1) * deltaf;
//        double freq = f1 + (time-0.5)*(s + 1) / Ntime  - 0.0;
        int ifreq = (int)(freq + 0.5);
        A(time, ifreq) = 1;
//        printf("%i/%i  %i %i  %f\n", deltaf, Ntime - 1, time, ifreq,freq);
        }
    }


// load up time/frequency array with random values.  
void setAtestvals4(Atype* A) {
    srand(1234);                                            //make this test repeatable (for example CPU vs GPU)
    printf("loading random values... (which is damn slow)\n");
    for (int time = 0; time < Ntime; time++) {
        for (int freq = 0; freq < Nfreq; freq++) {
            A(time, freq) = (float)rand() / 32768000.;
            }
        }
    }

void set_A_test_vals_matching_python(Atype* A) {
  for (int time = 0; time < Ntime; time++) {
    for (int freq = 0; freq < Nfreq; freq++) {
      A(time, freq) = get_test_value(time, freq);
    }
  }
}


// compare two result arrays (for example CPU vs GPU)
void compareA(Atype* A, Atype* B, int Nfreqlimit) {
    float maxerr = -1;
    int tmax = -1;
    int fmax = -1;
    for (int time = 0; time < Ntime; time++) {
        for (int freq = 0; freq < Nfreqlimit; freq++) {
            float x1 = A(time, freq);
            float x2 = B(time, freq);
            float error = abs(x1 - x2);
            if (!(error <= maxerr)) {
                maxerr = error;
                tmax = time;
                fmax = freq;
                }
            }
        }
    if (tmax == -1 || fmax == -1) printf("compareA foulup\n");
    printf("largest difference = %f at %i %i   A1=%f  A2=%f\n", maxerr, tmax, fmax, A(tmax, fmax), B(tmax, fmax));
    }


// locate "steps" in transform of a random array
void findstepsA(Atype* A) {
    for (int time = 0; time < Ntime; time += 23) {
        for (int freq = 1; freq < Nfreq; freq++) {
            float ratio = A(time, freq) / A(time, freq - 1);
            if (ratio > 2.0 && A(time, freq) > 0.1) printf("step  up  at %i %i  %f %f\n", time, freq, A(time, freq - 1), A(time, freq));
            if (ratio < 0.5 && A(time, freq - 1)>0.1) printf("step down at %i %i  %f %f\n", time, freq, A(time, freq - 1), A(time, freq));
            }
        }
    }
