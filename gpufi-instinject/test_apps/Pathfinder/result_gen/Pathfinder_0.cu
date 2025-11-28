#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0, d2h_time = 0, close_time = 0, total_time = 0;

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define M_SEED 3415

//#define BENCH_PRINT

void run(int argc, char **argv);

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;

// Input generator integrated from pathfinder_gen_input_1.cu
static void generate_input_1(int argc, char **argv) {
    if (argc == 4) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
    } else {
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(0);
    }

    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    // Generate inputs directly instead of reading from a file
    srand(M_SEED);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = rand() % 10;
        }
    }
}

void init(int argc, char **argv) {
    // Call the integrated input generator
    generate_input_1(argc, argv);
}

void fatal(char *s) {
    fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols, int rows,
                               int startStep, int border) {
    asm volatile (
    "// inline asm\n\t"
    "	.reg .pred 	%p<19>;\n\t"
    "		.reg .b16 	%rs<9>;\n\t"
    "		.reg .b32 	%r<122>;\n\t"
    "		.reg .b64 	%rd<13>;\n\t"
    "		// demoted variable\n\t"
    "		.shared .align 4 .b8 _ZZ14dynproc_kerneliPiS_S_iiiiE4prev[1024];\n\t"
    "		// demoted variable\n\t"
    "		.shared .align 4 .b8 _ZZ14dynproc_kerneliPiS_S_iiiiE6result[1024];\n\t"
    "		ld.param.u32 	%r11, [_Z14dynproc_kerneliPiS_S_iiii_param_0];\n\t"
    "		ld.param.u64 	%rd1, [_Z14dynproc_kerneliPiS_S_iiii_param_1];\n\t"
    "		ld.param.u64 	%rd2, [_Z14dynproc_kerneliPiS_S_iiii_param_2];\n\t"
    "		ld.param.u64 	%rd3, [_Z14dynproc_kerneliPiS_S_iiii_param_3];\n\t"
    "		ld.param.u32 	%r12, [_Z14dynproc_kerneliPiS_S_iiii_param_4];\n\t"
    "		ld.param.u32 	%r13, [_Z14dynproc_kerneliPiS_S_iiii_param_6];\n\t"
    "		ld.param.u32 	%r14, [_Z14dynproc_kerneliPiS_S_iiii_param_7];\n\t"
    "		mov.u32 	%r15, %ctaid.x;\n\t"
    "		shl.b32 	%r16, %r11, 1;\n\t"
    "		mov.u32 	%r17, 256;\n\t"
    "		sub.s32 	%r18, %r17, %r16;\n\t"
    "		mul.lo.s32 	%r19, %r15, %r18;\n\t"
    "		sub.s32 	%r20, %r19, %r14;\n\t"
    "		shr.s32 	%r21, %r20, 31;\n\t"
    "		mov.u32 	%r22, %tid.x;\n\t"
    "		add.s32 	%r23, %r20, %r22;\n\t"
    "		neg.s32 	%r24, %r20;\n\t"
    "		and.b32  	%r25, %r21, %r24;\n\t"
    "		add.s32 	%r26, %r12, -1;\n\t"
    "		add.s32 	%r27, %r22, -1;\n\t"
    "		max.s32 	%r1, %r25, %r27;\n\t"
    "		setp.gt.s32	%p1, %r23, -1;\n\t"
    "		setp.le.s32	%p2, %r23, %r26;\n\t"
    "		and.pred  	%p3, %p1, %p2;\n\t"
    "		@!%p3 bra 	BB0_2;\n\t"
    "		bra.uni 	BB0_1;\n\t"
    "	BB0_1:\n\t"
    "		cvta.to.global.u64 	%rd4, %rd2;\n\t"
    "		mul.wide.s32 	%rd5, %r23, 4;\n\t"
    "		add.s64 	%rd6, %rd4, %rd5;\n\t"
    "		ld.global.u32 	%r36, [%rd6];\n\t"
    "		shl.b32 	%r37, %r22, 2;\n\t"
    "		mov.u32 	%r38, _ZZ14dynproc_kerneliPiS_S_iiiiE4prev;\n\t"
    "		add.s32 	%r39, %r38, %r37;\n\t"
    "		st.shared.u32 	[%r39], %r36;\n\t"
    "	BB0_2:\n\t"
    "		bar.sync 	0;\n\t"
    "		setp.lt.s32	%p4, %r11, 1;\n\t"
    "		@%p4 bra 	BB0_10;\n\t"
    "		shl.b32 	%r41, %r1, 2;\n\t"
    "		mov.u32 	%r42, _ZZ14dynproc_kerneliPiS_S_iiiiE4prev;\n\t"
    "		add.s32 	%r2, %r42, %r41;\n\t"
    "		mov.u32 	%r45, 1;\n\t"
    "		sub.s32 	%r120, %r45, %r11;\n\t"
    "		mad.lo.s32 	%r46, %r13, %r12, %r22;\n\t"
    "		mad.lo.s32 	%r50, %r15, %r18, %r46;\n\t"
    "		sub.s32 	%r119, %r50, %r14;\n\t"
    "		mov.u32 	%r121, 0;\n\t"
    "		cvta.to.global.u64 	%rd7, %rd1;\n\t"
    "	BB0_4:\n\t"
    "		mov.u32 	%r52, 254;\n\t"
    "		sub.s32 	%r53, %r52, %r121;\n\t"
    "		setp.le.s32	%p5, %r22, %r53;\n\t"
    "		add.s32 	%r121, %r121, 1;\n\t"
    "		setp.ge.s32	%p6, %r22, %r121;\n\t"
    "		and.pred  	%p7, %p5, %p6;\n\t"
    "		add.s32 	%r61, %r20, 255;\n\t"
    "		setp.gt.s32	%p8, %r61, %r26;\n\t"
    "		mov.u32 	%r63, -255;\n\t"
    "		sub.s32 	%r64, %r63, %r20;\n\t"
    "		add.s32 	%r65, %r12, %r64;\n\t"
    "		add.s32 	%r66, %r65, 254;\n\t"
    "		selp.b32	%r67, %r66, 255, %p8;\n\t"
    "		setp.le.s32	%p9, %r22, %r67;\n\t"
    "		setp.ge.s32	%p10, %r22, %r25;\n\t"
    "		and.pred  	%p11, %p9, %p10;\n\t"
    "		and.pred  	%p12, %p7, %p11;\n\t"
    "		mov.u16 	%rs8, 0;\n\t"
    "		@!%p12 bra 	BB0_6;\n\t"
    "		bra.uni 	BB0_5;\n\t"
    "	BB0_5:\n\t"
    "		ld.shared.u32 	%r70, [%r2];\n\t"
    "		shl.b32 	%r72, %r22, 2;\n\t"
    "		add.s32 	%r74, %r42, %r72;\n\t"
    "		setp.lt.s32	%p14, %r22, %r67;\n\t"
    "		add.s32 	%r88, %r22, 1;\n\t"
    "		selp.b32	%r89, %r88, %r67, %p14;\n\t"
    "		shl.b32 	%r90, %r89, 2;\n\t"
    "		add.s32 	%r91, %r42, %r90;\n\t"
    "		ld.shared.u32 	%r92, [%r74];\n\t"
    "		min.s32 	%r93, %r92, %r70;\n\t"
    "		ld.shared.u32 	%r94, [%r91];\n\t"
    "		min.s32 	%r95, %r94, %r93;\n\t"
    "		mul.wide.s32 	%rd8, %r119, 4;\n\t"
    "		add.s64 	%rd9, %rd7, %rd8;\n\t"
    "		ld.global.u32 	%r96, [%rd9];\n\t"
    "		add.s32 	%r97, %r95, %r96;\n\t"
    "		mov.u32 	%r98, _ZZ14dynproc_kerneliPiS_S_iiiiE6result;\n\t"
    "		add.s32 	%r99, %r98, %r72;\n\t"
    "		st.shared.u32 	[%r99], %r97;\n\t"
    "		mov.u16 	%rs8, 1;\n\t"
    "	BB0_6:\n\t"
    "		bar.sync 	0;\n\t"
    "		setp.eq.s32	%p15, %r120, 0;\n\t"
    "		@%p15 bra 	BB0_10;\n\t"
    "		setp.eq.s16	%p16, %rs8, 0;\n\t"
    "		@%p16 bra 	BB0_9;\n\t"
    "		shl.b32 	%r101, %r22, 2;\n\t"
    "		mov.u32 	%r102, _ZZ14dynproc_kerneliPiS_S_iiiiE6result;\n\t"
    "		add.s32 	%r103, %r102, %r101;\n\t"
    "		ld.shared.u32 	%r104, [%r103];\n\t"
    "		add.s32 	%r106, %r42, %r101;\n\t"
    "		st.shared.u32 	[%r106], %r104;\n\t"
    "	BB0_9:\n\t"
    "		bar.sync 	0;\n\t"
    "		add.s32 	%r120, %r120, 1;\n\t"
    "		add.s32 	%r119, %r119, %r12;\n\t"
    "		setp.lt.s32	%p17, %r121, %r11;\n\t"
    "		@%p17 bra 	BB0_4;\n\t"
    "	BB0_10:\n\t"
    "		and.b16  	%rs6, %rs8, 255;\n\t"
    "		setp.eq.s16	%p18, %rs6, 0;\n\t"
    "		@%p18 bra 	BB0_12;\n\t"
    "		shl.b32 	%r108, %r22, 2;\n\t"
    "		mov.u32 	%r109, _ZZ14dynproc_kerneliPiS_S_iiiiE6result;\n\t"
    "		add.s32 	%r110, %r109, %r108;\n\t"
    "		ld.shared.u32 	%r111, [%r110];\n\t"
    "		cvta.to.global.u64 	%rd10, %rd3;\n\t"
    "		mul.wide.s32 	%rd11, %r23, 4;\n\t"
    "		add.s64 	%rd12, %rd10, %rd11;\n\t"
    "		st.global.u32 	[%rd12], %r111;\n\t"
    "	BB0_12:\n\t"
    "		ret;\n\t"
    );
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, int pyramid_height, int blockCols, int borderCols) {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        dynproc_kernel<<<dimGrid, dimBlock>>>(MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
                                              gpuResult[dst], cols, rows, t, borderCols);

        // for the measurement fairness
        cudaDeviceSynchronize();
    }
    return dst;
}

int main(int argc, char **argv) {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1)
        cudaSetDevice(DEVICE);

    run(argc, argv);

    return EXIT_SUCCESS;
}

void run(int argc, char **argv) {
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    int *gpuWall, *gpuResult[2];
    int size = rows * cols;

    cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols);
    cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols));
    cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols), cudaMemcpyHostToDevice);

#ifdef TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

#ifdef TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols, cudaMemcpyDeviceToHost);
    // output result array to console instead of txt file
    for (int i = 0; i < cols; ++i) {
        printf("%d%c", result[i], (i == cols - 1) ? '\n' : ' ');
    }

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete[] data;
    delete[] wall;
    delete[] result;
}