

// Conv 1x1 PE

#include "main.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"

dtype_FM_acc mutiple_16(dtype_WT x0,  dtype_FM y0,
					  dtype_WT x1,  dtype_FM y1,
					  dtype_WT x2,  dtype_FM y2,
					  dtype_WT x3,  dtype_FM y3,
					  dtype_WT x4,  dtype_FM y4,
					  dtype_WT x5,  dtype_FM y5,
					  dtype_WT x6,  dtype_FM y6,
					  dtype_WT x7,  dtype_FM y7,
					  dtype_WT x8,  dtype_FM y8,
					  dtype_WT x9,  dtype_FM y9,
					  dtype_WT x10, dtype_FM y10,
					  dtype_WT x11, dtype_FM y11,
					  dtype_WT x12, dtype_FM y12,
					  dtype_WT x13, dtype_FM y13,
					  dtype_WT x14, dtype_FM y14,
					  dtype_WT x15, dtype_FM y15)
{
	dtype_32_10 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	dtype_32_10 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	dtype_32_10 add0, add1, add2, add3,  add4,  add5,  add6;
	dtype_32_10 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = x0  * y0;
	mul1  = x1  * y1;
	mul2  = x2  * y2;
	mul3  = x3  * y3;
	mul4  = x4  * y4;
	mul5  = x5  * y5;
	mul6  = x6  * y6;
	mul7  = x7  * y7;
	mul8  = x8  * y8;
	mul9  = x9  * y9;
	mul10 = x10 * y10;
	mul11 = x11 * y11;
	mul12 = x12 * y12;
	mul13 = x13 * y13;
	mul14 = x14 * y14;
	mul15 = x15 * y15;

	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = add0 + add1;
	add9  = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8  + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;

}




void load_weights(dtype_WT weights[32][32], dtype_WT wrightbuff[32][16], int CI)
{
	for(int ci = 0; ci < 16; ci++) {
#pragma HLS pipeline


		for(int co = 0; co < 32; co++) {
		#pragma HLS unroll
			
			for(int i=0;i<15;i++)
			{
				#pragma HLS unroll
				wrightbuff[co][i]=wrightbuff[co][i+1];
			}
			wrightbuff[co][15]=weights[co][ci + CI];
		}
	}
}


void convolution1x1(dtype_FM input[32][44][84],
			  dtype_FM_acc output[32][44][84],
			  dtype_xT weights[32][32],
			  dtype_xT bias[32],
			  int skip,
			  bool first)
{
	#pragma HLS array_partition variable=output dim=1 complete
	#pragma HLS array_partition variable=input dim=1 complete
	#pragma HLS array_partition variable=xeights dim=1 complete
	#pragma HLS array_partition variable=bias dim=1 complete
	dtype_WT wrightbuff[32][16];
	#pragma HLS array_partition variable=wrightbuff dim=1 complete
	#pragma HLS array_partition variable=wrightbuff dim=2 complete
	for(int ci = 0; ci < 2-skip; ci++) {
		load_weights(weights, wrightbuff, ci*16);
		int offset = ci*16;
		for(int h = 1; h <= 42; h++){
			for(int x = 1; x <= 82; x++) {
			#pragma HLS pipeline II=2

				for(int coo = 0; coo < 32; coo++) {
				#pragma HLS unroll
					dtype_FM_acc residual;
					if(ci==0 && first)
						residual=bias[coo];
					else
						residual=output[coo][h][x];
					output[coo][h][x] =residual + mutiple_16(
							wrightbuff[coo][0],   input[offset+0][h][x],
							wrightbuff[coo][1],   input[offset+1][h][x],
							wrightbuff[coo][2],   input[offset+2][h][x],
							wrightbuff[coo][3],   input[offset+3][h][x],
							wrightbuff[coo][4],   input[offset+4][h][x],
							wrightbuff[coo][5],   input[offset+5][h][x],
							wrightbuff[coo][6],   input[offset+6][h][x],
							wrightbuff[coo][7],   input[offset+7][h][x],
							wrightbuff[coo][8],   input[offset+8][h][x],
							wrightbuff[coo][9],   input[offset+9][h][x],
							wrightbuff[coo][10],  input[offset+10][h][x],
							wrightbuff[coo][11],  input[offset+11][h][x],
							wrightbuff[coo][12],  input[offset+12][h][x],
							wrightbuff[coo][13],  input[offset+13][h][x],
							wrightbuff[coo][14],  input[offset+14][h][x],
							wrightbuff[coo][15],  input[offset+15][h][x]);
				}
			}
		}
	}
}
