

// conv 3x3

#include "main.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"


inline dtype_FM relu( dtype_FM d ) {
        if( d > 6 )
                return 6;
        if( d < 0 )
                return 0;
        return d;
}




ap_fixed<24, 7> mul1234(dtype_WT w1, dtype_FM b1, dtype_WT w2, dtype_FM b2)
{
	#pragma HLS pipeline

	ap_fixed<24, 7> tmp =  w1 * b1 + w2 * b2;
	return tmp;

}


void convolution3x3(dtype_FM input[32][44][84],
					dtype_FM output[32][44][84],
					dtype_WT weights[32][3][3],
					dtype_WT bias[32],
					int relu
					)
{

#pragma HLS array_partition variable=output dim=1 complete
#pragma HLS array_partition variable=input dim=1 complete
#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=bias dim=1 complete


		for(int h = 1; h <= 42; h++){
			for(int w = 1; w <= 82; w++){
			#pragma HLS pipeline II=5
				for(int co = 0; co < 32; co++){
				#pragma HLS unroll
					ap_fixed<24, 7> tmp1 = mul1234(weights[co][0][0], input[co][h-1][w-1], weights[co][0][1], input[co][h-1][w  ]);
					ap_fixed<24, 7> tmp2 = mul1234(weights[co][0][2], input[co][h-1][w+1], weights[co][1][0], input[co][h  ][w-1]);
					ap_fixed<24, 7> tmp3 = mul1234(weights[co][1][1], input[co][h  ][w  ], weights[co][1][2], input[co][h  ][w+1]);
					ap_fixed<24, 7> tmp4 = mul1234(weights[co][2][0], input[co][h+1][w-1], weights[co][2][1], input[co][h+1][w  ]);
					ap_fixed<24, 7> tmp5 = mul1234(weights[co][2][2], input[co][h+1][w+1], 0, 0);
					ap_fixed<24, 7> sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
					output[co][h][w] = ((dtype_FM)bias[co])+sum;
				}
			}
		}


	if(relu == 1) {
		for(int h = 1; h <= 42; h+=2){
			for(int w = 1; w <= 82; w++){
			#pragma HLS pipeline
				for(int co = 0; co < 32; co++){
					output[co][h  ][w] = relu( output[co][h  ][w ]);
					output[co][h+1][w] = relu( output[co][h+1][w ]);
				}
			}
		}
	}
}
