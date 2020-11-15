#include <cstddef>
#include <stdio.h>
#include <math.h>
//#include <ap_fixed.h>
//#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>
//#include <dsp_builtins.h>


// #define CSIM_DEBUG
// #define CSIM_CMP_OUTPUT


// for .range(Hi, Lo)
#define FM_RG			7
#define FM_ACC_RG		12
#define WT_RG			10


#ifdef CSIM_DEBUG
	typedef float dtype_32_4;	//fix point
	typedef float dtype_32_25;	//fix point
	typedef float dtype_FM;	//fix point for feature map
	typedef float dtype_FM_acc;	//fix point for feature map
	typedef float dtype_FM_last;
	typedef float dtype_WT;	//fix point for weights
	typedef float dtype_32_16;
	typedef float dtype_32_10;
	typedef float dtype_32_12;
	typedef float dtype_16_6;
	typedef float dtype_16_5;
	typedef float dtype_16_4;
	typedef float dtype_16_10;

	typedef float uint8;
	typedef float uint16;
	typedef float uint128;
	typedef float uint256;
	typedef float uint512;

#else

	typedef ap_fixed<9,  3, AP_RND, AP_SAT> dtype_FM;	//fix point for feature map
	typedef ap_fixed<13, 4, AP_RND, AP_SAT> dtype_FM_acc;	//fix point for accumulation
	typedef ap_fixed<11, 4, AP_RND, AP_SAT> dtype_WT;	//fix point for weights

	typedef ap_fixed<16, 8, AP_RND, AP_SAT> dtype_16_8;
	typedef ap_fixed<16, 6, AP_RND, AP_SAT> dtype_16_6;
	typedef ap_fixed<16, 5, AP_RND, AP_SAT> dtype_16_5;
	typedef ap_fixed<16, 4, AP_RND, AP_SAT> dtype_16_4;
	typedef ap_fixed<16, 3, AP_RND, AP_SAT> dtype_16_3;
	typedef ap_fixed<16, 10, AP_RND, AP_SAT> dtype_16_10;
	typedef ap_fixed<32,16, AP_RND, AP_SAT> dtype_32_16;
	typedef ap_fixed<32,12, AP_RND, AP_SAT> dtype_32_12;
	typedef ap_fixed<32,10, AP_RND, AP_SAT> dtype_32_10;
	typedef ap_fixed<32, 4, AP_RND, AP_SAT> dtype_32_4;
	typedef ap_fixed<32, 7, AP_RND, AP_SAT> dtype_32_7;
	typedef ap_fixed<32,25, AP_RND, AP_SAT> dtype_32_25;

	typedef ap_uint<2> uint2;
	typedef ap_uint<4> uint4;
	typedef ap_uint<8> uint8;
	typedef ap_uint<16> uint16;
	typedef ap_uint<256> uint256;
	typedef ap_uint<512> uint512;


#endif


void SkyNet(	int bindata_in[3*162*324*2],

				uint512 conv_weight_1x1_all[1000][32],
				uint512 wright3x3[1000][3][3],
				uint512 bias_all[500],

				uint256* DDR3buff, // depth 524288*2
				float predict_boxes[1][5],
				int constant[1][3]
);

void convolution3x3(dtype_FM input[32][44][84],
					dtype_FM output[32][44][84],
					dtype_WT weights[32][3][3],
					dtype_WT bias[32],
					int relu
);

void CONV_1x1_bias(dtype_FM bottom[32][44][84],
			  dtype_FM_acc top[32][44][84],
			  dtype_WT weights[32][32],
			  dtype_WT bias[32],
			  int skip,
			  bool first_ci_flag=true
);




