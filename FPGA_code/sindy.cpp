#include "main.h"
#include "hls_stream.h"

template <typename T>
void kernel_mmult(T a[N2], T b[N2], T out[N2]) {
L1:
  for (int m = 0; m < N; ++m) {
  L2:
    for (int n = 0; n < N; ++n) {
#pragma HLS PIPELINE II = 1
      T sum = 0;
    L3:
      for (int k = 0; k < N; ++k)
        sum += a[m * N + k] * b[k * N + n];
      out[m * N + n] = sum;
    }
  }
  return;
}

template <typename T>
void transpose_matrix(T in[N2], T out[N2]) {
L4:
  for (int i = 0; i < N; ++i) {
  L5:
    for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II = 1
      out[j * N + i] = in[i * N + j];
    }
  }
}

//// Matrix inversion using Gaussian elimination
//template <typename T>
//void invert_matrix(T in[N2], T out[N2]) {
//  T identity[N2]; // Identity matrix
//#pragma HLS ARRAY_PARTITION variable = identity factor = 16 dim = 1 cyclic
//
//  // Initialize identity matrix
//L6:
//  for (int i = 0; i < N; ++i) {
//  L7:
//    for (int j = 0; j < N; ++j) {
//#pragma HLS PIPELINE II = 1
//      identity[i * N + j] = (i == j) ? 1 : 0;
//      out[i * N + j] = in[i * N + j]; // Copy input matrix
//    }
//  }
//
//  // Perform Gaussian elimination
//L8:
//  for (int k = 0; k < N; ++k) {
//#pragma HLS PIPELINE II = 1
//    T diag_elem = out[k * N + k];
//    // Normalize the pivot row
//    for (int j = 0; j < N; ++j) {
//      out[k * N + j] /= diag_elem;
//      identity[k * N + j] /= diag_elem;
//    }
//
//    // Eliminate other rows
//  L9:
//    for (int i = 0; i < N; ++i) {
//      if (i != k) {
//        T factor = out[i * N + k];
//        for (int j = 0; j < N; ++j) {
//          out[i * N + j] -= factor * out[k * N + j];
//          identity[i * N + j] -= factor * identity[k * N + j];
//        }
//      }
//    }
//  }
//
//  // Copy the inverted matrix to output
//L10:
//  for (int i = 0; i < N; ++i) {
//  L11:
//    for (int j = 0; j < N; ++j) {
//#pragma HLS PIPELINE II = 1
//      out[i * N + j] = identity[i * N + j];
//    }
//  }
//}

extern "C" {
void matmult_accel(hls::stream<axis_t> &in, hls::stream<axis_t> &out) {
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS INTERFACE axis port = in
#pragma HLS INTERFACE axis port = out

  DataType l_A[N2];
  DataType l_B[N2];
  DataType l_C[N2];
//  DataType l_C_inv[N2]; // Inverted matrix
  DataType l_A_transposed[N2];

#pragma HLS ARRAY_PARTITION variable = l_A factor = 16 dim = 1 cyclic
#pragma HLS ARRAY_PARTITION variable = l_B factor = 16 dim = 1 block
#pragma HLS ARRAY_PARTITION variable = l_C factor = 16 dim = 1 cyclic
#pragma HLS ARRAY_PARTITION variable = l_A_transposed factor = 16 dim = 1 cyclic
//#pragma HLS ARRAY_PARTITION variable = l_C_inv factor = 16 dim = 1 cyclic

  int j_limit = 512 / DataTypeSize;
  int i_limit = N2 / j_limit;
  converter_t converter;

load_A:
  for (int i = 0; i < i_limit; i++) {
    axis_t temp = in.read();
    for (int j = 0; j < j_limit; j++) {
      int high = j * DataTypeSize + DataTypeSize - 1;
      int low = j * DataTypeSize;
      int index = i * 16 + j;

      converter.i = temp.data.range(high, low);
      l_A[index] = converter.d;
    }
  }

// Transpose matrix l_A into l_A_transposed
transpose_matrix<DataType>(l_A, l_A_transposed);

load_B:
  for (int i = 0; i < i_limit; i++) {
    axis_t temp = in.read();
    for (int j = 0; j < j_limit; j++) {
      int high = j * DataTypeSize + DataTypeSize - 1;
      int low = j * DataTypeSize;
      int index = i * 16 + j;

      converter.i = temp.data.range(high, low);
      l_B[index] = converter.d;
    }
  }

  kernel_mmult<DataType>(l_A_transposed, l_B, l_C);

//  // Invert the result matrix l_C
//  invert_matrix<DataType>(l_C, l_C_inv);

writeC:
  for (int i = 0; i < i_limit; i++) {
    axis_t temp;
    for (int j = 0; j < j_limit; j++) {
      int high = j * DataTypeSize + DataTypeSize - 1;
      int low = j * DataTypeSize;
      converter.d = l_C[i * 16 + j]; // Write the inverted matrix
      temp.data.range(high, low) = converter.i;
    }
    ap_uint<1> last = 0;
    if (i == i_limit - 1) {
      last = 1;
    }
    temp.last = last;
    temp.keep = -1; // enabling all bytes
    out.write(temp);
  }
}
}
