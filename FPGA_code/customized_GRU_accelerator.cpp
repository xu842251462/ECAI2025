#include "hls_stream.h"
#include "ap_int.h"
#include "math.h"
#include "main.h"

// Constants (modify as needed)
const int input_size = 7;
const int hidden_size = 64;
const int output_size = 10;
const int sequence_length = 24;


// Helper sigmoid function
DataType sigmoid(DataType x) {
    return 1 / (1 + exp(-x));
}

// Helper tanh function
DataType tanh(DataType x) {
    return 2 / (1 + exp(-2 * x)) - 1;
}

// Helper softmax function
void softmax(DataType input[output_size], DataType output[output_size]) {
    DataType max_val = input[0];
    for (int i = 1; i < output_size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    DataType sum = 0;
    for (int i = 0; i < output_size; i++) {
        sum += exp(input[i] - max_val);
    }

    for (int i = 0; i < output_size; i++) {
        output[i] = exp(input[i] - max_val) / sum;
    }
}


template <typename T> void forward_acc(T a[N2], T b[N2], T d[N2], T out[N2]) {

	// Declare weights and biases (dummy initialization)
		DataType Wr[hidden_size * (hidden_size + input_size)];
		DataType Wz[hidden_size * (hidden_size + input_size)];
		DataType Wa[hidden_size * (hidden_size + input_size)];
		DataType Wy[output_size * hidden_size];

		DataType br[hidden_size];
		DataType bz[hidden_size];
		DataType ba[hidden_size];
		DataType by[output_size];

		// Declare buffers for intermediate results
		DataType X[sequence_length][input_size];
		DataType r[sequence_length][hidden_size];
		DataType z[sequence_length][hidden_size];
		DataType c[sequence_length][hidden_size];
		DataType cc[sequence_length][hidden_size];
		DataType y_pred[sequence_length][output_size];

		// Initialize hidden state and cell state
		DataType a_prev[hidden_size];
		DataType c_prev[hidden_size];

	// Forward pass over the sequence
		forward_pass:
		for (int t = 0; t < sequence_length; ++t) {
		#pragma HLS PIPELINE II=1
			// Concatenate hidden state and input
			DataType concat[hidden_size + input_size];

			for (int i = 0; i < hidden_size; ++i) {
				concat[i] = a_prev[i];
			}
			for (int i = 0; i < input_size; ++i) {
				concat[hidden_size + i] = X[t][i];
			}

			// Compute reset gate r[t]
			for (int i = 0; i < hidden_size; ++i) {
				r[t][i] = sigmoid(Wr[i] * concat[i] + br[i]);
			}

			// Compute update gate z[t]
			for (int i = 0; i < hidden_size; ++i) {
				z[t][i] = sigmoid(Wz[i] * concat[i] + bz[i]);
			}

			// Compute candidate cell state cc[t]
			for (int i = 0; i < hidden_size; ++i) {
				cc[t][i] = tanh(Wa[i] * concat[i] * r[t][i] + ba[i]);
			}

			// Update cell state c[t]
			for (int i = 0; i < hidden_size; ++i) {
				c[t][i] = z[t][i] * cc[t][i] + (1 - z[t][i]) * c_prev[i];
			}

			// Update hidden state a[t]
			for (int i = 0; i < hidden_size; ++i) {
				a[t][i] = c[t][i];
			}

			// Compute output y_pred[t]
			DataType raw_output[output_size];
			for (int i = 0; i < output_size; ++i) {
				raw_output[i] = 0;
				for (int j = 0; j < hidden_size; ++j) {
					raw_output[i] += Wy[i * hidden_size + j] * a[t][j];
				}
				raw_output[i] += by[i];
			}

			softmax(raw_output, y_pred[t]);

			// Update previous hidden and cell state
			for (int i = 0; i < hidden_size; ++i) {
				a_prev[i] = a[t][i];
				c_prev[i] = c[t][i];
			}
		}
		return;
}

// Kernel function for GRU forward pass
void gru_forward(hls::stream<axis_t> &in, hls::stream<axis_t> &out) {
	#pragma HLS INTERFACE s_axilite port = return bundle = control
	#pragma HLS INTERFACE axis port = in
	#pragma HLS INTERFACE axis port = out

	DataType l_A[N2];
	DataType l_B[N2];
	DataType l_D[N2];
	DataType l_C[N2];

	#pragma HLS ARRAY_PARTITION variable = l_A factor = 16 dim = 1 cyclic
	#pragma HLS ARRAY_PARTITION variable = l_B factor = 16 dim = 1 block
	#pragma HLS ARRAY_PARTITION variable = l_D factor = 16 dim = 1 cyclic
	#pragma HLS ARRAY_PARTITION variable = l_C factor = 16 dim = 1 cyclic

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

	load_D:
	  for (int i = 0; i < i_limit; i++) {
		axis_t temp = in.read();
		for (int j = 0; j < j_limit; j++) {
		  int high = j * DataTypeSize + DataTypeSize - 1;
		  int low = j * DataTypeSize;
		  int index = i * 16 + j;

		  converter.i = temp.data.range(high, low);
		  l_D[index] = converter.d;
		}
	  }

	forward_acc<DataType>(l_A, l_B, l_D, l_C);

	writeC:
	  for (int i = 0; i < i_limit; i++) {
	    axis_t temp;
	    for (int j = 0; j < j_limit; j++) {
	      int high = j * DataTypeSize + DataTypeSize - 1;
	      int low = j * DataTypeSize;
	      converter.d = l_C[i * 16 + j];
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
