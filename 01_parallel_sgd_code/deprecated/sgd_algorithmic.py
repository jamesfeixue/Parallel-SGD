#imports
#essentials
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

#pycuda
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

"""Import MNIST Data"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels


class GradientDescent:
	"""
	Implementation of SGD with:
		-fixed step size
		-no early stopping
		-logistic regression for MNIST data

	Four Modes:
		-algorithmic
		-batch
		-multiclass
		-Hogwild!
	"""

	sgd_algorithmic_kernel_code = """
		#include <stdio.h>
		#include <math.h>
		__global__ void SGD_algorithmic(float *X_train, float *y_train, float *weights, float *eta, int rows)
		{

		//initialize -------------
		int tx = threadIdx.x;
		//int bx = blockDim.x * blockIdx.x + tx;
		const int data_rows = rows;
		const int data_dimension = 784;

		//put weights in shared memory ----------
		__shared__ float weight_shared[10][data_dimension];
		__shared__ float temp_dot_product[10][data_dimension];
		__shared__ float y_hat[10];
		__shared__ float coefficients[10];

		for (int i = 0; i<10; i++){
			if (tx < data_dimension){
				weight_shared[i][tx] = weights[i*data_dimension + tx];
			}
		}
		__syncthreads();

		//loop==>
		for (int data=0; data<data_rows; data++){
			int X_beginning = data * data_dimension;
			int y_beginning = data;

			// calculate dot product for each class
			// do addition with tree struction
			for (int j = 0; j<10; j++){
				if (tx < data_dimension) {
					temp_dot_product[j][tx] = weight_shared[j][tx] * X_train[X_beginning+tx];  //check this is right
				}
			}
			__syncthreads();

			for (int j = 0; j<10; j++){
				for (int maximum = blockDim.x; maximum>0; maximum = (maximum+1)/2){
					if (tx <= maximum/2 && (2*tx+1) <= maximum){
						temp_dot_product[j][tx] = temp_dot_product[j][2*tx] + temp_dot_product[j][2*tx+1];
					}
					__syncthreads();
					if (tx <= maximum/2 && (2*tx+1) == maximum){
						temp_dot_product[j][tx] = temp_dot_product[j][2*tx];
					}
				}
			}
			__syncthreads();

			//we have the dot product of
			//sgd formula
			float e = 2.718281828459;
			if (tx == 0){
				for (int k = 0; k<10; k++){
					float y_star = 0; 
					//current y_train
					float current_y_train = y_train[y_beginning];

					//convert to 0/1
					if (current_y_train == k){
						y_star = 1;
					}

					//get y_hat
					y_hat[k] = 1/(1+powf(e, (-1*temp_dot_product[k][tx])) );
					coefficients[k] = eta * (y_star - y_hat[k])*y_hat[k]*(1-y_hat[k]); //enum type
					printf("%f", coefficients[k]); 
				}

			}

			//update weights
			//double check this is correct
			for (int m = 0; m<10; m++){
				if (tx < data_dimension){
					weight_shared[m][tx] = weight_shared[m][tx] + coefficients[m]*X_train[X_beginning+tx]; //make sure bx is correct
				}
			}
			__syncthreads();
		}
		//write weights back
		for (int n = 0; n<10; n++){
			if (tx < data_dimension){
				weights[n*data_dimension + tx] = weight_shared[n][tx];
			}
		}
		}

		"""
	
	prg_sgd_algorithmic = SourceModule(sgd_algorithmic_kernel_code)

	def __init__(self):
		self.X_train_gpu = None
		self.y_train_gpu = None
	
	def prepare_data(self, X_train, y_train):
		if self.X_train_gpu is None:
			self.X_train_gpu = gpuarray.to_gpu(X_train)
		if self.y_train_gpu is None:
			self.y_train_gpu = gpuarray.to_gpu(y_train)

	def sgd_algorithmic(self, X_train, y_train, eta0, weights=None):
		#get data size
		rows = X_train.shape[0]
		columns = X_train.shape[1]
		print("row: ", rows, "columns :", columns)

		#initialize weight array (1-d array with size of columns)
		if weights is None:
			weights = (np.random.rand(10, columns)).astype(np.float32)
			weights_gpu = gpuarray.to_gpu(weights)
			print("weights: ", weights)
		else:
			weights_gpu = gpuarray.to_gpu(weights)
			print("weights taken from input")

		#prepare data
		self.prepare_data(X_train, y_train)

		#set block size
		#assuming the data is mnist, we want a block to be 1x*columns
		#the grid size will then be *rows
		#[thread thread thread thread]
		block_x = (columns).astype(np.int)
		block_y = 1

		#set grid size
		grid_x = (rows).astype(np.int)
		grid_y = 1

		#timing event
		evt = GradientDescent.prg_sgd_algorithmic.get_function("SGD_algorithmic")

		start = cuda.Event()
		end = cuda.Event()

		start.record()
		evt(self.X_train_gpu, self.y_train_gpu, weights_gpu, eta0, rows, block=(columns, 1, 1))
		end.record()
		end.synchronize()

		time = start.time_till(end)*1e-3 #get units
		final_weights = weights_gpu.get()
		return time, final_weights

"""
Accuracy Testing
"""

def accuracy_test(test_data, test_label, weights):
	accuracy = 0
	#weights is size columns x 10
	prediction_matrix = np.matmul(test_data, weights)
	predictions = np.argmax(prediction_matrix, axis=1)
	accuracy = 1 - np.count_nonzero(test_label - predictions)/len(test_label)
	print("accuracy: ", accuracy)
	return accuracy

"""
Testing and Plotting
"""

if __name__ == '__main__':

    weights = (np.random.rand(10, columns)).astype(np.float32)
    times_sgd_algorithmic = [0]
    times_sgd_algorithmic_whole = [0]
    accuracies = [0]

    i = 1
    while i < 100:
    	#find times and accuracies for 99 epoches for sgd_algorithmic
    	gradient_descent = GradientDescent()

    	start_algorithmic = time.time()
    	time_new, weights = gradient_descent.sgd_algorithmic(X_train, y_train, weights)
    	time_new_whole = time.time() - start_algorithmic

    	time_new = times_sgd_algorithmic[i-1] + time_new
    	times_sgd_algorithmic.append(time_new)

    	time_new_whole = times_sgd_algorithmic_whole[i-1] + time_new_whole
    	times_sgd_algorithmic_whole.append(time_new_whole)

    	accuracy = accuracy_test(X_test, y_test, weights)
    	accuracies.append(accuracy)

    	i += 1

    #plot
    plt.subplot(2, 1, 1)
    plt.title('SGD algorithmic times')
    sizes = np.array(range(1, 100))
    plt.plot(sizes, times_sgd_algorithmic, 'g--', label='kernel')
    plt.plot(sizes, times_sgd_algorithmic_whole, 'r--', label='whole')

    plt.subplot(2, 1, 2)
    plt.title('SGD algorithmic accuracies')
    plt.plot(sizes, accuracies, 'b--', label='kernel')

    plt.xlabel('iteration')
    plt.ylabel('run time (secs)')
    plt.legend(loc='upper left')

    plt.savefig('sgd_algorithmic_runtime.png')

    #plot side by size with benchmark
