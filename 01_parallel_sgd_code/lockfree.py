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

#sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve

"""Import MNIST Data"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

X_train = np.float32(X_train)
X_test = np.float32(X_test)
y_train = np.int32(y_train)
y_test = np.int32(y_test)

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

	sgd_lockfree_kernel_code = """
		#include <stdio.h>
		#include <math.h>
		__global__ void SGD_lockfree(float *X_train, int *y_train, float *weights, float *output, float eta, int rows)
		{

		//initialize -------------
		int tx = threadIdx.x;
		int bx = blockDim.x * blockIdx.y + tx;
		const int data_rows = rows;
		const int data_dimension = 784;
		float e = 2.718281828459;

		//put weights in shared memory ----------
		__shared__ float weight_shared[10][data_dimension];

		for (int i = 0; i<data_dimension; i++){
			if (tx < 10){
				weight_shared[tx][i] = weights[tx*data_dimension + i];
			}
		}
		__syncthreads();

		// calculate dot product for each class
		// 10 threads for 10 classes
		// tx should be between 0 and 9
		// not truly random since you can't call rand() inside pycuda kernel

		for (int data=0; data<data_rows; data++){
			float temp_dot_product = 0.0; 
			int rand_array[30]; 
			for (int i=0; i<30; i++){
				int rand_temp = (data * 9239 + i) % 784;   
				rand_array[i] = rand_temp; 
				temp_dot_product += weight_shared[tx][rand_temp] * X_train[data_dimension * data + rand_temp]; 
			}

			int y_star = 1; 
			int current_y_train = y_train[data]; 
			if (current_y_train != tx){ 
				y_star = 0;
			}

			//get y_hat 
			float y_hat = 1.0/(1.0+powf(e, (-1.0*temp_dot_product)));

			//get coefficient
			float coefficients = eta * (y_star - y_hat)*y_hat*(1.0-y_hat);

			for (int i=0; i<20; i++){
				int rand_curr = rand_array[i]; 
				weight_shared[tx][rand_curr] += coefficients * X_train[data_dimension * data + rand_curr]; 
			}
			__syncthreads(); 
		}

	
		//write weights back
		for (int n = 0; n<data_dimension; n++){
			if (tx < 10){
				output[tx*data_dimension + n] = weight_shared[tx][n];
			}
		}
		}

		"""
	
	prg_sgd_lockfree = SourceModule(sgd_lockfree_kernel_code)

	def __init__(self):
		self.X_train_gpu = None
		self.y_train_gpu = None
	
	def prepare_data(self, X_train, y_train):
		if self.X_train_gpu is None:
			self.X_train_gpu = gpuarray.to_gpu(X_train)
		if self.y_train_gpu is None:
			self.y_train_gpu = gpuarray.to_gpu(y_train)

	def sgd_lockfree(self, X_train, y_train, eta0, weights=None):
		#get data size
		rows = np.int32(X_train.shape[0])
		columns = X_train.shape[1]
		classes = 10
		eta = np.float32(eta0)
		# print("row: ", rows, "columns :", columns)

		#initialize weight array (1-d array with size of columns)
		if weights is None:
			weights = (np.random.rand(10, columns)).astype(np.float32)
			weights_gpu = gpuarray.to_gpu(weights)
			# print("weights: ", weights)
		else:
			weights_gpu = gpuarray.to_gpu(weights)
			# print("weights taken from input")

		#prepare data
		self.prepare_data(X_train, y_train)
		output = np.empty_like(weights)
		output_gpu = gpuarray.to_gpu(output)

		#timing event
		evt = GradientDescent.prg_sgd_lockfree.get_function("SGD_lockfree")

		start = cuda.Event()
		end = cuda.Event()

		start.record()
		evt(self.X_train_gpu, self.y_train_gpu, weights_gpu, output_gpu, 
			eta, rows, 
			block = (10, 1, 1))

		#float *X_train, int *y_train, float *weights, float *output, float eta, int rows

		end.record()
		end.synchronize()

		time = start.time_till(end)*1e-3 #get units
		final_weights = output_gpu.get()
		return time, final_weights

"""
Accuracy Testing
"""

def accuracy_test(test_data, test_label, weights):
	accuracy = 0
	#weights is size columns x 10
	weight_transposed = np.transpose(weights)
	prediction_matrix = np.matmul(test_data, weight_transposed)
	predictions = np.argmax(prediction_matrix, axis=1)
	accuracy = 1 - np.count_nonzero(test_label - predictions)/len(test_label)
	# print("-"*60)
	# print(predictions[0:10])
	# print(test_label[0:10])
	print("accuracy: ", accuracy)
	return accuracy

"""
Testing and Plotting
"""

if __name__ == '__main__':
	#parameters
	epochs = 100
	eta = np.float32(1) 

	columns = int(X_train.shape[1])
	weights = np.float32(np.zeros((10, columns)))
	times_sgd = [0]
	accuracies = [0]

	# print(weights)

	i = 1
	while i < epochs:
		#find times and accuracies for 99 epoches for sgd_lockfree
		gradient_descent = GradientDescent()

		start = time.time()
		time_new, weights = gradient_descent.sgd_lockfree(X_train, y_train, eta, weights)
		time_new_whole = time.time() - start

		time_new = times_sgd[i-1] + time_new
		times_sgd.append(time_new)

		accuracy = accuracy_test(X_test, y_test, weights)
		accuracies.append(accuracy)

		# print(weights.shape)

		i += 1

	#SGDClassifier
	sgd_best = SGDClassifier(loss = 'log',
						penalty='none', 
						tol=0.0, 
						fit_intercept= False,  
                        eta0=0.01, 
						learning_rate='constant')

	# param_range = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	param_range = range(1, 10)

	times = [] 
	train_scores = [] 
	test_scores = [] 
	for iteration in param_range: 
		sgd_temp = SGDClassifier(loss = 'log',penalty='none', 
								tol=0.000001, eta0=0.01, 
								learning_rate='constant', 
								max_iter = iteration, 
								fit_intercept=False)
		start = time.time()
		sgd_temp.fit(X_train, y_train)
		train_scores.append(sgd_temp.score(X_train, y_train))
		test_scores.append(sgd_temp.score(X_test, y_test))
		times.append(time.time()-start)

	#plot
	plt.subplot(2, 1, 1)
	plt.title('SGD lockfree times')
	sizes = np.array(range(0, epochs))
	plt.plot(sizes, times_sgd, 'g--', label='jx2181-parallel')
	plt.plot(param_range, times, 'b--', label='sklearn-serial')
	plt.legend(loc='upper left')
	plt.ylabel('run time (secs)')
	plt.legend(loc='upper left')
	plt.tight_layout()

	plt.subplot(2, 1, 2)
	plt.title('SGD lockffree accuracies')
	plt.plot(sizes, accuracies, 'g--', label='parallel')
	plt.plot(param_range, test_scores, 'b--', label='sklearn-serial')
	# plt.ylim(0.5, 0.925)
	plt.xlabel('iteration')
	plt.ylabel('accuracy')
	plt.legend(loc='upper left')
	plt.tight_layout()

	plt.rcParams["figure.figsize"] = [8, 8]
	plt.savefig('sgd_lockfree_runtime.png')

	#plot side by size with benchmark
