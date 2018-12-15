import numpy as np
import matplotlib.pyplot as plt 
import time
import tensorflow as tf

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import matplotlib as mpl
mpl.use('agg')

"""Import MNIST Data"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels


class GradientDescent: 

    tile_x = np.int(784)

    batch_gradient_descent_kernel_code = """
    #include <stdio.h>
    #include <math.h>
    __global__ void Batch_Gradient_Descent(float *X_train, float *y_train, float *weight_matrix, 
    									   float *eta, int batch_size, int rows, int columns)
    {
    	int bx = blockIdx.x;  
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int grid_row = by * blockDim.y + ty;  
        int grid_col = bx * blockDim.x + tx;

        //which row of data to use 
        data_row = grid_row / 10 ; 
        class_id = grid_row % 10; 

        // share weight
        // use the first 9 rows to share the weights
        /* __shared__ float shared_weight[10][columns]
        if (by <= 9){
        	shared_weight[class_id][tx] = weight_matrix[class_id * columns + tx]
        	__syncthreads(); 
        } */ 


        // share eta
        __shared__ float shared_eta = eta;  

        // share y_train 
        //use the first thread of the 0 class to map all the y_trains
        /* __shared__ float shared_y_train[rows]; 
        if (tx==0 **by ==0){
        	shared_y_train[data_row] = y_train[data_row]; 
        	__syncthreads(); 
        } */ 

        __shared__ float temp_derivative[784]
        __shared__ float temp_sum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 
        __shared__ float temp_y = 0.0; 


        if (y_train[data_row] == class_id){
        	temp_y = 1.0; 
        } 
        	
        //find w_t*x
        float one_dim_wx = X_train[data_row*columns+tx]*shared_weight[class_id][tx]

        __syncthreads(); 

        //find sum 


        //find e**-(w_t *x)
        float temp_val = 1/(1 + exp(sum)); 

        //get derivative of cost function 
        temp_derivative[tx] = X_train[data_row*columns+tx]*temp_val; 
	    
	    //	

        	}
        }
    }

    """

    prg_batch_gradient_descent = SourceModule(batch_gradient_descent_kernel_code)

    def __init__(self): 
    	self.X_train_gpu = None 
    	self.y_train_gpu = None 

    def prepare_data(self, X_train, y_train): 
    	if self.X_train_gpu is None: 
    		self.X_train_gpu = gpuarray.to_gpu(X_train) 
    	if self.y_train_gpu is None: 
    		self.y_train_gpu = gpuarray.to_gpu(y_train)

    """Initialize Weights with Random Numbers between -1 and 1""" 
    def batch_gradient_descent(self, X_train, y_train, eta0, batch_size): 
    	#initialize the weight matrix
    	rows = X_train.shape[0]
    	columns = X_train.shape[1]
    	weight_matrix = np.random.rand(10, columns) #10 classes
    	weight_gpu = gpuarray.to_gpu(weight_matrix)

    	#prepare data
    	self.prepare_data(X_train, y_train)

    	#set number of blocks 
    	block_x = (1).astype(np.int)
    	block_y = (rows).astype(np.int)*10

    	evt = GradientDescent.prg_batch_gradient_descent
    	start.record() 
    	evt(self.X_train_gpu, self.y_train_gpu, weight_gpu, np.float32(eta0) np.uint32(batch_size), 
    		np.uint32(rows), np.uint32(columns), 
    		block = (columns, rows, 1), 
    		grid = (block_x, block_y, 1))
    	end.record() 
    	end.synchronize() 
    	time_batch = start.time_till(end)*1e-3 

    	final_weights = weight_gpu.get() 

