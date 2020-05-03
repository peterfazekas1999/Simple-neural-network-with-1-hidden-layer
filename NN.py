import numpy as np




def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

training_input = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])
training_output = np.array([0,1,1,0])
training_output = training_output.reshape(4,1)

np.random.seed(1)

weights1 = 2*np.random.random((3,1))-1
weights2 = 2*np.random.random((2,1))-1
#print(weights1)
bias = np.random.rand(1)
lr = 0.05
for i in range(3000):
    inputs = training_input
    output = training_output
    WX = np.dot(inputs,weights1) 
    
    z = sigmoid(WX)
    
    error = z - output
    delta = d_sigmoid(z)*error
    #print(np.dot(inputs.T,delta))
    weights1 = weights1 -lr*np.dot(inputs.T,delta)
    print(error.sum())
    

print(sigmoid(np.dot([0,0,1],weights1)))





    
    

