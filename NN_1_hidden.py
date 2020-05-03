import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1-(x))

#training_input = np.array([[0,0,1],
                           #[1,1,1],
                           #[1,0,1],
                           #[0,1,1]])

#training_output = np.array([0,1,1,0])


#training_output = training_output.reshape(4,1)
training_input, training_output = datasets.make_moons(300,noise = 0.15)
color = training_output
training_output = training_output.reshape(len(training_output),1)
#print(training_input)
#print(training_output)
np.random.seed(1)

w1 = 2*np.random.random((2,2))-1
w2 = 2*np.random.random((2,1))-1
epoch = 6000
loss_x = []
error_arr = []
def network_train(epoch):
    w1 = 2*np.random.random((2,2))-1
    w2 = 2*np.random.random((2,1))-1
    b1 = np.random.rand(1)
    b2 = np.random.rand(1)
    lr = 0.001
    for i in range(epoch):
        inputs = training_input
        output = training_output
        x = inputs
        layer1 = sigmoid(np.dot(x,w1))
        temp = sigmoid(np.dot(layer1,w2))
        layer2 = sigmoid(np.dot(layer1,w2))
        error = output-layer2
        dw2 = np.dot(layer1.T,2*(output-layer2)*d_sigmoid(layer2))
        w2 = w2+lr*dw2
        dw1 = np.dot(x.T, 
            np.dot(d_sigmoid(layer2)*2*(output-layer2),
            w2.T)*d_sigmoid(layer1))
        w1 = w1+lr*dw1
        error_arr.append(abs(error.sum()/(len(error))))
        loss_x.append(i)


def predict(data,w1,w2):
    layer1 = sigmoid(np.dot(x,w1))
    layer2 = sigmoid(np.dot(layer1,w2))
    if(layer2>0.5):
        print(1)
    else:
        print(0)


x = [0.5,-1]
network_train(epoch)
predict(x,w1,w2)
plt.plot(loss_x,error_arr)
plt.title("error vs epoch")
plt.show()
plt.scatter(x[0],x[1],c = "r")
plt.scatter(training_input[:,0],training_input[:,1],c = color)
plt.show()


       

    
    
    

   
    
    


    
    
    
    


    
    

    
    

    
