import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.linalg import cho_factor, cho_solve
from data_utils import load_dataset
from sklearn.metrics import mean_squared_error
from scipy.linalg import cho_factor, cho_solve
import scipy.linalg
import matplotlib.pyplot as plt
import math
import copy


x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x = x_train[:1000]
y = y_train[:1000]



# initialize the model parameters
#x = np.concatenate((np.ones((x.shape[0],1),x)),axis=1)
x = np.concatenate((np.ones((1000, 1)), x), axis=1)
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
x_valid = np.concatenate((np.ones((x_valid.shape[0], 1)), x_valid), axis=1)
weights = np.zeros((x.shape[1],1))



# define the loss function
def loss_function(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# load the dataset

def q1_FB():

    # set the learning rate and number of iterations
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    x = x_train[:1000]
    y = y_train[:1000]

    # initialize the model parameters
    #x = np.concatenate((np.ones((x.shape[0],1),x)),axis=1)
    x = np.concatenate((np.ones((1000, 1)), x), axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    x_valid = np.concatenate((np.ones((x_valid.shape[0], 1)), x_valid), axis=1)
    weights = np.zeros((x.shape[1],1))
    learning_rate_set = [0.00000001, 0.000001, 0.00009, 0.000095,0.0001]
    num_iterations = 50
    length = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    loss=[]

    for learning_rate in (learning_rate_set):
        weights = np.zeros((x.shape[1],1))
        loss = []
    # perform full-batch gradient descent
        for i in range(num_iterations):
            # compute the gradients for the entire dataset
            d_w=[]
            # Compute the predicted output
            for s in range(len(x)):
                y_pred = np.dot(x[s], weights)
                #gradient of log likelihood
                d_single = -2*(y_pred - y[s]) * x[s].T
                d_w.append(d_single)
            d_weights = np.sum(d_w,axis=0).reshape(-1,1)
            weights -= learning_rate * d_weights
          
            
            #plot loss vs iteration
            y_pred = np.dot(x_valid, weights)
            loss.append(loss_function(y_valid, y_pred))
            

        #find the rmse of the test set
        y_pred_test = np.dot(x_test, weights)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))
        print("RMSE of the test set is: ", rmse)
        plt.plot(range(num_iterations), loss,'-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss vs epoch full batch')
        #legend with lr = learning_set_rate
        plt.legend(learning_rate_set)
    
    plt.show()


def q1_batch():
    # set the learning rate and number of iterations

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    x = x_train[:1000]
    y = y_train[:1000]

    # initialize the model parameters
    #x = np.concatenate((np.ones((x.shape[0],1),x)),axis=1)
    x = np.concatenate((np.ones((1000, 1)), x), axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    x_valid = np.concatenate((np.ones((x_valid.shape[0], 1)), x_valid), axis=1)
    weights = np.zeros((x.shape[1],1))
    batch_size = 10
    #reorder data
    N = len(x)
    inx = np.random.permutation(N)
    x = np.take(x, inx, axis=0)
    y = np.take(y, inx, axis=0)
    weights = np.zeros((x.shape[1],1))
    d_weights_prev = np.zeros((x.shape[1],1))
    # loss = []
    learning_rate_set = [0.000001, 0.000001, 0.0001]
    learning_rate = 0.001
    losses = []
    rmse_valid = []
    beta = 0.9
    the_list = np.arange(0.00001, 0.001, 0.0001)
    learning_rate_set = [0.000001,0.0001,0.001, 0.01, 0.1]
    num_iterations = 100
    loss_lr = []
    loss=[]

    for learning_rate in the_list:
        weights = np.zeros((x.shape[1],1))
        losses = []
        for i in range(num_iterations):
            for j in range(0, N, batch_size):
                # Select the mini-batch
                X_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
            # Select the mini-batch
    
                d_w=[]
            # Compute the predicted output
                for s in range(batch_size):
                    y_pred = (np.dot(X_batch[s], weights))
                    #gradient of log likelihood
                    d_single = (y_pred - y_batch[s]) * X_batch[s].T
                    d_w.append(d_single)
                d_weights = np.sum(d_w,axis=0).reshape(-1,1)
                # d_weights = np.mean(d_weights,axis=0).reshape(-1,1)
                # Compute the gradient of the loss function
                # d_weights = np.dot(x.T, y_pred - y) / len(y)
                

            # Update the weights and biases
                weights =weights - learning_rate * d_weights
                #for momentum :
                # weights = weights - learning_rate *(beta*(d_weights_prev)+(1-beta)*d_weights)
            

                # Compute and store the loss
            y_final = np.dot(x,weights)
            loss = loss_function(y,y_final)
            losses.append(loss)

        y_pred_val = np.dot(x_valid, weights)
        plt.plot(range(num_iterations), losses,'-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('loss vs iteration batch = 10')
        plt.legend(the_list)
        
        rmse = math.sqrt(mean_squared_error(y_valid, y_pred_val))
        rmse_valid.append(rmse)


    print("RMSE of the validation set is: ", rmse)
        #plot 
        

    plt.show()
    plt.plot(the_list, rmse_valid,'-')
    #plot the minimum rmse
    plt.plot(the_list[np.argmin(rmse_valid)], min(rmse_valid), 'ro')
    plt.xlabel('learning rate')
    plt.ylabel('rmse')
    plt.title('rmse vs learning rate')
    plt.show()

def q1_momentum():
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    x = x_train[:1000]
    y = y_train[:1000]

    # initialize the model parameters
    #x = np.concatenate((np.ones((x.shape[0],1),x)),axis=1)
    x = np.concatenate((np.ones((1000, 1)), x), axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    x_valid = np.concatenate((np.ones((x_valid.shape[0], 1)), x_valid), axis=1)
    weights = np.zeros((x.shape[1],1))
    batch_size = 1
    #reorder data
    N = len(x)
    inx = np.random.permutation(N)
    x = np.take(x, inx, axis=0)
    y = np.take(y, inx, axis=0)
    weights = np.zeros((x.shape[1],1))
    d_weights_prev = np.zeros((x.shape[1],1))
    # loss = []
    learning_rate_set = [0.001]
    learning_rate = 0.001
    losses = []
    rmse_valid = []
    beta = 0.9
    num_iterations = 50
    y_pred_beta=[]
    the_list = np.arange(0.0000001,0.01,0.001)
    beta_list = np.arange(0.1,0.9,0.1)
    for beta in beta_list:
        weights = np.zeros((x.shape[1],1))
        d_weights_prev = np.zeros((x.shape[1],1))
        for learning_rate in learning_rate_set:
            for i in range(num_iterations):
                for j in range(0, N, batch_size):
                    X_batch = x[j:j+batch_size]
                    y_batch = y[j:j+batch_size]
                # Select the mini-batch
        
                    d_w=[]
                # Compute the predicted output
                    for s in range(batch_size):
                        y_pred = (np.dot(X_batch[s], weights))
                        #gradient of log likelihood
                        d_single = (y_pred - y_batch[s]) * X_batch[s].T
                        d_w.append(d_single)
                    d_weights = np.sum(d_w,axis=0).reshape(-1,1)
                    # d_weights = np.mean(d_weights,axis=0).reshape(-1,1)
                    # Compute the gradient of the loss function
                    # d_weights = np.dot(x.T, y_pred - y) / len(y)
                    

         

                    # Update the weights and biases
                    # weights -= learning_rate * d_weights
                    #for momentum :
                    weights = weights - learning_rate *(beta*(d_weights_prev)+(1-beta)*d_weights)
                    d_weights_prev = d_weights
                

                    # Compute and store the loss
        y_final = np.dot(x,weights)
        loss = loss_function(y,y_final)
        losses.append(loss)
        y_pred_val = np.dot(x_valid, weights)
        
        rmse = math.sqrt(mean_squared_error(y_valid, y_pred_val))
        rmse_valid.append(rmse)
        #plot rmse_valid by learning rate
        
        
    plt.plot(beta_list, losses,'*')
    plt.xlabel('beta')
    plt.ylabel('loss')
    plt.title('loss vs beta')
    #make scale betwen 0 to 0.9
    # plt.yticks(np.arange(0, 1, 0.1))


    # plt.legend(['beta = 0.1', 'beta = 0.4', 'beta = 0.7'])

        # print("RMSE of the validation set is: ", rmse)
        # #plot 
        # plt.plot(length, losses,'*')
        # plt.xlabel('iteration')
        # plt.ylabel('loss')
        # plt.title('loss vs iteration')

    plt.show()

####Q2######
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_likelihood(y, y_pred):
    res =[]
    for i in range(len(y)):
        res.append(y[i]*np.log(sigmoid(y_pred[i]))+(1-y[i])*np.log(1-sigmoid(y_pred[i])))
    return np.sum(res)
def q2_FBGD():

  
    #implement gradient decent on iris dataset
    #load data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    x =x_train

    y= y_train

    y = y[:,(1,)].astype(int)
    y_valid = y_valid[:,(1,)].astype(int)
    losses=[]
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    x_valid = np.concatenate((np.ones((x_valid.shape[0], 1)), x_valid), axis=1)
    weights = np.zeros((x.shape[1],1))
    num_iterations = 100
    log_like_train = []
    log_like_test = []  


    the_list_list = np.arange(0.00001,0.01,0.001)
    for learning_rate in the_list_list:
        log_like_train = []
        weights = np.zeros((x.shape[1],1))
        for i in range(num_iterations):

            # Select the mini-batch
    
            d_w=[]
            # Compute the predicted output
            for s in range(x.shape[0]):

                y_pred = sigmoid(np.dot(x[s], weights))
                #gradient of log likelihood
                d_single = (y_pred - y[s]) * x[s].T
                d_w.append(d_single)
            d_weights = np.sum(d_w,axis=0).reshape(-1,1)
            # d_weights = np.mean(d_weights,axis=0).reshape(-1,1)
            # Compute the gradient of the loss function
            # d_weights = np.dot(x.T, y_pred - y) / len(y)
            

            # Update the weights and biases
            weights =weights - learning_rate * d_weights
            #for momentum :
            # weights = weights - learning_rate *(beta*(d_weights_prev)+(1-beta)*d_weights)
        
            log_like_train.append(-1*log_likelihood(y,np.dot(x,weights)))
        log_like_test.append(-1*log_likelihood(y_valid,np.dot(x_valid,weights)))
    
            # Compute and store the loss
        #plot loglike and epoch
        plt.plot(range(num_iterations), log_like_train,'-')
    
        
       #chcek the loss on test set
            

        #plot the loss

    #label for each learning rate
    plt.xlabel('epoch')
    plt.ylabel('log likelihood')
    plt.title('log likelihood vs epoch')
    #legend for each learning rate
    plt.legend(the_list_list)
    plt.show()

    #plot the test set
    plt.plot(the_list_list, log_like_test,'-')
    plt.xlabel('learning rate')
    plt.ylabel('log likelihood')
    plt.title('log likelihood vs learning rate on test set')
    plt.show()


def q2_SGD2():
    #implement gradient decent on iris dataset
    #load data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')

    
    N = len(x_train)
    inx = np.random.permutation(N)
    x = np.take(x_train, inx, axis=0)
    y = np.take(y_train, inx, axis=0)

    y = y[:,(1,)].astype(int)
    y_valid = y_valid[:,(1,)].astype(int)
    losses=[]
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    x_valid = np.concatenate((np.ones((x_valid.shape[0], 1)), x_valid), axis=1)
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    weights = np.zeros((x.shape[1],1))
    num_iterations = 100
    log_like_train = []
    log_like_test = []  
    batch_size = len(x)
    N = len(x)
    best_log_like_test = float('-inf')


    the_list_list = np.arange(0.00001,0.01,0.001)
    for learning_rate in the_list_list:
        log_like_train = []
        weights = np.zeros((x.shape[1],1))
        for i in range(num_iterations):
            for j in range(0, N, batch_size):
                X_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
            # Select the mini-batch
    
                d_w=[]
            # Compute the predicted output
                for s in range(batch_size):
                    y_pred = sigmoid(np.dot(X_batch[s], weights))
                    #gradient of log likelihood
                    d_single = (y_pred - y_batch[s]) * X_batch[s].T
                    d_w.append(d_single)
                d_weights = np.sum(d_w,axis=0).reshape(-1,1)
                # d_weights = np.mean(d_weights,axis=0).reshape(-1,1)
                # Compute the gradient of the loss function
                # d_weights = np.dot(x.T, y_pred - y) / len(y)
                

            # Update the weights and biases
                weights =weights - learning_rate * d_weights
            #for momentum :
            # weights = weights - learning_rate *(beta*(d_weights_prev)+(1-beta)*d_weights)
        
            log_like_train.append(-1*log_likelihood(y,np.dot(x,weights)))
        log_like_test = (-1*log_likelihood(y_valid,np.dot(x_valid,weights)))
        if log_like_test > best_log_like_test:
            best_lr = learning_rate
            best_log_like_test = log_like_test

            # Compute and store the loss
        #plot loglike and epoch
        plt.plot(range(num_iterations), log_like_train,'-')
    
        
       #chcek the loss on test set
            

        #plot the loss

    #label for each learning rate
    plt.xlabel('epoch')
    plt.ylabel('log likelihood')
    plt.title('log likelihood vs epoch')
    #legend for each learning rate
    plt.legend(the_list_list)
    plt.show()

    #print the best learning rate, and the best log likelihood on test set
    print('best learning rate is :',best_lr)
    print('best log likelihood on test set is :',best_log_like_test)
    #find accuracy on test set
    y_pred = sigmoid(np.dot(x_test, weights))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print('accuracy on test set is :',accuracy)

if __name__ == "__main__":

    #q1_FB()
    # q1_batch()
    # q1_momentum()
    # q2_FBGD()
    # q2_SGD2()






    
        
