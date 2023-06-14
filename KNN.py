import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random 
import argparse
from sklearn.neighbors import KDTree
import time
from data_utils import load_dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns


def split_list(lst, n):
    """Split a list into n segments and store each segment in a separate variable."""
    segment_length = len(lst) // n
    if len(lst) % n != 0:
        '''if not divisible by n then shorten the list to the nearest multiple of n'''
        lst = lst[:segment_length * n]
    segments = [lst[i:i + segment_length] for i in range(0, len(lst), segment_length)]

    return segments

'''knn classification using kdtree'''


def knn_classification_kdtree(x_train, y_train, x_test, y_test, k=1):
    kdt = KDTree(x_train, metric='euclidean')
    distances, indices = kdt.query(x_test, k=k)
    y_pred = []
    for y in y_train[indices]:
        vote, count = np.unique(y, return_counts=True)
        y_pred.append(vote[np.argmax(count)])
    accuracy = np.mean(y_pred == y_test)
    return accuracy


    
    acc = np.mean(y_pred == y_test)
    return acc
'''function that loops through different values of k and returns the k with the best accuracy using knn_classification_kdtree'''
def best_k_classification(x_train, y_train,x_test,y_test):
    k_values = [1,2, 3, 5,6, 7, 11, 13, 19,20,70]
    acc_values = []
    for k in k_values:
        acc = knn_classification_kdtree(x_train, y_train, x_test, y_test, k=k)
        acc_values.append(acc)
        print('k = {}, accuracy = {}'.format(k, acc))
    best_k = k_values[np.argmax(acc_values)]
    plt.plot(k_values, acc_values, 'o-')
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.title('k vs Accuracy, Euclidean Distance')
    plt.show()
    print(best_k)
    return best_k


def knn_regression(x_train, y_train, x_test, y_test, l,k=1):
    rmse_mean = []
    y_pred_values = []
    for t in range(len(x_test)):
        distances = []
        for i in range(len(x_train)):
            if l == "l1":
                '''manhattan distance'''
                distance = np.sum(abs(x_test[t] - x_train[i]))
            if l == "l2":
                '''euclidean distance'''
                distance = np.sqrt(np.sum((x_test[t] - x_train[i])**2))
            distances.append(distance.item())
        k_nearest = np.argsort(distances, axis=0)[0:k]
        y_pred = np.mean(y_train[k_nearest], axis=0)
        y_pred_values.append(y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[t], y_pred))
        rmse_mean.append(rmse)
    rmse_mean = np.mean(rmse)
    return rmse_mean, y_pred_values



def knn_regression_kdtree(x_train, y_train, x_test, y_test, k=1):
    kdt = KDTree(x_train,metric='euclidean')
    distances, indices = kdt.query(x_test, k=k)
    y_pred = np.zeros(y_test.shape)
    for i in range(len(x_test)):
        y_pred[i] = np.mean(y_train[indices[i]])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return
    '''plot x_test vs y_test and x_test vs y_pred'''
    plt.plot(x_test, y_test, 'o',markersize=2)
    plt.plot(x_test, y_pred, 'o', color='red',markersize=2)
    plt.plot(x_train, y_train, 'o', color='green',markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y_train', 'y_pred for '+str(k)+' nearest neighbours'])
    plt.title('x vs y')
    plt.show()
    return rmse, y_pred


def best_kd_tree(x_train, y_train,x_test,y_test,l):
    k_values = [1,2, 3, 5, 7, 11, 13, 19,20,30]
    rmse_values = []
    y_pred_val_mean =[]
    rmse= []
    y_final_pred = []
    for k in k_values:
        y_final = []
        rmse_val, y_pred_val = knn_kd(x_train, y_train, x_test, y_test,l, k=k)
        rmse.append(rmse_val)
        y_final_pred.append(y_pred_val)
        print('k = {}, rmse = {}'.format(k, np.mean(rmse)))

    best_k = k_values[np.argmin(rmse)]
    plt.plot(x_test, y_test, 'o',markersize=2)
    plt.plot(x_test, y_final_pred[np.argmin(rmse)], 'o', color='red',markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y_train', 'y_pred for '+str(best_k)+' nearest neighbours'])
    if l=="l1":
        plt.title('x vs y'+" Manhattan Distance")
    if l=="l2":
        plt.title('x vs y'+" Euclidean Distance")
    plt.show()
    plt.plot(k_values, rmse, 'o-')
    plt.xlabel('k values')
    plt.ylabel('RMSE')
    if l=="l1":
        plt.title('RMSE vs k values'+" Manhattan Distance")
    if l=="l2": 
        plt.title('RMSE vs k values'+" Euclidean Distance")
    plt.show()
    return best_k






'''the same function as best_k but doesnt split the data into 5 folds'''
def best_k_no_split(x_train, y_train,x_test,y_test,l):
    k_values = [1,2, 3, 5, 7, 11, 13, 19,20,30]
    rmse_values = []
    y_pred_val_mean =[]
    rmse= []
    y_final_pred = []
    for k in k_values:
        y_final = []
        rmse_val, y_pred_val = knn_regression(x_train, y_train, x_test, y_test,l, k=k)
        rmse.append(rmse_val)
        y_final_pred.append(y_pred_val)
        print('k = {}, rmse = {}'.format(k, np.mean(rmse)))
    
    best_k = k_values[np.argmin(rmse)]
    plt.plot(x_train, y_train, 'o', color='green',markersize=2)
    plt.plot(x_test, y_test, 'o',markersize=2)
    plt.plot(x_test, y_final_pred[np.argmin(rmse)], 'o', color='red',markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y_train','y_test', 'y_pred for '+str(best_k)+' nearest neighbours'])
    if l=="l1":
        plt.title('x vs y'+" Manhattan Distance")
    if l=="l2":
        plt.title('x vs y'+" Euclidean Distance")
    plt.show()
    plt.plot(k_values, rmse, 'o-')
    plt.xlabel('k values')
    plt.ylabel('RMSE')
    if l=="l1":
        plt.title('RMSE vs k values'+" Manhattan Distance")
    if l=="l2": 
        plt.title('RMSE vs k values'+" Euclidean Distance")
    plt.show()
    return best_k



def best_k(x_train, y_train,l):
    print(l)
    N = len(x_train)
    inx = np.random.permutation(N)
    #x_train = np.take(x_train, inx, axis=0)
    #y_train = np.take(y_train, inx, axis=0)
    x = split_list(x_train, 5)
    y = split_list(y_train, 5)
    x_size = len(x[0])
    k_values = [1,2, 3, 5, 6,7, 11, 13, 19,20,30,35,45,50,55]
    rmse_values = []
    y_pred_val_mean =[]
    y_final_pred = []


    for k in k_values:
        rmse = []
        y_final = []
        for i in range(len(x)):
            x_train_folds = np.concatenate(x[:i] + x[i+1:])
            y_train_folds = np.concatenate(y[:i] + y[i+1:])
            x_test = x[i]
            y_test = y[i]
            rmse_val, y_pred_val = knn_regression(x_train_folds, y_train_folds, x_test, y_test,l, k=k)
            rmse.append(rmse_val)
            '''concatenate the predicted values for each fold'''
            y_final = y_final + y_pred_val
        y_final_pred.append(y_final)
        print('k = {}, rmse = {}'.format(k, np.mean(rmse)))
        rmse_values.append(np.mean(rmse))
      
    best_k = k_values[np.argmin(rmse_values)]
    '''plot two graphs on the same plot, one x_train vs y_train and the other x_train vs y_final of the best k . include the title and labels'''
    plt.plot(x_train, y_train, 'o',markersize=2)
    plt.plot(x_train[0:(5*x_size)], y_final_pred[np.argmin(rmse_values)], 'o', color='red',markersize=2)
    plt.plot(x_train[0:(5*x_size)], y_final_pred[5], 'o', color='green',markersize=2)
    plt.plot(x_train[0:(5*x_size)], y_final_pred[8], 'o', color='yellow',markersize=2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    '''include a legend to show the difference between the two graphs'''

    plt.legend(['y_train', 'y_pred for '+str(best_k)+' nearest neighbours','y_pred for ' + str(k_values[5])+ ' nearest neighbours', 'y_pred for ' + str(k_values[8])+ ' nearest neighbours'])
    if l=="l1":
        plt.title('x vs y'+" Manhattan Distance")
    if l=="l2":
        plt.title('x vs y'+" Euclidean Distance")
    plt.show()
    '''plot the graph of k vs rmse'''
 
    plt.plot(k_values, rmse_values, 'o-')
    plt.xlabel('k values')
    plt.ylabel('RMSE')
    if l=="l1":
        plt.title('RMSE vs k values'+" Manhattan Distance")
    if l=="l2": 
        plt.title('RMSE vs k values'+" Euclidean Distance")
    plt.show()
    return best_k, min(rmse_values)

    '''function that perform linear regression on training data using svd'''
def linear_regression_svd(x_train, y_train):    
    '''perform svd on x_train'''
    u, s, v = np.linalg.svd(x_train, full_matrices=False)
    '''compute the pseudo inverse of x_train'''
    u = u[:,:len(s)]
    s_inv = np.diag(1/s)
    w = np.dot(v.T ,np.dot(s_inv, np.dot(u.T,y_train)))
    return w
   
'''function that perform linear classification on training data using svd'''

if __name__ == "__main__":


#_______________Q1_____________________
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    # x = np.concatenate((x_train, x_valid))
    # y = np.concatenate((y_train, y_valid))
    # a,b = knn_regression(x,y,x_test, y_test, 'l2',k=12)
    # print(a,b)
    #best_k(x_train, y_train, 'l2')
    #best_k_no_split(x_train, y_train, x_test,y_test,'l1')



#_________Q2___________

    # d = [2,3,4,10,15,20,30]
    # times_kdtree = []
    # times_no_split = []
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock',n_train=5000,d=2)
    # for i in range(len(d)):

    #     start = time.time()
    #     k = knn_regression_kdtree(x_train, y_train,x_test,y_test,5)
    #     print(d[i])

    #     end = time.time()
    #     times_kdtree.append(end-start)

        # start = time.time()
        # k = knn_regression(x_train, y_train,x_test,y_test,'l2',5)
        # print(d[i])

        # end = time.time()
        # times_no_split.append(end-start)
     
    # plt.plot(d, times_kdtree, 'o-',color='red')
    # #plt.plot(d, times_no_split, 'o-', color='red')
    # plt.xlabel('d values')
    # plt.ylabel('time')
    # plt.title('time vs d values')
    # plt.legend([ 'kdtree'])
    # plt.show()


   #_________________________________Q3_______________________________________________________
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    # y_train = np.argmax(y_train, axis=1)
    # y_train = y_train.reshape(-1,1)
    # y_valid = np.argmax(y_valid, axis=1)
    # y_valid = y_valid.reshape(-1,1)
    # y_test = np.argmax(y_test, axis=1)
    # y_test = y_test.reshape(-1,1)
  
    # best_k_classification(x_train, y_train, x_test, y_test)
  
   
  
#_______________Q4_____________________
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock',n_train=5000,d=2)
    type = 'regression' #SELECT regression OR classification
    if type == 'classification':
        y_train = np.argmax(y_train, axis=1)
        y_train = y_train. reshape(-1,1)
        y_valid = np.argmax(y_valid, axis=1)
        y_valid = y_valid. reshape(-1,1)
        y_test = np.argmax(y_test, axis=1)
        y_test = y_test. reshape(-1,1)
        x_train = np.concatenate((x_train,x_valid))
        y_train = np.concatenate((y_train,y_valid))

    w = linear_regression_svd(x_train, y_train)
    y_predicted = np.dot(x_test, w)
   
    #error = np.sqrt(np.mean(np.square(y_predicted - y_test)))

    if type == 'classification':
        y_pred = np.round(y_predicted)  
        acc = np.mean(y_pred == y_test)
        print(acc)
    else:
        error = np.sqrt(np.mean(np.square(y_predicted - y_test)**2))
        print(error)
    
 
    #print(error)
 
    



