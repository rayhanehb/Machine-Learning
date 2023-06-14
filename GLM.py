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

################# Q3 #########################

def gaussian_kernel(x, z, theta=1.):
    """
    Evaluate the Gram matrix for a Gaussian kernel between points in x and z.
    Inputs:
        x : array of shape (N, d)
        z : array of shape (M, d)
        theta : lengthscale parameter (>0)
    Outputs:
        k : Gram matrix of shape (N, M)
    """
    # reshape matricies 
    x = np.expand_dims(x, axis=1)
    z = np.expand_dims(z, axis=0)
    # evaluate the kernel using the euclidean distances 
    return np.exp(-np.sum(np.square(x-z)/theta, axis=2, keepdims=False))


# load the dataset
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
# x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
x = np.concatenate((x_train, x_valid))
y = np.concatenate((y_train, y_valid))

# define the grid of shape parameter values
theta = [0.05, 0.1, 0.5, 1, 2]
# define the grid of regularization parameters
reg = [0.001, 0.01, 0.1, 1]
# define the grid of hyperparameters
hyperparameters = [(t, r) for t in theta for r in reg]
# define the list to store the mse values
mse_values = []

# iterate over the hyperparameters
for t, r in hyperparameters:
    # calculate the kernel matrix using the training set and the given hyperparameters
    K = gaussian_kernel(x_train, x_train, theta=t)
    # calculate the weight matrix using Cholesky factorization
    L, lower = cho_factor(K + r * np.eye(K.shape[0]))
    alpha = cho_solve((L, lower), y_train)
    # calculate the predicted values for the validation set
    y_pred = gaussian_kernel(x_valid, x_train, theta=t) @ alpha
    # calculate the mean squared error
    mse = mean_squared_error(y_valid, y_pred)
    # append the mse value to the list
    mse_values.append(mse)

# find the hyperparameters that give the lowest mse
best_idx = np.argmin(mse_values)
rmse = np.sqrt(mse_values[best_idx])
print('RMSE: {}'.format(rmse))
#print the lower
best_t, best_r = hyperparameters[best_idx]
print('Best hyperparameters: theta = {}, reg = {}'.format(best_t, best_r))

# train the final model on the combined training and validation set using the best hyperparameters
K = gaussian_kernel(x_train, x_train, theta=best_t)
L, lower = cho_factor(K + best_r * np.eye(K.shape[0]))
alpha = cho_solve((L, lower), y_train)
# calculate the predicted values for the test set
y_pred = gaussian_kernel(x_test, x_train, theta=best_t) @ alpha
# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('RMSE on test set: {}'.format(rmse))


############################################################################################################
#Q4



# load the dataset
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
x = np.concatenate((x_train, x_valid))
y = np.concatenate((y_train, y_valid))

# define the basis functions
class cos():
    def __init__(self, omega,phi):
        self.omega = omega
        self.phi = phi
    def r(self, x):
        return np.cos(self.omega*x + self.phi)

class polynomial():
    def __init__(self, degree):
        self.degree = degree
    def r(self, x):
        return np.power(x, self.degree)
    
class exponential():
    def __init__(self, mu):
        self.mu = mu
    def r(self, x):
        return np.exp(self.mu*(x))



def basis_functions(n=200):
    #intialize empty array 
    basis = []
    #cosine parameters
    omega = np.linspace(50, 150, n//3)
    phi = np.linspace(0, 2*np.pi, n//3)
    #polynomial parameters
    degree = np.linspace(1, 10, n//3)
    #exponential parameters
    mu = np.linspace(0, 1, n//3)
    #iterate over parameters and append to basis
    for i in range(n//3):
        basis.append(cos(omega[i], phi[i]))
        basis.append(polynomial(degree[i]))
        basis.append(exponential(mu[i]))
    return basis



#ignore this function
def cosine_similarity(fun,res,x):
    return np.dot(fun.r(x).T, res) / (np.linalg.norm(fun.r(x)) * np.linalg.norm(res))

def J_function(phi_i,res):
    return np.square(np.dot(phi_i.T, res)) / np.abs(np.dot(phi_i.T,phi_i))


def greedy_selection(x, y, basis_functions):
    # Initialize a residual 
    res = y
    # Threshold
    epsilon = 0.1
    # Initialize a list of basis functions
    basis_list = []
    # Create a copy of basis functions
    basis_functions_list = copy.deepcopy(basis_functions)
    loss = np.linalg.norm(res)
    N= x.shape[0]
    k = len(basis_list)
    # Set of weights
    w = []
    #equate J to negative infinity
    cur_J = -np.inf
    J_list=[]
    first_it = True
    '''N/2 LOG(L2 - LOSS +K/2 LOG N)'''
    MDL = (N/2)*np.log(loss)+((k/2)*np.log(N))
    prev_MDL =   MDL


    while loss > epsilon and  MDL<=prev_MDL:
        cur_J = -np.inf
        # Initialize a list of cosine similarities
        cos_sim = []
        # Iterate over basis functions
        for fun in basis_functions_list:
            
            phi_i = fun.r(x)
            J =J_function(phi_i,res)
            
            #check to see if current J is better than previous J
            if not math.isnan(J): #TO AVOID NAN COMPARISON
                #add J to array J_list
                if J>cur_J:
                    cur_J = J
                    best_f = fun
            else:
                # print("J is nan")
                pass



        # Get the index of the basis function with the highest  similarity
        # index = np.argmax(np.array(J))

        # Append the basis function to the basis list
        #FIND INDEX OF fun IN basis_functions_list
        basis_functions_list.remove(best_f)
        # basis_functions_list.pop(index)

        if  first_it:
            print("basis list is empty")
            first_it = False
            phi = best_f.r(x)
        else:
            phi = np.concatenate((phi, best_f.r(x)), axis=1)

        basis_list.append(best_f)
        # Compute the weights using pseudo inverse
        w = np.linalg.pinv(phi).dot(y)
        #residual 
        res = y - phi.dot(w)
        #plot predictioj
        k = len(basis_list)
        loss = np.linalg.norm(res)
        prev_MDL = MDL
        MDL = (N/2)*np.log(loss)+((k/2)*np.log(N))
        
        plt.plot(x, y, 'b.',markersize=2)
        plt.plot(x, phi.dot(w), 'r.',markersize=2)
        #include title, axis labels, and legend
        plt.title('Greedy Selection, basis function # = ' + str(k) )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['data', 'prediction'])

        plt.show()




    return basis_list, w, res



'''evalutate test set using the basis functions and weights from the training set'''
def evaluate_test(x_train, y_train, x_test, y_test, basis_list, w):
    # Initialize a list of basis functions
    basis_functions_list = copy.deepcopy(basis_list)
    # Initialize a list of basis functions
    phi = []
    # Iterate over basis functions
    for fun in basis_functions_list:
        phi.append(fun.r(x_test))
    # Compute the weights using pseudo inverse
    phi = np.concatenate(phi, axis=1)
    # Compute the weights using pseudo inverse
    w = np.linalg.pinv(phi).dot(y_test)
    #residual 
    res = y_test - phi.dot(w)
    #calulate rmse
    rmse = np.sqrt(np.mean(np.square(res)))
    #plot predictioj
    plt.plot(x_test, y_test, 'b.')
    plt.plot(x_test, phi.dot(w), 'r.')
    '''AXIS LABELS'''
    plt.xlabel('x') 
    plt.ylabel('y')
    '''title'''
    plt.title('Test Set Predictions')
    # legen
    plt.legend(['Test Data', 'Predictions'])
    plt.show()
    return res
        
 

# plot the predictions
basis_list, w, res = greedy_selection(x,y,basis_functions())
print(len(basis_list))
'''evaluate test set using the basis functions and weights from the training set'''
res = evaluate_test(x_train, y_train, x_test, y_test, basis_list, w)

print(np.linalg.norm(res))


