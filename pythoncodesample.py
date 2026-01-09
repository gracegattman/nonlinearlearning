import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# import and split training and testing data sets
test_data_header = pd.read_csv('P3test.txt', sep='\t')
test_data = pd.read_csv('P3test.txt', sep='\t', skiprows = 1, names=['x1', 'x2', 'y'])
train_data_header = pd.read_csv('P3train.txt', sep='\t')
train_data = pd.read_csv('P3train.txt', sep='\t', skiprows = 1, names=['x1', 'x2', 'y'])

m = train_data_header.columns[0]
n = train_data_header.columns[1]

# define a logistic regression function based on combinations of the given x1 and x2 features

x1 = train_data.x1.to_numpy()
x2 = train_data.x2.to_numpy()
X = []

# highest power for function chosen after several rounds of manual testing and comparison to determine best fit
thePower = 4
for j in range(thePower+1): 
    for i in range(thePower+1):
        temp = (x1**i)*(x2**j)
        X.append(temp)

# weights set to 0 initially
init_weights = np.zeros((len(X), 1))
X = np.array(X)

# apply sigmoid function
Z = np.dot(init_weights.T, X)
sig = 1/(1+np.exp(-Z))

# to find the actual weights, gradient descent algorithm is applied:

# 1. define the hyperparameters:
# numbers of epoch (epoch_num), learning rate (lr), and the initial weights(w)
epoch_num = 10000
lr = 0.01
w = init_weights

# the J-curve is the training loss curve, below variables added to track it
J = 0
track_J = []

# 2. define the loss:
# sums the differences between predicted and true probabilities
def cross_entropy_loss(y_pred,y):
    loss = 0
    for i in range(len(y_pred[0])):
        # actual equation for loss here
        loss += -1*(y[i]*np.log(y_pred[0][i]) + (1-y[i])*(np.log(1-y_pred[0][i])))
    avg_loss = loss/len(y_pred[0])
    return avg_loss

# 3. calculate the gradient function:
# finds direction of greatest change, which allows us to go the opposite direction for descent
def gradient_func(J,w):
    Z = np.dot(w.T, X)
    sig = 1/(1+np.exp(-Z))
    m = len(X[0])
    gradient_value = (np.dot(X, (sig[0] - train_data.y).T)) / m
    return gradient_value

# 4. implement the gradient descent algorithm using a for loop
def Vanilla_GD(epoch_num,lr,w,J):
    for i in range(epoch_num):
        # init y_pred and J
        Z = np.dot(w.T, X)
        sig = 1/(1+np.exp(-Z))
        J = cross_entropy_loss(sig, train_data.y)
        
        # track J curve for plotting
        track_J.append(J)
        
        # calculate gradient
        gradient = gradient_func(J, w)
        gradient = np.array(gradient).reshape((len(w),1))
        # update weights
        w = w - (lr * gradient)
    
    return w

# call function to get weights
w = Vanilla_GD(epoch_num, lr, w, J)
print('Final weights:\n', w)

# plot the J curve with respect to the iteration numbers
plt.figure()
plt.plot(range(epoch_num), track_J, color = 'red')
plt.xlabel('Number of Iterations')
plt.ylabel('J value')
plt.title('J Curve/Loss Curve of Training')
plt.show()

# model evaluation metrics

# initialize testing data like training data
x1 = test_data.x1.to_numpy()
x2 = test_data.x2.to_numpy()
X_test_expand = []
thePower = 4
for j in range(thePower+1): 
    for i in range(thePower+1):
        temp = (x1**i)*(x2**j)
        if(temp[0] != 1.0):
            X_test_expand.append(temp)

# key difference: now the function has weights determined from training, not zeroed out
Z = w[0] + np.dot(w[1:].T, X_test_expand)
sig = 1/(1+np.exp(-Z))

# setting up probabilities
# data results should show passing cases as 1s and failing cases as 0s
prob = (sig >= 0.5).astype(int)
Y_test = test_data.y.to_numpy()

# set up confusion matrix
TP = np.sum((Y_test == 1) & (prob == 1))
print("True Positives:", TP)
FP = np.sum((Y_test == 0) & (prob == 1))
print("False Positives:", FP)
TN = np.sum((Y_test == 0) & (prob == 0))
print("True Negatives:", TN)
FN = np.sum((Y_test == 1) & (prob == 0))
print("False Negatives:", FN)

# find accuracy
accuracy = np.mean(prob[0] == Y_test)
print("Accuracy:", accuracy)

# find precison
if(TP+FP != 0):
    precision = TP/(TP+FP)
    print("Precision:", precision)
    
# find recall
if(TP+FN != 0):
    recall = TP/(TP+FN)
    print("Recall:", recall)
    
# find F1-score
F1 = 2 * ((precision*recall)/(precision+recall))
print("F1:", F1)

# support vector machine (SVM) implementation
# determining best model for comparison to logistic regression

#initialize data
X_train = np.column_stack((train_data.x1, train_data.x2))
X_test = np.column_stack((x1, x2))
Y_train = train_data.y.to_numpy()

# linear kernel
linKernel = SVC(kernel='linear')
linKernel.fit(X_train, train_data.y)
Y_pred = linKernel.predict(X_test)
accuracy = np.mean(Y_pred == Y_test)
print("Linear Kernel Accuracy:", accuracy)

# poly kernel
polyKernel = SVC(kernel='poly', degree=10)
polyKernel.fit(X_train, train_data.y)
Y_pred = polyKernel.predict(X_test)
accuracy = np.mean(Y_pred == Y_test)
print("Poly Kernel Accuracy:", accuracy)

# rbf kernel
rbfKernel = SVC(kernel='rbf')
rbfKernel.fit(X_train, train_data.y)
Y_pred = rbfKernel.predict(X_test)
accuracy = np.mean(Y_pred == Y_test)
print("RBF Kernel Accuracy:", accuracy)

# sigmoid kernel
sigKernel = SVC(kernel='sigmoid')
sigKernel.fit(X_train, train_data.y)
Y_pred = sigKernel.predict(X_test)
accuracy = np.mean(Y_pred == Y_test)
print("Sigmoid Kernel Accuracy:", accuracy)

# from these accuracy results, the rbf kernel is the best fit

# visualization of logistic regression and the rbf kernel
# setting up bounds for plot of data
x_min = train_data.x1.min() - 0.5
x_max = train_data.x1.max() + 0.5
y_min = train_data.x2.min() - 0.5
y_max = train_data.x2.max() + 0.5
xgrid, ygrid = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.zeros(xgrid.shape)
thePower = 4
w_val = 0
for j in range(thePower+1): 
    for i in range(thePower+1):
        Z += w[w_val] * (xgrid**i)*(ygrid**j)
        w_val = w_val + 1

# plotting LR
plt.figure()
plt.contour(xgrid, ygrid, Z, levels=[0.5], colors='k')
plt.scatter(train_data.x1, train_data.x2, c=Y_train, cmap='prism')
plt.title('Logistic Regression Visualization')
plt.show()

# adjusting data for SVM
rbfKernel = SVC(kernel='rbf')
rbfKernel.fit(X_train, train_data.y)
xgrid, ygrid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.zeros(xgrid.shape)
for i in range(xgrid.shape[0]):
    for j in range(ygrid.shape[0]):
        data = np.array([[xgrid[i, j], ygrid[i, j]]])
        Z[i, j] = rbfKernel.decision_function(data)[0]
        
# plotting SVM
plt.figure()
plt.contourf(xgrid, ygrid, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap='autumn')
plt.scatter(train_data.x1, train_data.x2, c=Y_train, cmap="coolwarm")
plt.scatter(rbfKernel.support_vectors_[:, 0], rbfKernel.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5)
plt.title('SVM Visualization: RBF')
plt.show()