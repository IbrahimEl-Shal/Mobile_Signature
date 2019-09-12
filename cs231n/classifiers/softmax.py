import numpy as np
from random import shuffle
from past.builtins import xrange

#X = np.load("npy_dataset/X_dev.npy")
#y = np.load("npy_dataset/y_dev.npy")
#W = np.random.randn(3073, 10) * 0.0001

# In[]:
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  p = np.zeros(num_classes) 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  
  for i in xrange(num_train):
      
    class_score = scores[i,:]
    ## numerical stability by subtracting max from score vector. ##
    ## First Shift of values ##
    class_score -= np.max(class_score)  
    correct_class_score = class_score[y[i]] #Get correct labels
    
    for j in xrange(num_classes):
        p = np.exp(class_score[j]) / np.sum(np.exp(class_score))   
        # Gradient calculation.
        if j == y[i]:
            dW[:, j] += (-1 + p) * X[i]
        else:
            dW[:, j] += p * X[i]
        
    # Calculate loss for this example.
    Res = -correct_class_score + np.log(np.sum(np.exp(class_score)))
    loss += Res  
    
  # we want it to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # regularize the weights
  dW += reg*W 
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)  
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  class_score = scores - np.max(scores, axis= 1)[:,np.newaxis]
   
  # Calculate softmax scores.
  softmax_scores = np.exp(class_score)/ np.sum(np.exp(class_score), axis=1)[:,np.newaxis]

  # Calculate dScore, the gradient wrt. softmax scores.
  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1

  # Backprop dScore to calculate dW, then average and add regularisation.
  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += reg*W

  # Calculate our cross entropy Loss.
  correct_class_scores = np.choose(y, class_score.T)  # Size N vector
  loss = -correct_class_scores + np.log(np.sum(np.exp(class_score), axis=1))
  loss = np.sum(loss)

  # Average our loss then add regularisation.
  loss /= num_train
  loss += reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

