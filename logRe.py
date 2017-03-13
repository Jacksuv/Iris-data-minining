from numpy import *
def sigmoid(inX):
	return 1.0/(1+exp(-inX))

def gradascent(X,y):
	m,n = shape(X)
	al = 0.005
	max = 500
	w = zeros((n,1))
	for k in range(max):
	    h =  sigmoid(X*w)
	    error = (y.T-h)
	    w = w + al*(X.T)*error
	return w

