import numpy as np

STUDENT = {'name': 'Omri Ben Hemo Ben Ganon',
           'ID': '313255242_318731007',
           }

def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.

        # Modify x[ix] with h defined above to compute the numerical gradient
        x_before_change = x[ix]  # Store the original value
        x[ix] = x_before_change + h  # Perturb the element at index ix
        fx_plus_h, _ = f(x)  # Evaluate f(x + h)
        x[ix] = x_before_change - h  # Perturb the element at index ix again
        fx_minus_h, _ = f(x)  # Evaluate f(x - h)
        x[ix] = x_before_change  # Restore the original value

        numeric_gradient = (fx_plus_h - fx_minus_h) / (2 * h)  # Compute the numerical gradient

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
            return
    
        it.iternext() # Step to next index

    print("Gradient check passed!")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradient_check(quad, np.array(123.456))      # scalar test
    gradient_check(quad, np.random.randn(3,))    # 1-D test
    gradient_check(quad, np.random.randn(4,5))   # 2-D test
    #more sanity checks here
    print()
    print("Running my sanity checks...")
    # Test case 1
    f1 = lambda x: (np.sin(x), np.cos(x))
    x1 = np.array([0.5])
    gradient_check(f1, x1)

    # Test case 2
    f2 = lambda x: (np.sum(x ** 3), 3 * x ** 2)
    x2 = np.array([1.0, 2.0, 3.0])
    gradient_check(f2, x2)

    # Test case 3
    f3 = lambda x: (np.sum(np.tanh(x)), 1 - np.tanh(x) ** 2)
    x3 = np.random.randn(5, )
    gradient_check(f3, x3)

    print()

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()
