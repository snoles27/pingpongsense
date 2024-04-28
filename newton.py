import numpy as np 
import warnings
from numpy import linalg 
from typing import Callable

def newtonSolver(f:Callable[[np.ndarray, tuple], np.ndarray], df:Callable[[np.ndarray, tuple], np.ndarray], args:tuple, x0:np.ndarray, thresh = 1e-12, iterLim = 100) -> np.ndarray:
    """
    newton solver that finds solutions to f(x, *args) = 0 where f returns size ax1 and x is size bx1
    f: function looking for the roots of. Function should accept nd.array of size b as an argument
    df: function that takes in ndarray (ax1) and returns the jacobian of f evaluated at x size (axb)
    args: argumens passed to f and df 
    """

    numIter = 0
    magChange = float('inf')
    xi = x0
    while magChange > thresh and numIter < iterLim:

        #calculate the function and the jacobian at this value of xi
        fi = f(xi, args)
        dfi = df(xi, args)

        xi1 = newtonStep(xi, fi, dfi)

        #do some checks
        magChange = linalg.norm(xi1 - xi)
        numIter += 1

        xi = xi1
    
    if numIter >= iterLim:
        warnings.warn("Reached iteration limit")
    
    return xi


def newtonStep(xi:np.ndarray, fi:np.ndarray, dfi:np.ndarray) -> np.ndarray:
    """
    Performs one step of newton iteration given the currnet value, the function evaluated at that value and the jacobian eveluated at that value
    """

    dx = linalg.solve(dfi, -1 * fi)
    return xi + dx


if __name__ == "__main__":
    pass
