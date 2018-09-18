from numpy import *
from a1 import *

def test_gauss_multiple():
    a = array([[6, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
    b = array([[-14, 22], [36, -18], [6, 7]], dtype=float_)
    gauss_multiple(a, b) # result is now in b
    na = array([[6, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
    nb = array([[-14, 22], [36, -18], [6, 7]], dtype=float_)
    from numpy.linalg import solve
    solution = solve(na, nb)
    epsilon = 10E-15
    assert(sum(abs(solution - b)) < epsilon)

def test_gauss_multiple_pivot():
    a = array([[0, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
    b = array([[-14, 22], [36, -18], [6, 7]], dtype=float_)
    gauss_multiple_pivot(a, b) # result is now in b
    na = array([[0, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
    nb = array([[-14, 22], [36, -18], [6, 7]], dtype=float_)
    from numpy.linalg import solve
    solution = solve(na, nb)
    epsilon = 10E-15
    assert(sum(abs(solution - b)) < epsilon)