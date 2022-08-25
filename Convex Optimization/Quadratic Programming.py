# Let a polyhedron is represented by Ax <= b. 
# Find a point u in this polyhedron and distance(x, u) is min
# --> x = argmin 1/2 ||x - u|| ^ 2  subject to: Gx <= h
# cost function: 1/2 ||x - u|| ^ 2 = 1/2 (x - u)^T . (x - u) 
# = 1/2.x^T.x - u^T.x + 1/2.u^t.u
# form of quadratic programming: 1/2.x^t.P.x + q^T.x + r 
# P = I, q = -u, r = 1/2.u^t.u  
# If polyhedron is represented by: x,y >= 0; x + 4y <= 32; x + y <= 10; 2x + y <= 16

from cvxopt import matrix, solvers
P = matrix([[1., 0.], [0., 1.]])
q = matrix([-10., -10.])
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0])

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h)

print('Solution:')
print(sol['x'])