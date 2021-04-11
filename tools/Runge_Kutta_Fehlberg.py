import numpy as np
from math import sqrt, isnan


class Runge_Kutta_Fehlberg:
	''' explicit, order 4'''
	s = 6
	a = [[0, 0, 0, 0,0,0],
         [0.25,0,0, 0, 0, 0],
         [3./32, 9./32, 0,0,0, 0],
         [1932./2197, -7200./2197, 7296./2197, 0,0,0],
         [439./216, -8., 3680./513, -845./4104,0,0],
         [-8./27, 2., -3544./2565, 1859./4104, -11./40,0]]
	b = [[16./135, 0, 6656./12825, 28561./56430, -9./50, 2./55],
         [25./216, 0, 1408./2565, 2197./4104, -1./5, 0]]
	c = [0, 0.25, 3./8, 12./13, 1., 0.5]
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)

def RK45(f,y_0,t_0,h,corrM):
	arraySize=y_0.size
	RK = Runge_Kutta_Fehlberg
	h_max=0.1
	h_min=0.0001
	t_n,x_n = t_0,y_0
	
	k = np.zeros((RK.s, arraySize), dtype=complex)
	while t_n<t_0+h_min:
		if h<h_min:
			h=h_min
		if h>h_max:
			h=h_max
	
		for s in range(RK.s):
			t_s = t_n + h*RK.c[s]         
			a_tmp = RK.a[s,:]        
			a_tmp = a_tmp[:,None]
			x_s = x_n + ((a_tmp*k).sum(0))
			k[s,:] =  h*f(t_s , x_s, corrM)

		delta_b=(np.subtract(RK.b[0,:],RK.b[1,:])[:,None]*k).sum(0)
		error=sqrt(np.dot(delta_b,delta_b.conjugate()).real)
		if isnan(error)==1:
			print('Fatal NaN Error')
			#t_n=t_n+1000
			#x_n4=0
			#break
#		error=np.abs(delta_b.real[0])
#		error=h**5
		delta_0=1.e-5

		if error<=delta_0:
			x_n4 = x_n + (RK.b[1,:, None]*k).sum(0)
			t_n=h+t_n
			h=h*1.2#0.98*h*(delta_0/error)**0.25
		else:
			h=h*0.9#0.98*h*(delta_0/error)**0.2
			#print(error)
			
	return np.array([x_n4,h,t_n,error])


def rkf( f, a, b, x0, tol, hmax, hmin, *params):
	"""Runge-Kutta-Fehlberg method to solve x' = f(x,t) with x(t[0]) = x0.

	USAGE:
		t, x = rkf(f, a, b, x0, tol, hmax, hmin)

	INPUT:
		f     - function equal to dx/dt = f(x,t)
		a     - left-hand endpoint of interval (initial condition is here)
		b     - right-hand endpoint of interval
		x0    - initial x value: x0 = x(a)
		tol   - maximum value of local truncation error estimate
		hmax  - maximum step size
		hmin  - minimum step size

	OUTPUT:
		t     - NumPy array of independent variable values
		x     - NumPy array of corresponding solution function values

	NOTES:
		This function implements 4th-5th order Runge-Kutta-Fehlberg Method
		to solve the initial value problem

			dx
			-- = f(x,t),     x(a) = x0
			dt

		on the interval [a,b].

		Based on pseudocode presented in "Numerical Analysis", 6th Edition,
		by Burden and Faires, Brooks-Cole, 1997.
	"""

	# Coefficients used to compute the independent variable argument of f

	a2  =   2.500000000000000e-01  #  1/4
	a3  =   3.750000000000000e-01  #  3/8
	a4  =   9.230769230769231e-01  #  12/13
	a5  =   1.000000000000000e+00  #  1
	a6  =   5.000000000000000e-01  #  1/2

	# Coefficients used to compute the dependent variable argument of f

	b21 =   2.500000000000000e-01  #  1/4
	b31 =   9.375000000000000e-02  #  3/32
	b32 =   2.812500000000000e-01  #  9/32
	b41 =   8.793809740555303e-01  #  1932/2197
	b42 =  -3.277196176604461e+00  # -7200/2197
	b43 =   3.320892125625853e+00  #  7296/2197
	b51 =   2.032407407407407e+00  #  439/216
	b52 =  -8.000000000000000e+00  # -8
	b53 =   7.173489278752436e+00  #  3680/513
	b54 =  -2.058966861598441e-01  # -845/4104
	b61 =  -2.962962962962963e-01  # -8/27
	b62 =   2.000000000000000e+00  #  2
	b63 =  -1.381676413255361e+00  # -3544/2565
	b64 =   4.529727095516569e-01  #  1859/4104
	b65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

	r1  =   2.777777777777778e-03  #  1/360
	r3  =  -2.994152046783626e-02  # -128/4275
	r4  =  -2.919989367357789e-02  # -2197/75240
	r5  =   2.000000000000000e-02  #  1/50
	r6  =   3.636363636363636e-02  #  2/55

    # Coefficients used to compute 4th order RK estimate

	c1  =   1.157407407407407e-01  #  25/216
	c3  =   5.489278752436647e-01  #  1408/2565
	c4  =   5.353313840155945e-01  #  2197/4104
	c5  =  -2.000000000000000e-01  # -1/5

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
  
	t = a
	x = x0
	h = 0.01#hmax

    # Initialize arrays that will be returned

	T = np.array( [t] )
	X = np.array( [x] )
	H = np.array( [h] )

	while t < b:

		# Adjust step size when we get to last interval

		if t + h > b:
			h = b - t;

		# Compute values needed to compute truncation error estimate and
		# the 4th order RK estimate.

		k1 = h * f( x, t, *params )
		k2 = h * f( x + b21 * k1, t + a2 * h, *params )
		k3 = h * f( x + b31 * k1 + b32 * k2, t + a3 * h, *params )
		k4 = h * f( x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h, *params )
		k5 = h * f( x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h, *params )
		k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
					t + a6 * h, *params )

		# Compute the estimate of the local truncation error.  If it's small
		# enough then we accept this step and save the 4th order estimate.
	
		r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
		if len( np.shape( r ) ) > 0:
			r = max( r )
		if np.isnan(r)==1:
			print('Fatal NaN Error')
			break
		if np.isinf(r)==1:
			print('infinity detected')
			break
		if r <= tol:
			t = t + h
			x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
			T = np.append( T, t )
			H = np.append( H, h )
			X = np.append( X, [x], 0 )
			
			h=h*1.2
		else:
			h=h*0.9

        # Now compute next step size, and make sure that it is not too big or
        # too small.

		#h = h * min( max( 0.94 * ( tol / r )**0.25, 0.1 ), 4.0 )
		
		#if h > hmax:
			#h = hmax elif
		if h < hmin:
			print("Error: stepsize should be smaller than %e." % hmin)
			print("at time t=%e" % t)
			print("and error r=%e" % r)
			break

    # endwhile

	return ( T, X, H)

	
	
def rkf5(selma, x0, y0, t0, h, *params):
	'''
	Adapted from
	http://people.sc.fsu.edu/~jburkardt/c_src/stochastic_rk/stochastic_rk.html
	'''
	a2  =   2.500000000000000e-01  #  1/4
	a3  =   3.750000000000000e-01  #  3/8
	a4  =   9.230769230769231e-01  #  12/13
	a5  =   1.000000000000000e+00  #  1
	a6  =   5.000000000000000e-01  #  1/2

	# Coefficients used to compute the dependent variable argument of f

	b21 =   2.500000000000000e-01  #  1/4
	b31 =   9.375000000000000e-02  #  3/32
	b32 =   2.812500000000000e-01  #  9/32
	b41 =   8.793809740555303e-01  #  1932/2197
	b42 =  -3.277196176604461e+00  # -7200/2197
	b43 =   3.320892125625853e+00  #  7296/2197
	b51 =   2.032407407407407e+00  #  439/216
	b52 =  -8.000000000000000e+00  # -8
	b53 =   7.173489278752436e+00  #  3680/513
	b54 =  -2.058966861598441e-01  # -845/4104
	b61 =  -2.962962962962963e-01  # -8/27
	b62 =   2.000000000000000e+00  #  2
	b63 =  -1.381676413255361e+00  # -3544/2565
	b64 =   4.529727095516569e-01  #  1859/4104
	b65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

	r1  =   2.777777777777778e-03  #  1/360
	r3  =  -2.994152046783626e-02  # -128/4275
	r4  =  -2.919989367357789e-02  # -2197/75240
	r5  =   2.000000000000000e-02  #  1/50
	r6  =   3.636363636363636e-02  #  2/55

    # Coefficients used to compute 4th order RK estimate

	c1  =   1.157407407407407e-01  #  25/216
	c3  =   5.489278752436647e-01  #  1408/2565
	c4  =   5.353313840155945e-01  #  2197/4104
	c5  =  -2.000000000000000e-01  # -1/5
	

	x = x0
	y = y0

	k1 = h * selma( x, y, *params )
	k2 = h * selma( x + b21 * k1[0], y + b21 * k1[1], *params )
	k3 = h * selma( x + b31 * k1[0] + b32 * k2[0], y + b31 * k1[1] + b32 * k2[1], *params )
	k4 = h * selma( x + b41 * k1[0] + b42 * k2[0] + b43 * k3[0], y + b41 * k1[1] + b42 * k2[1] + b43 * k3[1], *params )
	k5 = h * selma( x + b51 * k1[0] + b52 * k2[0] + b53 * k3[0] + b54 * k4[0], y + b51 * k1[1] + b52 * k2[1] + b53 * k3[1] + b54 * k4[1], *params )
	#k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
				#t + a6 * h, *params )

	# # return (xvec, yvec, time)
	return (x + c1 * k1[0] + c3 * k3[0] + c4 * k4[0] + c5 * k5[0], y + c1 * k1[1] + c3 * k3[1] + c4 * k4[1] + c5 * k5[1], t0+h)
	
def rkf5_1param(selma, x0, h, t, *params):
	'''
	Adapted from
	http://people.sc.fsu.edu/~jburkardt/c_src/stochastic_rk/stochastic_rk.html
	'''
	a2  =   2.500000000000000e-01  #  1/4
	a3  =   3.750000000000000e-01  #  3/8
	a4  =   9.230769230769231e-01  #  12/13
	a5  =   1.000000000000000e+00  #  1
	a6  =   5.000000000000000e-01  #  1/2

	# Coefficients used to compute the dependent variable argument of f

	b21 =   2.500000000000000e-01  #  1/4
	b31 =   9.375000000000000e-02  #  3/32
	b32 =   2.812500000000000e-01  #  9/32
	b41 =   8.793809740555303e-01  #  1932/2197
	b42 =  -3.277196176604461e+00  # -7200/2197
	b43 =   3.320892125625853e+00  #  7296/2197
	b51 =   2.032407407407407e+00  #  439/216
	b52 =  -8.000000000000000e+00  # -8
	b53 =   7.173489278752436e+00  #  3680/513
	b54 =  -2.058966861598441e-01  # -845/4104
	b61 =  -2.962962962962963e-01  # -8/27
	b62 =   2.000000000000000e+00  #  2
	b63 =  -1.381676413255361e+00  # -3544/2565
	b64 =   4.529727095516569e-01  #  1859/4104
	b65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

	r1  =   2.777777777777778e-03  #  1/360
	r3  =  -2.994152046783626e-02  # -128/4275
	r4  =  -2.919989367357789e-02  # -2197/75240
	r5  =   2.000000000000000e-02  #  1/50
	r6  =   3.636363636363636e-02  #  2/55

    # Coefficients used to compute 4th order RK estimate

	c1  =   1.157407407407407e-01  #  25/216
	c3  =   5.489278752436647e-01  #  1408/2565
	c4  =   5.353313840155945e-01  #  2197/4104
	c5  =  -2.000000000000000e-01  # -1/5
	

	x = x0

	k1 = h * selma( x, t, *params )
	k2 = h * selma( x + b21 * k1, t + a2 * h, *params )
	k3 = h * selma( x + b31 * k1 + b32 * k2, t + a3 * h, *params )
	k4 = h * selma( x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h, *params )
	k5 = h * selma( x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h, *params )
	#k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
				#t + a6 * h, *params )

	# # return (xvec, yvec, time)
	return x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
	
def euler_1param(selma, x0, h, t, *params):
	x = x0
	return x + h * selma(x, t, *params)

def rkf4(c, evolve, dt, *params):
	'''
	Adapted from
	http://people.sc.fsu.edu/~jburkardt/c_src/stochastic_rk/stochastic_rk.html
	'''
	a21 =   2.71644396264860
	a31 = - 6.95653259006152
	a32 =   0.78313689457981
	a41 =   0.0
	a42 =   0.48257353309214
	a43 =   0.26171080165848
	a51 =   0.47012396888046
	a52 =   0.36597075368373
	a53 =   0.08906615686702
	a54 =   0.07483912056879

	q1 =   2.12709852335625
	q2 =   2.73245878238737
	q3 =  11.22760917474960
	q4 =  13.36199560336697

	x1 = c
	k1 = dt * evolve(x1, *params) + T.sqrt(dt) * c * rv_n

	x2 = x1 + a21 * k1
	k2 = dt * evolve(x2, *params) + T.sqrt(dt) * c * rv_n

	x3 = x1 + a31 * k1 + a32 * k2
	k3 = dt * evolve(x3, *params) + T.sqrt(dt) * c * rv_n

	x4 = x1 + a41 * k1 + a42 * k2
	k4 = dt * evolve(x4, *params) + T.sqrt(dt) * c * rv_n

	return T.cast(x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, 'float32')

