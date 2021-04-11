import numpy as np
from math import sqrt, isnan


def rkf5_1param(fct, x0, h, t, *params):
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

	k1 = h * fct( x, t, *params )
	k2 = h * fct( x + b21 * k1, t + a2 * h, *params )
	k3 = h * fct( x + b31 * k1 + b32 * k2, t + a3 * h, *params )
	k4 = h * fct( x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h, *params )
	k5 = h * fct( x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h, *params )
	#k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
				#t + a6 * h, *params )

	return x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
	
def euler_1param(fct, x0, h, t, *params):
	x = x0
	return x + h * fct(x, t, *params)

