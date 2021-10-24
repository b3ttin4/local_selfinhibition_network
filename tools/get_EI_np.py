import sys
import numpy as np



def conn_profile(x, y, sigma_x, sigma_y, ampl, profile_mode):
	"""
	generates connectivitiy profile

	input:
	x: x-coordinates
	y: y-coordinates
	sigma_x: standard deviation in x direction
	sigma_y: standard deviation in y direction
	ampl: amplitude of connectivity weights
	profile_mode: str, either 'gaussian' or 'exponential'

	output:
	r: connectivity matrix
	"""
	if profile_mode=='gaussian':
		if deltay is not None:
			delta = 1.*x**2/sigma_x**2 + 1.*y**2/sigma_y**2
			r = ampl*np.exp(-delta/2.)
		else:
			r = ampl * 1./np.sqrt(2*np.pi)/sigma_x*np.exp(-1.*x**2/2./sigma_x**2)
		
	elif profile_mode=='exponential':
		if deltay is not None:
			delta = np.sqrt(1.*x**2/sigma_x**2 + 1.*y**2/sigma_y**2)
			r = ampl*np.exp(-delta/2.)
		else:
			r = ampl * 1./2./sigma_x*np.exp(-1.*abs(x)/2./sigma_x)
	return r



def gauss_1d_pbc(coord, mu, sigma, constant, delta_space=None, profile_mode='gaussian'):
	''' return 1d Gaussian/exponentially distributed connectivity profile for 2d

	input:
	coord_x: x coordinates
	coord_y: y coordinates
	mu: mean 
	sigma: standard deviation
	delta_space: spacing of coordinates, default is 1
	profile_mode: str, 'gaussian' or 'exponential'

	output:
	r: connectivity matrix
	'''
	if isinstance(sigma,np.ndarray):
		sigma = sigma[:,None]
	
	if isinstance(coord,list):
		coord1, coord2 = coord
		N = coord1.size
		M = coord2.size
		if delta_space is None:
			delta_space = 1.*N/M
		
		deltax = np.abs(coord1[:,None]-coord2[None,:]*delta_space - mu)
		deltax = np.nanmin([deltax, 1-deltax],axis=0)
	else:
		N = coord.shape[0]
		deltax = np.abs(coord[:,None]-coord[None,:] - mu)
		deltax = np.nanmin([deltax, 1-deltax],axis=0)
	
	r = conn_profile(deltax, None, sigma, None, 1., profile_mode)

	return r
	



def gauss_2d_pbc(coord_x, coord_y, mu, sigma, constant, delta_space=None, profile_mode='gaussian'):
	''' return 2d Gaussian/exponentially distributed connectivity profile for 2d

	input:
	coord_x: x coordinates
	coord_y: y coordinates
	mu: mean 
	sigma: standard deviation
	delta_space: spacing of coordinates, default is 1
	profile_mode: str, 'gaussian' or 'exponential'

	output:
	r: connectivity matrix
	'''
	if isinstance(mu,list):
		mu_x, mu_y = mu
	else:
		mu_x, mu_y = mu, mu
	if isinstance(sigma,list):
		sigma_x, sigma_y = sigma
	else:
		sigma_x, sigma_y = sigma, sigma
	
	if isinstance(coord_x,list):
		coordx1,coordx2 = coord_x[0],coord_x[1]
		coordy1,coordy2 = coord_y[0],coord_y[1]
		M1,N1 = coordx1.shape		#N: size in x; M: size in y-direction
		M2,N2 = coordx2.shape
		if delta_space is None:
			delta_spacex = 1.*N1/N2
			delta_spacey = 1.*M1/M2
		else:
			delta_spacex = delta_space
			delta_spacey = delta_space
			
		deltax = np.abs(coordx1[:,:,None,None]-coordx2[None,None,:,:]*delta_spacex - mu_x)
		deltay = np.abs(coordy1[:,:,None,None]-coordy2[None,None,:,:]*delta_spacey - mu_y)

	else:
		if delta_space is None:
			delta_space = 1
		M,N = coord_x.shape		#N: size in x; M: size in y-direction
		
		deltax = np.abs(coord_x[:,:,None,None]-coord_x[None,None,:,:]*delta_space - mu_x)
		deltay = np.abs(coord_y[:,:,None,None]-coord_y[None,None,:,:]*delta_space - mu_y)
		
	deltax = np.nanmin([deltax, 1-deltax],axis=0)
	deltay = np.nanmin([deltay, 1-deltay],axis=0)
	
	# apply connectivity profile
	r = conn_profile(deltax, deltay, sigma_x, sigma_y, constant, profile_mode)
	
	# normalise integral over connectivity coming from one neuron to 1
	N1,M1,N2,M2 = r.shape
	normalising_factor = np.mean(r.reshape(N1*M1,N2*M2),axis=1).reshape(N1,M1)
	r = r/normalising_factor[:,:,None,None]

	return r


def get_EI_2d(sigmas, Ampl, nE, nI, alpha, conv_params):
	""" generating 2d connectivity matrix
	
	input:
	sigmas: list or float of standard deviation of connectivity
	Ampl: list or float of standard deviation of connectivity
	nE: size of excitatory units
	nI: size of inhibitory units
	alpha: float, 0<=alpha<=1, strength of self-inhibitory connections
	kwargs: can contain 'add_autapses' and  must contain 'profile_mode'

	output:
	connectivity matrix
	"""
	sigEE = sigmas[0]
	sigEI = sigmas[1]
	sigIE = sigmas[2]
	sigII = sigmas[3]

	aEE = Ampl[0] 
	aEI = Ampl[1]
	aIE = Ampl[2]
	aII = Ampl[3]
	aII_a = aII*alpha
	aII_s = aII*(1-alpha)
	
	nE2 = nE*nE
	nI2 = nI*nI
	nEI2 = nI*nE

	coord_e = np.linspace(0,1,nE,endpoint=False)
	coord_i = np.linspace(0,1,nI,endpoint=False)
	
	##check for chosen connectivity profile
	try:
		profile_mode = conv_params['profile_mode']
	except:
		profile_mode = 'gaussian' # default mode

		
	
	##EE matrix
	sig = sigEE
	coord_x, coord_y = np.meshgrid(coord_e, coord_e)

	Mee = gauss_2d_pbc(coord_x, coord_y, 0, sig, 1, delta_space=1.,\
						profile_mode=profile_mode)
	Mee = (Mee/np.mean(Mee)).reshape(nE2,nE2)
	MEE = aEE*Mee
	del Mee

	##EI matrix
	sig = sigEI;
	coord_x1, coord_y1 = np.meshgrid(coord_e, coord_e)
	coord_x2, coord_y2 = np.meshgrid(coord_i, coord_i)

	Mei = gauss_2d_pbc([coord_x1,coord_x2], [coord_y1,coord_y2], 0, sig, 1.,\
						delta_space=1., profile_mode=profile_mode)
	Mei = (Mei/np.mean(Mei)).reshape(nE2,nI2)
	MEI = aEI*Mei
	del Mei

	##IE matrix
	sig = sigIE
	coord_x1, coord_y1 = np.meshgrid(coord_i, coord_i)
	coord_x2, coord_y2 = np.meshgrid(coord_e, coord_e)

	Mie = gauss_2d_pbc([coord_x1,coord_x2], [coord_y1,coord_y2], 0, sig, 1.,\
						delta_space=1., profile_mode=profile_mode)
	Mie = (Mie/np.mean(Mie)).reshape(nI2,nE2)
	MIE = aIE*Mie
	del Mie


	##II matrix
	sig = sigII
	coord_x, coord_y = np.meshgrid(coord_i, coord_i)
	Mii = gauss_2d_pbc(coord_x, coord_y, 0, sig, 1, delta_space=1.,\
						profile_mode=profile_mode)
	Mii = (Mii/np.mean(Mii)).reshape(nI2,nI2)
	MII = Mii*aII_s
	del Mii
	
	try:
		if conv_params['add_autapses']:
			print('add_autapses',MII[:3,:3]);sys.stdout.flush()
			Mii_autapse = gauss_2d_pbc(coord_x, coord_y, 0, 0.1/100., 1,\
				delta_space=1., profile_mode=profile_mode)
		
			Mii_autapse = aII_a*(Mii_autapse/np.mean(Mii_autapse)).reshape(nI2,nI2)
			MII += Mii_autapse
	except:
		pass

	M1 = np.zeros((nI2+nE2,nI2+nE2))
	M1[:nE2,:nE2] = MEE
	M1[:nE2,nE2:] = -MIE.T
	M1[nE2:,:nE2] = MEI.T
	M1[nE2:,nE2:] = -MII
	
	M1 = 1.*M1/nE2#(nI2+nE2)#
	
	return M1


def get_EI_1d(sigmas, Ampl, nE, nI, alpha, **kwargs):
	""" generating 1d connectivity matrix
	
	input:
	sigmas: list or float of standard deviation of connectivity
	Ampl: list or float of standard deviation of connectivity
	nE: size of excitatory units
	nI: size of inhibitory units
	alpha: float, 0<=alpha<=1, strength of self-inhibitory connections
	kwargs: can contain 'add_autapses' and  must contain 'profile_mode'

	output:
	connectivity matrix
	"""
	sigEE = sigmas[0]
	sigEI = sigmas[1]
	sigIE = sigmas[2]
	sigII = sigmas[3]

	aEE = Ampl[0]
	aEI = Ampl[1]
	aIE = Ampl[2]
	aII = Ampl[3]
	aII_a = aII*alpha
	aII_s = aII*(1-alpha)
	
	##check for chosen connectivity profile
	profile_mode = kwargs['profile_mode']
	
	##EE matrix
	sig = sigEE
	coord_x = np.linspace(0,1,nE,endpoint=False)
	
	Mee = gauss_1d_pbc(coord_x, 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mee = Mee/np.mean(Mee)
	MEE = Mee * aEE
	del Mee
	
	
	##E to I matrix
	sig = sigEI;
	coord1, coord2 = np.linspace(0,1,nE,endpoint=False),np.linspace(0,1,nI,endpoint=False)

	Mei = gauss_1d_pbc([coord1, coord2], 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mei = Mei/np.mean(Mei)
	MEI = Mei * aEI
	del Mei


	##I to E matrix
	sig = sigIE
	coord1, coord2 = np.linspace(0,1,nI,endpoint=False),np.linspace(0,1,nE,endpoint=False)

	Mie = gauss_1d_pbc([coord1, coord2], 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mie = Mie/np.mean(Mie)
	MIE = Mie * aIE
	del Mie


	##II matrix
	sig = sigII
	coord_x = np.linspace(0,1,nI,endpoint=False)

	Mii = gauss_1d_pbc(coord_x, 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mii = Mii/np.mean(Mii)
	MII = Mii * aII_s
	del Mii
	
	if "add_autapses" in kwargs.keys():
		if kwargs['add_autapses']:
			print('add_autapses',MII[:3,:3]);sys.stdout.flush()
			Mii_autapse = gauss_1d_pbc(coord_x, 0, 0.1/100., 1, delta_space=1.,\
										profile_mode=profile_mode)
			Mii_autapse = aII_a*(Mii_autapse/np.mean(Mii_autapse)).reshape(nI,nI)
			MII += Mii_autapse


	M1 = np.zeros((nE+nI,nE+nI))
	M1[:nE,:nE] = MEE
	M1[:nE,nE:] = -MIE.T
	M1[nE:,:nE] = MEI.T
	M1[nE:,nE:] = -MII
	M1 = 1.*M1/nE
	
	return M1


def get_EI(dim, sigmas, Ampl, nE, nI, alpha, **kwargs):
	""" wrapper function for connectivity matrix
	
	input:
	dim: dimensionality, 1 or 2
	sigmas: list or float of standard deviation of connectivity
	Ampl: list or float of standard deviation of connectivity
	nE: size of excitatory units
	nI: size of inhibitory units
	alpha: float, 0<=alpha<=1, strength of self-inhibitory connections
	kwargs: can contain 'add_autapses' and  must contain 'profile_mode'

	output:
	connectivity matrix
	"""
	if dim==1:
		return get_EI_1d(sigmas, Ampl, nE, nI, alpha, **kwargs)
	elif dim==2:
		return get_EI_2d(sigmas, Ampl, nE, nI, alpha, **kwargs)


if __name__=="__main__":
	import matplotlib.pyplot as plt
	
	conn_strength = np.array([22.2, 21.6, 21.6, 19.8])
	conn_width_mean = np.array([3.3, 3.3, 2.5, 2.5])/40.

	VERSION = 112
	index = 0
	add_autapses = True
	
	alpha = 1.
	nE = 30#80#
	nI = 30#80#
	rng = np.random.RandomState(465+VERSION*101+index)
	conv_params = {"profile_mode" : "gaussian", "add_autapses" : add_autapses}

	M1, ew = get_EI_1d(conn_width_mean, conn_strength, nE, nI, alpha, rng, conv_params)
	print(M1.shape, ew)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	im = ax.imshow(M1,interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)
	
	plt.show()
	



