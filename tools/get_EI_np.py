import sys
import numpy as np



def conn_profile(deltax, deltay, sigma_x, sigma_y, constant, profile_mode):
	if profile_mode=='gaussian':
		if deltay is not None:
			delta = 1.*deltax**2/sigma_x**2 + 1.*deltay**2/sigma_y**2
			r = constant*np.exp(-delta/2.)
		else:
			r = constant * 1./np.sqrt(2*np.pi)/sigma_x*np.exp(-1.*deltax**2/2./sigma_x**2)
		
	elif profile_mode=='exponential':
		if deltay is not None:
			delta = np.sqrt(1.*deltax**2/sigma_x**2 + 1.*deltay**2/sigma_y**2)
			r = constant*np.exp(-delta/2.)
		else:
			r = constant * 1./2./sigma_x*np.exp(-1.*abs(deltax)/2./sigma_x)
	#print('r',np.nanmin(r),np.nanmax(r))		
	return r



def gauss_1d_pbc(coord, mu, sigma, constant, delta_space=None,profile_mode='gaussian'):
	## sigma can be int or array of ints with shape NxM
	## to introduce heterogeneities
	
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
	
	#print('g0',np.nanmean(r.reshape(N,N),axis=0)[:10])
	#print('g1',np.nanmean(r.reshape(N,N),axis=1)[:10])
	return r
	



def gauss_2d_pbc(coord_x, coord_y, mu, sigma, constant, delta_space=None, profile_mode='gaussian'):
	''' return Gaussian distributed connectivity profile for 2d'''
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
	
	## apply connectivity profile
	r = conn_profile(deltax, deltay, sigma_x, sigma_y, constant, profile_mode)
	
	
	## Exponential fct
	#r = constant*np.exp(-np.sqrt(delta))
	
	## normalise integral over connectivity coming from one neuron to 1
	N1,M1,N2,M2 = r.shape
	normalising_factor = np.mean(r.reshape(N1*M1,N2*M2),axis=1).reshape(N1,M1)
	r = r/normalising_factor[:,:,None,None]
	
	#print('g0',np.nanmean(r.reshape(N1*M1,N2*M2),axis=0)[:10])
	#print('g1',np.nanmean(r.reshape(N1*M1,N2*M2),axis=1)[:10])
	#print('normalising_factor',normalising_factor)
	return r

def get_EI_2d(sigmas, Vars, nE, nI, alpha, conv_params):
	sigEE = sigmas[0]
	sigEI = sigmas[1]
	sigIE = sigmas[2]
	sigII = sigmas[3]

	varEE = Vars[0] 
	varEI = Vars[1]
	varIE = Vars[2]
	varII = Vars[3]
	varII_a = varII*alpha
	varII_s = varII*(1-alpha)
	
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
	MEE = varEE*Mee
	del Mee

	##EI matrix
	sig = sigEI;
	coord_x1, coord_y1 = np.meshgrid(coord_e, coord_e)
	coord_x2, coord_y2 = np.meshgrid(coord_i, coord_i)

	Mei = gauss_2d_pbc([coord_x1,coord_x2], [coord_y1,coord_y2], 0, sig, 1.,\
		delta_space=1., profile_mode=profile_mode)
	Mei = (Mei/np.mean(Mei)).reshape(nE2,nI2)
	MEI = varEI*Mei
	del Mei

	##IE matrix
	sig = sigIE
	coord_x1, coord_y1 = np.meshgrid(coord_i, coord_i)
	coord_x2, coord_y2 = np.meshgrid(coord_e, coord_e)

	Mie = gauss_2d_pbc([coord_x1,coord_x2], [coord_y1,coord_y2], 0, sig, 1.,\
		delta_space=1., profile_mode=profile_mode)
	Mie = (Mie/np.mean(Mie)).reshape(nI2,nE2)
	MIE = varIE*Mie
	del Mie


	##II matrix
	sig = sigII
	coord_x, coord_y = np.meshgrid(coord_i, coord_i)
	Mii = gauss_2d_pbc(coord_x, coord_y, 0, sig, 1, delta_space=1.,\
		profile_mode=profile_mode)
	Mii = (Mii/np.mean(Mii)).reshape(nI2,nI2)
	MII = Mii*varII_s
	del Mii
	
	try:
		if conv_params['add_autapses']:
			print('add_autapses',MII[:3,:3]);sys.stdout.flush()
			Mii_autapse = gauss_2d_pbc(coord_x, coord_y, 0, 0.1/100., 1,\
				delta_space=1., profile_mode=profile_mode)
		
			Mii_autapse = varII_a*(Mii_autapse/np.mean(Mii_autapse)).reshape(nI2,nI2)
			MII += Mii_autapse
	except:
		pass

	M1 = np.zeros((nI2+nE2,nI2+nE2))
	M1[:nE2,:nE2] = MEE
	M1[:nE2,nE2:] = -MIE.T
	M1[nE2:,:nE2] = MEI.T
	M1[nE2:,nE2:] = -MII
	
	M1 = 1.*M1/(nI2+nE2)
	
	return M1


def get_EI_1d(sigmas, Vars, nE, nI, alpha, conv_params=None):
	sigEE = sigmas[0]
	sigEI = sigmas[1]
	sigIE = sigmas[2]
	sigII = sigmas[3]

	varEE = Vars[0]
	varEI = Vars[1]
	varIE = Vars[2]
	varII = Vars[3]
	varII_a = varII*alpha
	varII_s = varII*(1-alpha)
	
	##check for chosen connectivity profile
	profile_mode = conv_params['profile_mode']
	
	##EE matrix
	sig = sigEE
	coord_x = np.linspace(0,1,nE,endpoint=False)
	
	Mee = gauss_1d_pbc(coord_x, 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mee = Mee/np.mean(Mee)
	MEE = Mee * varEE
	del Mee
	
	
	##E to I matrix
	sig = sigEI;
	coord1, coord2 = np.linspace(0,1,nE,endpoint=False),np.linspace(0,1,nI,endpoint=False)

	Mei = gauss_1d_pbc([coord1, coord2], 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mei = Mei/np.mean(Mei)
	MEI = Mei * varEI
	del Mei


	##I to E matrix
	sig = sigIE
	coord1, coord2 = np.linspace(0,1,nI,endpoint=False),np.linspace(0,1,nE,endpoint=False)

	Mie = gauss_1d_pbc([coord1, coord2], 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mie = Mie/np.mean(Mie)
	MIE = Mie * varIE
	del Mie


	##II matrix
	sig = sigII
	coord_x = np.linspace(0,1,nI,endpoint=False)

	Mii = gauss_1d_pbc(coord_x, 0, sig, 1, delta_space=1., profile_mode=profile_mode)
	Mii = Mii/np.mean(Mii)
	MII = Mii * varII_s
	del Mii
	
	if conv_params is not None:
		if conv_params['add_autapses']:
			print('add_autapses',MII[:3,:3]);sys.stdout.flush()
			Mii_autapse = gauss_1d_pbc(coord_x, 0, 0.1/100., 1, delta_space=1., profile_mode=profile_mode)
			Mii_autapse = varII_a*(Mii_autapse/np.mean(Mii_autapse)).reshape(nI,nI)#2.8
			MII += Mii_autapse


	M1 = np.zeros((nE+nI,nE+nI))
	M1[:nE,:nE] = MEE
	M1[:nE,nE:] = -MIE.T
	M1[nE:,:nE] = MEI.T
	M1[nE:,nE:] = -MII
	M1 = 1.*M1/nE
	
	return M1

def get_EI(dim, sigmas, Vars, nE, nI, alpha, conv_params=None):
	if dim==1:
		return get_EI_1d(sigmas, Vars, nE, nI, alpha, conv_params)
	elif dim==2:
		return get_EI_2d(sigmas, Vars, nE, nI, alpha, conv_params)


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
	



