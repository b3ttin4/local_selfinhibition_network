import numpy as np

## Fit functions
def gauss(x,sigma,ampl):
	return ampl*np.exp(-(x-nE/2.)**2/2./sigma**2)

def cosine(x,k,sigma,A):
	return A*np.cos(x*k)*np.exp(-(x)**2/2./sigma**2)
	

## Connectivity
def H(k,sigma):
	return np.exp(-k**2/2.*sigma*sigma)

## Trace, Determinant and Eigenvalue
def tracek(k,aee,aii,see,sii,tau=1,alpha=0):
	aii_s = aii*(1-alpha)
	aii_a = aii*alpha
	return -1 - (1 + aii_s*H(k,sii) + aii_a)/tau + aee*H(k,see)

def detk(k,aee,aeiaie,aii,see,sei,sii,tau=1,alpha=0):
	aii_s = aii*(1-alpha)
	aii_a = aii*alpha
	return ((1 + aii_s*H(k,sii) + aii_a)*(1 - aee*H(k,see)) + aeiaie*H(k,see)*H(k,sei))/tau

def eigval1(k,aee,aeiaie,aii,see,sei,sii,tau=1,alpha=0):
	tr = tracek(k,aee,aii,see,sii,tau,alpha)
	arg = tr**2 - 4*detk(k,aee,aeiaie,aii,see,sei,sii,tau,alpha)
	
	sign = np.ones_like(arg,dtype=float)
	sign[arg<0] = -1
	factor = np.ones_like(arg,dtype=complex)
	factor[arg<0] = 1j
	return tr/2. + factor*1./2*np.sqrt(arg*sign)

def eigval2(k,aee,aeiaie,aii,see,sei,sii,tau=1,alpha=0):
	tr = tracek(k,aee,aii,see,sii,tau,alpha)
	arg = tr**2 - 4*detk(k,aee,aeiaie,aii,see,sei,sii,tau,alpha)
	sign = np.ones_like(arg,dtype=float)
	sign[arg<0] = -1
	factor = np.ones_like(arg,dtype=complex)
	factor[arg<0] = 1j
	return tr/2. - factor*1./2*np.sqrt(arg*sign)

## helper functions
def c1(k,aee,aei,aii,see,sii,alpha):
	aii_s = aii*(1-alpha)
	aii_a = aii*alpha
	return (aee*H(k,see) + aii_a + aii_s*H(k,sii))/(2.*aei*H(k,see))

def c2(k,aee,aei,aie,aii,see,sii,alpha):
	aii_s = aii*(1-alpha)
	aii_a = aii*alpha
	radix = -4*aei*aie*H(k,sii)*H(k,see) + (aee*H(k,see) + aii_a + aii_s*H(k,sii))**2

	if (isinstance(k,np.ndarray) or isinstance(k,list)):
		if all(radix>0):
			return np.sqrt(radix)/(2.*aei*H(k,see))
		else:
			sqrt_radix = (1.*np.sqrt(radix)).astype(complex)
			sqrt_radix[radix<0] = (1j*np.sqrt(-radix))[radix<0]
			return sqrt_radix/(2.*aei*H(k,see))
	else:
		if radix>0:
			return np.sqrt(radix)/(2.*aei*H(k,see))
		else:
			return 1j*np.sqrt(-radix)/(2.*aei*H(k,see))

def ffct(k,aee,aei,aie,aii,see,sii,alpha):
	c1k = c1(k,aee,aie,aii,see,sii,alpha)
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	return 1./2./c2k*(c1k+c2k)

def ffct2(k,aee,aei,aie,aii,see,sii,alpha):
	c1k = c1(k,aee,aie,aii,see,sii,alpha)
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	return -1./2./c2k*(c1k-c2k)

def gfct(k,aee,aei,aie,aii,see,sii,alpha):
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	return 1./2./c2k

def gfct2(k,aee,aei,aie,aii,see,sii,alpha):
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	return -1./2./c2k

## Helper functions for pulse response
def effsigmaE(k,aee,aei,aie,aii,see,sii,alpha,t):
	''' outputs standard deviation of Gaussian
	in E response to pulse input; see Eq. () in manuscript '''
	fk = ffct(k,aee,aie,aei,aii,see,sii,alpha)
	dk = k[1]-k[0]
	f1k = np.gradient(fk,dk)
	f2k = np.gradient(f1k,dk)
	
	lk = eigval1(k,aee,aei*aie,aii,see,sii,sii,tau=1,alpha=alpha)
	l2k = np.gradient(np.gradient(lk,dk),dk)
	
	c1k = c1(k,aee,aie,aii,see,sii,alpha)
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	
	fidx = np.nanargmax(np.real(lk*t + np.log(fk)))
	return np.real(-l2k[fidx]*t - (f2k[fidx]*fk[fidx] - f1k[fidx]**2)/fk[fidx]**2)/see**2

def effsigmaI(k,aee,aei,aie,aii,see,sii,alpha,t):
	''' outputs standard deviation of Gaussian
	in I response to pulse input; see Eq. () in manuscript '''
	gk = gfct(k,aee,aie,aei,aii,see,sii,alpha)
	dk = k[1]-k[0]
	g1k = np.gradient(gk,dk)
	g2k = np.gradient(g1k,dk)
	
	lk = eigval1(k,aee,aei*aie,aii,see,sii,sii,tau=1,alpha=alpha)
	l2k = np.gradient(np.gradient(lk,dk),dk)
	
	c1k = c1(k,aee,aie,aii,see,sii,alpha)
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	
	gidx = np.nanargmax(np.real(lk*t + np.log(gk)))
	return np.real(-l2k[gidx]*t - (g2k[gidx]*gk[gidx] - g1k[gidx]**2)/gk[gidx]**2)/see**2

def deltasigma(k,aee,aei,aie,aii,see,sii,alpha,t):
	fk = ffct(k,aee,aie,aei,aii,see,sii,alpha)
	gk = gfct(k,aee,aie,aei,aii,see,sii,alpha)
	dk = k[1]-k[0]
	lk = eigval1(k,aee,aei*aie,aii,see,sii,sii,tau=1,alpha=alpha)

	fidx = np.nanargmax(np.real(lk*t + np.log(fk)))
	gidx = np.nanargmax(np.real(lk*t + np.log(gk)))
	
	kE_eff_sq = k[fidx]
	kI_eff_sq = k[gidx]
	diff_lambda_pxl = (1./kI_eff_sq - 1./kE_eff_sq)*2.*np.pi
	## convert to simulation box = 1, units in see
	return diff_lambda_pxl


def response_smalltime(k,aee,aie,aei,aii,see,sii,alpha,times):
	'''for small times response resembles gaussian 
	so returns width of gaussian here (up to order t**2)'''
	k = np.arange(-64,64,1.)
	T = tracek(k,aee,aii,see,sii,1,alpha)
	D = detk(k,aee,aei*aie,aii,see,sii,sii,1,alpha)
	Gex,Gix = [],[]
	for t in times:
		Gek = 1+ (aee*H(k,see) - 1)* t + 0.5* ((aee*H(k,see)-2-aii)*(1+aii) + (5*D*T**2 - 4*D**2 - T**4)/(4*D-T**2))* t**2
		Gex.append( np.real( np.fft.fft(np.fft.fftshift(Gek - Gek[0])) ) )
		
		Gik = aie*H(k,see)* t + 0.5*aie*H(k,see)* (aee*H(k,see)-2-aii)* t**2
		Gix.append( np.real( np.fft.fft(np.fft.fftshift(Gik - Gik[0])) ) )
	return np.array(Gex), np.array(Gix), np.fft.fftfreq(len(k),k[1]-k[0])



def greensfkt(aee,aei,aie,aii,see,sei,sii,alpha,times):
	''' inv FT of full greens fct '''
	# k = np.arange(-64,64,1.)
	k = np.arange(-64,64,1)
	l1 = eigval1(k,aee,aei*aie,aii,see,sei,sii,1,alpha)
	l2 = eigval2(k,aee,aei*aie,aii,see,sei,sii,1,alpha)
	f1 = ffct(k,aee,aie,aei,aii,see,sii,alpha)
	g1 = gfct(k,aee,aie,aei,aii,see,sii,alpha)
	f2 = ffct2(k,aee,aie,aei,aii,see,sii,alpha)
	g2 = gfct2(k,aee,aie,aei,aii,see,sii,alpha)
	
	det = detk(k,aee,aei*aie,aii,see,sei,sii,1,alpha)
	Jk = np.sqrt(2*np.pi)

	Gex, Gix = [],[]	
	lmaxE,lmaxI = [],[]
	for it in times:
		Gek = ( np.exp(l1 * it) * f1 + np.exp(l2 * it) * f2 ) #- Jk / det * (-1-aii*H(k,sii)+aie*H(k,sei))
		Gex.append( np.real( np.fft.fft(np.fft.fftshift(Gek)) ) )
		lmaxE.append( 1./abs(k[np.argmax(Gek)]) )
		
		Gik = ( np.exp(l1 * it) * g1 + np.exp(l2 * it) * g2 ) #- Jk / det * (-aei*H(k,see)-1+aee*H(k,see))
		Gix.append( np.real( np.fft.fft(np.fft.fftshift(Gik)) ) )
		lmaxI.append( 1./abs(k[np.argmax(Gik)]) )
	return np.array(Gex), np.array(Gix), lmaxE, lmaxI


def greensfkt2_Gaussapprox(aee,aei,aie,aii,see,sei,sii,alpha,times):
	''' Gauss approx of greens fct in frequency space;
	works nicely for larger times '''
	k = np.arange(-64,64,1.)
	Geks, Giks = [],[]
	l1 = eigval1(k,aee,aei*aie,aii,see,sei,sii,1,alpha)
	dk = k[1]-k[0]
	l11k = np.gradient(l1,dk)
	l12k = np.gradient(l11k,dk)
	f1 = ffct(k,aee,aie,aei,aii,see,sii,alpha)
	g1 = gfct(k,aee,aie,aei,aii,see,sii,alpha)
	
	f1k = np.gradient(np.log(f1),dk)
	g1k = np.gradient(np.log(g1),dk)
	f2k = np.gradient(f1k,dk)
	g2k = np.gradient(g1k,dk)
	for it in times:
		maxidf = np.nanargmax(np.real(l1*it + np.log(f1)))
		maxidg = np.nanargmax(np.real(l1*it + np.log(g1)))
		
		maxidf2 = len(k) - maxidf - 1
		maxidg2 = len(k) - maxidg - 1
		
		se_eff = (l12k * it + f2k)
		si_eff = (l12k * it + g2k)

		Gek = f1[maxidf] * np.exp(l1[maxidf] * it) * np.exp(1/2.*(k-k[maxidf])**2 * \
			se_eff[maxidf]) + f1[maxidf2] * np.exp(l1[maxidf2] * it) *\
			 np.exp(1/2.*(k-k[maxidf2])**2 * se_eff[maxidf2])
		Geks.append( np.fft.fftshift(np.real( np.fft.fft(np.fft.fftshift(Gek)) )) )
		
		Gik = g1[maxidg] * np.exp(l1[maxidg] * it) * np.exp(1/2.*(k-k[maxidg])**2 *\
		 si_eff[maxidg]) + g1[maxidg2] * np.exp(l1[maxidg2] * it) *\
		  np.exp(1/2.*(k-k[maxidg2])**2 * si_eff[maxidg2])
		Giks.append( np.fft.fftshift(np.real( np.fft.fft(np.fft.fftshift(Gik)) )) )

	Geks = np.array(Geks).reshape(times.shape[0],k.shape[0])
	Giks = np.array(Giks).reshape(times.shape[0],k.shape[0])

	return Geks, Giks


def amplitude_ratio(k,aee,aei,aie,aii,see,sii,alpha,t):
	aii_s = aii*(1-alpha)
	aii_a = aii*alpha
	c1k = c1(k,aee,aie,aii,see,sii,alpha)
	c2k = c2(k,aee,aie,aei,aii,see,sii,alpha)
	
	fk = ( c1k + c2k )
	gk = 1.
	dk = k[1]-k[0]
	
	f1k = np.gradient(fk,dk)
	f2k = np.gradient(f1k,dk)
	
	lk = eigval1(k,aee,aei*aie,aii,see,sii,sii,tau=1,alpha=alpha)
	l2k = np.gradient(np.gradient(lk,dk),dk)
	
	fidx = np.nanargmax(np.real(lk*t + np.log(fk)))
	gidx = np.nanargmax(np.real(lk*t))
	print('gidx',fidx,gidx,l2k[fidx],l2k[gidx])
	#fidx = 1510
	#gidx = 1510
	
	se_eff = np.sqrt( np.real( -l2k[fidx]*t - (f2k[fidx]*fk[fidx] - f1k[fidx]**2)/fk[fidx]**2 ) )
	si_eff = np.sqrt( np.real( -l2k[gidx]*t ) )
	
	AE = np.real( fk[fidx] * np.exp(lk[fidx] * t) )#/ se_eff )
	AI = np.real( gk * np.exp(lk[gidx] * t) )#/ si_eff )
	print('AE,Ai,t',AE,AI,t, AE/AI, np.exp((lk[fidx]-lk[gidx])*t), se_eff/si_eff)
	
	ratio = AE / AI
	return ratio


### Plotting style
def sequential_cmap(vmin,vmax):
	import matplotlib

	blue_hue = matplotlib.pyplot.get_cmap('winter_r')
	cNorm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	blue_s = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=blue_hue)
	
	bone_hue = matplotlib.pyplot.get_cmap('bone_r')
	cNorm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	bone_s = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=bone_hue)
	
	autm_hue = matplotlib.pyplot.get_cmap('autumn_r')
	cNorm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	autm_s = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=autm_hue)
	return blue_s, bone_s, autm_s


