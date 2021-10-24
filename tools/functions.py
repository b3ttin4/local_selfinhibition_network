import numpy as np
from scipy.stats import mode as modus


## Fit functions
def gauss(x,sigma,ampl):
	return ampl*np.exp(-(x-nE/2.)**2/2./sigma**2)

def cosine(x,k,sigma,A):
	return A*np.cos(x*k)*np.exp(-(x)**2/2./sigma**2)
	
def linfct(x,a,tau):
	return a + x*tau

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


def greensfkt(aee,aei,aie,aii,see,sei,sii,alpha,times,k=None):
	''' inv FT of full greens fct '''
	if k is None:
		k = np.arange(-64,64,1)
	# k = np.arange(-128,128,1)
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
		lmaxE.append( 1./abs(k[np.argmax(np.real(Gek))]) )
		
		Gik = ( np.exp(l1 * it) * g1 + np.exp(l2 * it) * g2 ) #- Jk / det * (-aei*H(k,see)-1+aee*H(k,see))
		Gix.append( np.real( np.fft.fft(np.fft.fftshift(Gik)) ) )
		lmaxI.append( 1./abs(k[np.argmax(np.real(Gik))]) )
	return np.array(Gex), np.array(Gix), np.array(lmaxE)*2*np.pi, np.array(lmaxI)*2*np.pi


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
	''' return ratio in amplitude of E units vs I units of linearized dynamics 
		for specific connectivity setting '''
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
	
	se_eff = np.sqrt( np.real( -l2k[fidx]*t - (f2k[fidx]*fk[fidx] - f1k[fidx]**2)/fk[fidx]**2 ) )
	si_eff = np.sqrt( np.real( -l2k[gidx]*t ) )
	
	AE = np.real( fk[fidx] * np.exp(lk[fidx] * t) )#/ se_eff )
	AI = np.real( gk * np.exp(lk[gidx] * t) )#/ si_eff )
	print('AE,Ai,t',AE,AI,t, AE/AI, se_eff/si_eff)
	
	ratio = AE / AI
	return ratio


def ratio_uei_approx(k,aee,aeiaie,aie,aii,aii_a,se,si):
	ratio = ( aee*H(k,se)+aii*H(k,si)+aii_a + np.sqrt((aee*H(k,se)+aii_a+aii*H(k,si))**2 - 4*aeiaie*H(k,se)*H(k,si)) ) /2./aie/H(k,se)
	#ratio = 1./2./aie * (aee + aii/H(k,se) + np.sqrt((aee+aii/H(k,se))**2-4*aeiaie*H(k,np.sqrt(se**2-si**2))) )
	return ratio

def ratio_coeff(s, r, aee, alpha, delta, mode):
	'''
	return ratio in amplitude of E units vs I units
	in phase space spanned by s, r and aee
	'''
	init_k = np.arange(0.01,50.,0.01)
	lk = np.zeros_like(init_k)
	neg_baseline = 0
	# ratios = np.empty((len(r),len(s),len(aee),2))*np.nan
	ratios = np.empty((len(r),len(s),len(aee),len(alpha),2))*np.nan
	for i,iss in enumerate(s):
		for j,jr in enumerate(r):
			ksei = np.sqrt(jr)
			for k,kaee in enumerate(aee):
				for l,lalph in enumerate(alpha):
				
					if kaee>(iss*(1+jr)):
						continue
					if jr<(1./iss):
						continue
					
					## find root in dlambda/dk (but quite hard)
					#lk[:] = 0
					#for l,linit in enumerate(init_k):
						#lk[l] = round(fsolve(cond_kmax, linit, args=(iss, jr, kaee, mode))[0], 1)
					#lk[lk<0] *= -1
					
					## PARAMETRIZATION
					kaii = (1-lalph)*(kaee - delta)
					kaii_a = lalph*(kaee - delta)
					kaeiaie = iss*(1+kaii+kaii_a)
					ksei = np.sqrt(jr)
					
					He = np.exp(-init_k**2/2.)
					Hi = np.exp(-init_k**2*jr/2.)
					#cond_kmax(init_k, iss, jr, kaee, mode)
					
					### all conditions satisfied, but lower amplitude ratios
					delta_tilde = 0.5
					delta2 = 1 - delta - delta_tilde
					kaei = kaee + delta2
					kaie = kaeiaie/kaei
					
					## yields high ampiltude ratio, but fixed point eqs not satisfied
					#kaie = kaee - 1 + 0.1#np.sqrt(kaeiaie)#
					#kaei = kaeiaie/kaie
					
					
					## calculate lambda as fct of k, find numerically maximum
					ew = eigval1(init_k,kaee,kaeiaie,kaee-delta,1,ksei,ksei,1,lalph)
					lk[:] = init_k[np.argmax(ew)]
					
					mod = modus(lk)
					if mod[1][0]>1:
						#print(jr,iss,kaee, (1 + kaie - kaee), (1 + kaii + kaii_a - kaei))
						#ratios[j,i,k,l] = ratio_uei(mod[0][0],kaee,kaeiaie,kaie,kaii,kaii_a,1.,ksei)
						ratios[j,i,k,l,0] = ratio_uei_approx(mod[0][0],kaee,kaeiaie,kaie,kaii,kaii_a,1.,ksei)
						ratios[j,i,k,l,1] = (1 + kaie - kaee)/(1 + kaii + kaii_a - kaei)
						if ratios[j,i,k,l,1]<0:
							neg_baseline += 1
						
	print('neg_baseline',neg_baseline,len(s)*len(r)*len(aee),1.*neg_baseline/len(s)/len(r)/len(aee))
	return ratios

## helper functions for plotting phase diagrams
##======== combine params: s = aei*aie/(1+aii) =====================
def det0_3(s, mode, *args):
	''' condition det(k=0)>0'''
	if mode=='3':
		aee = s[None,:] + 1
	elif mode=='aei':
		aee = 1/(1-s[None,:])
	elif mode=='exp':
		aee = s[None,:] + 1
	elif mode=='exc':
		aee = -s[None,:] + 1
	elif mode=='aii':
		aii_fix = args[0]
		aee = s[None,:]/(1.+aii_fix) + 1
	return aee

def min_existence_3(s, r, mode, *args):
	''' positive logarithm in kmin '''
	if mode=='3':
		aee = s[None,:]*(1+r[:,None])
	elif mode=='aei':
		aee = s[None,:]*(1+r[:,None])
	elif mode=='exp':
		aee = s[None,:]/(1-r[:,None])
	elif mode=='exc':
		aee = -s[None,:]*(1+r[:,None])
	elif mode=='aii':
		aii_fix = args[0]
		aee = s[None,:]*(1+r[:,None])/(1.+aii_fix)
	return aee

def neg_min_3(s, r, mode, *args):
	'''det(kmin)<0; aee > (1+r)s**(1/(1+r))r**(-r/(1+r))'''
	if mode=='3':
		aee = (1+r[:,None])/r[:,None]**(r[:,None]/(1+r[:,None])) * s[None,:]**(1/(1+r[:,None]))
	elif mode=='aei':
		aee = (1+r[:,None])**((1+r[:,None])/r[:,None])/r[:,None]*s[None,:]**(1/r[:,None])
	elif mode=='exp':
		aee = 2*np.sqrt(s[None,:]/r[:,None]) - 1./r[:,None] + 1
	elif mode=='exc':
		aee = ((1/r[:,None]+1)*(abs(s[None,:])*(1+r[:,None]))**(1/r[:,None]))**(r[:,None]/(1+r[:,None]))
	elif mode=='aii':
		aii_fix = args[0]
		aee = (1+r[:,None])/r[:,None]**(r[:,None]/(1+r[:,None])) * s[None,:]**(1/(1+r[:,None])) / (1.+aii_fix)**(1./(1+r[:,None]))
	return aee


## helper functions for alpha<1
def det0(s, r, mode):
	''' condition det(k=0)>0'''
	aee = (s[None,:,None] + 1.)*np.ones_like(r)[:,None,None]
	return aee

def neg_min(k, s, r, aee, alpha, delta, mode):
	'''det(kmin)<0'''
	aii = (1-alpha[None,None,None,:])*(aee[None,None,:,None] - delta)	#0.7#1.4#
	aii_a = alpha[None,None,None,:]*(aee[None,None,:,None] - delta)
	aii_a[aii_a<0] = 0
	aii[aii<0] = 0
	aeiaie = s[None,:,None,None]*(1+aii+aii_a)
	se = 1
	si = np.sqrt(r[:,None,None,None])
	neg_min_val = detk(k,aee[None,None,:,None],aeiaie,(aee[None,None,:,None] - delta),\
						se,si,si,tau=1,alpha=alpha)
	return neg_min_val

def neg_min_alpha(k, s, alpha, aee, delta, mode):
	'''det(kmax)<0'''
	aii = (1-alpha[:,None,None])*(aee[None,None,:] - delta)	#0.7#1.4#
	aii_a = alpha[:,None,None]*(aee[None,None,:] - delta)
	aii_a[aii_a<0] = 0
	aeiaie = s[None,:,None]*(1+aii+aii_a)
	se = 1.
	r = 0.8
	si = np.sqrt(r)
	
	neg_min_val = detk(k,aee[None,None,:],aeiaie,aee[None,None,:] - delta,se,si,si,tau=1.,\
						alpha=alpha[:,None,None])
	#print('neg_min_val',neg_min_val.size,neg_min_val)
	#neg_min_bool = neg_min_val<0
	#print('neg_min_bool',neg_min_bool.size,np.sum(neg_min_bool))
	return neg_min_val


def positive_lambda(s, r, aee, alpha, delta, mode):
	''' returns frequency k where eigenvalue is maximal'''
	frq = np.arange(0,5,0.005)
	pos_lambda = np.empty((len(r),len(s),len(aee),len(alpha),2))*np.nan
	for i,iss in enumerate(s):
		for j,jr in enumerate(r):
			for k,kaee in enumerate(aee):
				for l,lalph in enumerate(alpha):
					kaii = (1-lalph)*(kaee - delta)
					kaii_a = lalph*(kaee - delta)
					kaeiaie = iss*(1+kaii+kaii_a)
					ksi = np.sqrt(jr)
					
					eigvals = np.real(eigval1(frq,kaee,kaeiaie,kaee-delta,1,ksi,ksi,tau=1,alpha=lalph))
					if (eigvals>0).any():
						max_eigval = np.nanmax(eigvals)
						if max_eigval>eigvals[0]:
							pos_lambda[j,i,k,l,0] = max_eigval
							pos_lambda[j,i,k,l,1] = frq[np.nanargmax(eigvals)]
	return pos_lambda

def min_alpha_pf(s, r, aee, alpha, delta, mode):
	''' returns frequency k where eigenvalue is maximal'''
	frq = np.arange(0,5,0.005)
	pos_lambda = np.empty((len(r),len(s),len(aee),2))*np.nan
	lambda_alpha = np.empty((len(alpha)))*np.nan
	for i,iss in enumerate(s):
		for j,jr in enumerate(r):
			for k,kaee in enumerate(aee):
				lambda_alpha[:] = np.nan
				for l,lalph in enumerate(alpha):
					kaii = (1-lalph)*(kaee - delta)
					kaii_a = lalph*(kaee - delta)
					kaeiaie = iss*(1+kaii+kaii_a)
					ksi = np.sqrt(jr)
					
					eigvals = np.real(eigval1(frq,kaee,kaeiaie,kaee-delta,1.,ksi,ksi,tau=1,alpha=lalph))
					max_eigval = np.nanmax(eigvals)
					if max_eigval>eigvals[0]:
						lambda_alpha[l] = max_eigval
				if (lambda_alpha>0).any():
					min_alpha = np.where(lambda_alpha>0)[0][0]
					pos_lambda[j,i,k,0] = lambda_alpha[min_alpha]
					pos_lambda[j,i,k,1] = alpha[min_alpha]
	return pos_lambda

def positive_lambda_alpha(s, alpha, aee, delta, mode):
	''' returns frequency k where eigenvalue is maximal'''
	frq = np.arange(0,10,0.005)
	pos_lambda = np.empty((len(alpha),len(s),len(aee),2))*np.nan
	r = 0.8
	si = np.sqrt(r)
	for i,iss in enumerate(s):
		for j,jalpha in enumerate(alpha):
			for k,kaee in enumerate(aee):
				
				kaii = (1-jalpha)*(kaee - delta)#3.4#
				kaii_a = jalpha*(kaee - delta)
				kaeiaie = iss*(1+kaii+kaii_a)
				kaie = kaee - 1 + 0.1
				
				eigvals = np.real(eigval1(frq,kaee,kaeiaie,kaee-delta,1.,si,si,tau=1.,alpha=jalpha))
				if (eigvals>0).any():
					max_eigval = np.nanmax(eigvals)
					if max_eigval>eigvals[0]:
						pos_lambda[j,i,k,0] = max_eigval
						pos_lambda[j,i,k,1] = frq[np.nanargmax(eigvals)]
	return pos_lambda

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


if __name__=="__main__":
	import matplotlib.pyplot as plt

	aii = 3.0
	conn_strength = np.array([1.95+aii,1.4+aii,aii+0.9,aii])
	conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/100.
	see,sie,sei,sii = conn_width_mean
	aee,aie,aei,aii = conn_strength
	alpha = 1.
	times = np.array([2.5])
	Gex,Gix,lmaxE,lmaxI = greensfkt(aee,aei,aie,aii,see,sei,sii,alpha,times)
	print("Gex",Gex.shape,lmaxI[0]-lmaxE[0])

	k = np.arange(0,64,.1)
	print("deltasigma",deltasigma(k,aee,aei,aie,aii,see,sii,alpha,times[0]))

	# fig = plt.figure()
	# ax = fig.add_subplot(121)
	# ax.plot(Gek,'-k')
	# ax.plot(Gik,'-m')
	# # ax = fig.add_subplot(122)
	# # ax.plot(Gik,'-k')
	# plt.show()
	exit()

	npts = 5
	s = np.linspace(0.01,40,npts+1)
	r = np.linspace(0.01,2.,npts)
	aee = np.linspace(1.8,40.2,25)
	alpha = 1.
	delta = 1.95
	frq = np.arange(0,5,0.1)

	alpha = np.array([1])
	fig = plt.figure(figsize=(npts*6,npts*5))
	for i in range(npts):
		for j in range(npts):
			ax = fig.add_subplot(npts,npts,i*npts+j+1)
			ksi = np.sqrt(r[j])
			for kaee in aee:
				kaii = (1-alpha)*(kaee - delta)
				kaii_a = alpha*(kaee - delta)
				kaeiaie = s[i]*(1+kaii+kaii_a)
				eigvals = np.real(eigval1(frq,kaee,kaeiaie,kaee-delta,1,ksi,ksi,tau=1,alpha=alpha))
				ax.plot(eigvals,'-')

	plt.show()