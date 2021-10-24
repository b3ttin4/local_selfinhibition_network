import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
# import matplotlib.animation as animation
from scipy.optimize import curve_fit
# from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp2d,interp1d
import scipy.integrate as integrate

from tools import functions as fct
from . import image_dir

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'



def plot_activity_patterns(figure_no,activity_t,lin_operator,config_dict,network_params):
	"""
	plot E and I final activity patterns
	"""
	see,sie,sei,sii = config_dict["conn_width_mean"]
	aee,aie,aei,aii = config_dict["conn_strength"]
	nE, nI = config_dict["system_size"],config_dict["system_size"]
	u = activity_t[...,-1]

	fig_num_sim = plt.figure(figsize=(6*3,5*2))
	fig_num_sim.suptitle('nE={:.0f}, nI={:.0f}, tau={} \n\
						aee={:.1f},aie={:.1f},aei={:.1f},aii={:.1f} \n\
						see={:.3f}, sie={:.3f}, sei={:.3f}, sii={:.3f}'.format(\
						nE,nI,network_params["tau"],aee,aie,aei,aii,see,sie,sei,sii))


	if config_dict["dim"]==1:
		## Steady state solutions
		ue0 = network_params["inpE"]*(aei-aii-1)/((aii+1)*(aee-1)-aei*aie)
		ui0 = network_params["inpE"]*(aee-aie-1)/((aii+1)*(aee-1)-aei*aie)
		space = np.linspace(-1./2,1./2,nE,endpoint=False)/see
		k = np.arange(0,120,1.)
		ew = fct.eigval1(k,aee,aei*aie,aii,see,sei,sii,network_params["tau"],\
							config_dict["alpha"])
		kmax = k[np.argmax(np.real(ew))]

		ax_E = fig_num_sim.add_subplot(231)
		ax_I = fig_num_sim.add_subplot(232)
		ax_ampl = fig_num_sim.add_subplot(233)
		for ax in [ax_E,ax_I]:
			ax.set_ylabel('Activity')
			ax.set_xlabel(r'Distance ($\sigma_E$)')
		ax_E.plot([space[0],space[-1]],[ue0]*2,'--r',label=r'$\bar{u}_E$')
		ax_I.plot([space[0],space[-1]],[ui0]*2,'--b',label=r'$\bar{u}_I$')
		
		ax_E.plot(space,u[:nE],'-r',label=r"$u_E$")
		# ax_E.plot(space,u[nE:]-0.09245+0.0016,'-b',label=r"$u_I$")
		# ax_E.plot(space,u[nE:]+7.,'-b',label=r"$u_I$")
		ax_I.plot(space,(u[nE:]),'-b',label=r"$u_I$")
		# ax_E.set_ylim(0.00045,0.0012)
		# ax_I.set_ylim(0.00045,0.0012)
		
		## estimate amplitude ratio
		respE = u[:nE] - np.nanmean(u[:nE])
		respI = u[nE:] - np.nanmean(u[nE:])
		pexpE,_ = curve_fit(fct.cosine,space,respE,p0=[kmax*see,3,np.nanmax(respE)])	#k,phi,A
		pexpI,_ = curve_fit(fct.cosine,space,respI,p0=[kmax*see,3,np.nanmax(respI)])
		AE_fit = np.abs(pexpE[2])
		AI_fit = np.abs(pexpI[2])
		AE_k = np.nanmax(np.abs(np.fft.fft(respE)))
		AI_k = np.nanmax(np.abs(np.fft.fft(respI)))
		AE_mm = ( np.nanmax(respE) - np.nanmin(respE) ) * 0.5
		AI_mm = ( np.nanmax(respI) - np.nanmin(respI) ) * 0.5

		ampl_ratio_sim = AE_fit/AI_fit
		ampl_ratio_k = AE_k/AI_k
		ampl_ratio_mm = AE_mm/AI_mm
		ampl_ratio_theo = fct.amplitude_ratio(k,aee,aei,aie,aii,see,sii,config_dict["alpha"],\
											(network_params["timesteps"]-1)*network_params["dt"])
		ratio_labels = ["Simul","Freq based","DiffMinMax","Theory"]
		for i,iampl_ratio in enumerate([ampl_ratio_sim,ampl_ratio_k,ampl_ratio_mm,\
										ampl_ratio_theo]):
			ax_ampl.plot([i],[iampl_ratio],"o",label=ratio_labels[i])
		ax_ampl.set_xlim(-0.5,3.5)
		ax_ampl.set_ylim(0,3.0)
		ax_ampl.set_xticks(np.arange(len(ratio_labels)))
		ax_ampl.set_xticklabels(ratio_labels,rotation=45)
		
		## Spectrum of solution
		ax_fftE = fig_num_sim.add_subplot(234)
		ax_fftI = fig_num_sim.add_subplot(235)
		for ax in [ax_fftE,ax_fftI]:
			ax.set_ylabel('Spectrum')
			ax.set_xlabel('Frequency')
			ax.set_xlim(-60,60)
		freqE = np.fft.fftfreq(nE)*nE
		fftuE = np.fft.fft(u[:nE]-np.nanmean(u[:nE]))
		fftuI = np.fft.fft(u[nE:]-np.nanmean(u[nE:]))
		ax_fftE.plot(freqE*2*np.pi,np.abs(fftuE)*nE*np.sqrt(2*np.pi),'-or')
		ax_fftI.plot(freqE*2*np.pi,np.abs(fftuI)*nE*np.sqrt(2*np.pi),'-ob')

		## Eigenvalues
		ax_ew = fig_num_sim.add_subplot(236)
		ax_ew.plot(k[1:],np.real(ew[1:]),'-k')
		ax_ew.plot(k,k*0,'--',c='gray')
		ax_ew.set_ylabel('Eigenvalue (k)')
		ax_ew.set_xlabel('Frequency')
		for ax in [ax_E,ax_I,ax_fftE,ax_fftI,ax_ew]:
			ax.legend(loc='best',fontsize=10)


	else:
		nE2 = nE**2
		M1 = lin_operator
		ax_E = fig_num_sim.add_subplot(231)
		ax_I = fig_num_sim.add_subplot(232)
		ax_M = fig_num_sim.add_subplot(233)
		for ax in [ax_E,ax_I]:
			ax.set_ylabel('Distance')#($\sigma_E$)
			ax.set_xlabel('Distance')# ($\sigma_E$)
		ax_M.set_title('Recurrent connectivity')
		
		uE = u[:nE2].reshape(nE,nE)
		im=ax_E.imshow(uE,interpolation='nearest',cmap='binary_r',vmin=0)#,vmax=0.12)
		plt.colorbar(im,ax=ax_E)
		ax_E.set_title("Excitatory activity u_E")
		ax_E.set_xticks(np.arange(0,1,5*see)*nE)
		ax_E.set_yticks(np.arange(0,1,5*see)*nE)
		uI = u[nE2:].reshape(nE,nE)
		im=ax_I.imshow(uI,interpolation='nearest',cmap='binary_r',vmin=0)#,vmax=0.12)
		plt.colorbar(im,ax=ax_I)
		ax_I.set_xticks(np.arange(0,1,5*see)*nE)
		ax_I.set_yticks(np.arange(0,1,5*see)*nE)
		ax_I.set_title("Inhibitory activity u_I")
		
		# im=ax_M.imshow(M1,interpolation="nearest",cmap="binary")
		# plt.colorbar(im,ax=ax_M)
		ax_M.plot(np.nanmean(activity_t,axis=0),'-k',label="Avg")
		ax_M.plot(activity_t[0,:],'-m',label="(0,0)")
		ax_M.plot(activity_t[nE//2,:],'-g',label="(NE/2,0)")
		ax_M.plot(activity_t[nE//2*(nE+1),:],'-c',label="(NE/2,NE/2)")
		ax_M.legend(loc="best")

		## Spectrum of solution
		ax_fftE = fig_num_sim.add_subplot(234)
		ax_fftI = fig_num_sim.add_subplot(235)
		for ax in [ax_fftE,ax_fftI]:
			ax.set_ylabel('Frequency')
			ax.set_xlabel('Frequency')
		fftuE = np.fft.fftshift(np.fft.fft2(uE - np.nanmean(uE)))
		fftuI = np.fft.fftshift(np.fft.fft2(uI - np.nanmean(uI)))
		im=ax_fftE.imshow(np.abs(fftuE)*nE*np.sqrt(2*np.pi),cmap="binary")
		plt.colorbar(im,ax=ax_fftE)
		ax_fftE.set_title('Spectrum(u_E)')
		im=ax_fftI.imshow(np.abs(fftuI)*nE*np.sqrt(2*np.pi),cmap="binary")
		plt.colorbar(im,ax=ax_fftI)
		ax_fftI.set_title('Spectrum(u_I)')
		
		## Eigenvalues
		ax_ew = fig_num_sim.add_subplot(236)
		k = np.arange(0,120,1)
		real_ew = np.real(fct.eigval1(k,aee,aei*aie,aii,see,sei,sii,network_params["tau"],\
										config_dict["alpha"]))

		ax_ew.plot(k[1:],np.real(real_ew[1:]),'-k')
		ax_ew.plot(k,k*0,'--',c='gray')
		ax_ew.set_ylabel('Eigenvalue (k)')
		ax_ew.set_xlabel('Frequency')
		# ax_ew.set_ylim(-12,2.5)

	
	plt.savefig(image_dir + "Fig{}_activity.pdf".format(figure_no))
	plt.close(fig_num_sim)


def plot_greensfunction(figure_no,activity_t,lin_operator,config_dict,network_params):
	"""
	plot E and I response to pulse input and theoretically derived greens function plus 
	approximations
	"""
	see,sie,sei,sii = config_dict["conn_width_mean"]
	aee,aie,aei,aii = config_dict["conn_strength"]
	nE, nI = config_dict["system_size"],config_dict["system_size"]
	timesteps = network_params["timesteps"]
	M1 = lin_operator
	U = activity_t


	fig_greensfkt = plt.figure(figsize=(6*3,5*2))
	fig_greensfkt.suptitle(r'nE={:.0f}, nI={:.0f}, tau={}, alpha={}\n\
		aee={:.1f},aie={:.1f},aei={:.1f},aii={:.1f} \n\
		see={:.3f}, sie={:.3f}, sei={:.3f}, sii={:.3f}'.format(\
		nE,nI,network_params["tau"],config_dict["alpha"],aee,aie,aei,aii,see,sie,sei,sii))

	blue_s, bone_s, autm_s = fct.sequential_cmap(0,100)

	ax_E = fig_greensfkt.add_subplot(231)
	ax_I = fig_greensfkt.add_subplot(232)
	
	ax_EI = fig_greensfkt.add_subplot(233)
	ax_DL = fig_greensfkt.add_subplot(234)
	ax_inp = fig_greensfkt.add_subplot(235)

	x = np.linspace(-1./2,1./2,nE,endpoint=False)/see
	k = np.arange(0,64,0.01)
	ew = fct.eigval1(k,aee,aei*aie,aii,see,sei,sii,network_params["tau"],\
						config_dict["alpha"])
	kmax = k[np.argmax(np.real(ew))]
	max_lambda = ew[np.argmax(np.real(ew))]

	
	lmax = 2*np.pi/kmax
	all_inpE, all_inpI = [],[]
	visbin_inp = 1
	# for it in range(0,timesteps,visbin_inp):
	for it in [50,]:
		## Convolution response with connectivity
		inpEE = np.dot(M1[:nE,:nE],U[:nE,it]*nE - np.nanmean(U[:nE,0])*nE) *aee
		inpEI = np.dot(M1[nE:,:nE],U[nE:,it]*nE - np.nanmean(U[nE:,0])*nE) *aei
		all_inpE.append(inpEE)
		all_inpI.append(inpEI)
		
	# ax_inp.plot(x,all_inpE[1],'--r',label=r"$I_{EE}, t=0$")
	# ax_inp.plot(x,all_inpI[1],'--b',label=r"$I_{EI}, t=0$")
	# ax_inp.plot(x,all_inpE[1]-all_inpI[1],'--k',label=r"$I_{EE}-I_{EI}, t=0$")
	
	ax_inp.plot(x,inpEE,'-r',label=r"$I_{EE}$"+",t={}".format(it))
	ax_inp.plot(x,inpEI,'-b',label=r"$I_{EI}$"+",t={}".format(it))
	ax_inp.plot(x,inpEE-inpEI,'-k',label=r"$I_{EE}-I_{EI}$"+",t={}".format(it))
	ax_inp.set_xlabel(r"Space ($\sigma_{EE}$")
	ax_inp.set_ylabel("Input to E")
	ax_inp.legend(loc="best",fontsize=10)

	diff,difffull,diff_fit,diff_fitg,seeff,sieff,Aeff = [],[],[],[],[],[],[]
	seeff_th, sieff_th = [],[]
	visbin = 10
	start = 100
	print("timesteps",timesteps,U.shape,lmax)
	for it in range(start,timesteps,visbin):
		respE = -(U[:nE,it]-np.nanmean(U[:nE,it]))
		respI = -(U[nE:,it]-np.nanmean(U[nE:,it]))

		# respE = -(U[:nE,it]-U[0,it])
		# respI = -(U[nE:,it]-U[nE,it])

		try:
			## wavelength via cosine fit
			pexpE,_ = curve_fit(fct.cosine,x,respE,p0=[0.8,3,-0.5])	#k,sigma,A
			pexpI,_ = curve_fit(fct.cosine,x,respI,p0=[0.8,3,-0.3])
			
		except Exception as e:
			print(e)
			pexpE = np.array([np.nan,np.nan,np.nan])
			pexpI = np.array([np.nan,np.nan,np.nan])
		
		deltaE_fit = see/np.abs(pexpE[0])*2*np.pi
		deltaI_fit = see/np.abs(pexpI[0])*2*np.pi

		_,_,lmaxE,lmaxI = fct.greensfkt(aee,aie,aei,aii,see,sii,sii,config_dict["alpha"],\
										np.array([it*network_params["dt"]]),k=k)
		diff_fitg.append((lmaxI[0]-lmaxE[0])/lmax)
		difffull.append( fct.deltasigma(k,aee,aie,aei,aii,see,sii,config_dict["alpha"],\
										it*network_params["dt"])/lmax )
		diff_fit.append((deltaI_fit-deltaE_fit)/lmax)

	print("diff_fit",diff_fit)
	print("diff_fitg",diff_fitg)
	time = np.linspace(start,timesteps,len(diff_fit))*network_params["dt"]
	ax_DL.plot(time,diff_fit,'-c',label='Fit to simulation')
	ax_DL.plot(time,np.array(diff_fitg),'--g',label='Gaussian approximation')
	ax_DL.legend(loc='best',fontsize=10)
	# ax_DL.set_ylim(0,0.05)
	ax_DL.set_xlabel('Time (tau)')
	ax_DL.set_ylabel('Diff variance/sigma_E^2')
	ax_DL.set_ylim(0,0.25)
	ax_DL.set_xlim(1,5)


	visbin = 10
	start = 80
	for it in [50,99]:##assuming dt=0.01
		#ax.plot(VV[0,:nE,it]-VV[0,0,it],'-',c=scalarMap.to_rgba(it))
		#ax.plot(VV[0,nE:,it]-VV[0,nE,it],'-',c=bone_hue.to_rgba(it))
		
		respE = -(U[:nE,it]-np.nanmean(U[:nE,it]))
		respI = -(U[nE:,it]-np.nanmean(U[nE:,it]))

		respE = -(U[:nE,it]-U[0,it])
		respI = -(U[nE:,it]-U[nE,it])
		
		try:
			## wavelength via cosine fit
			pexpE,_ = curve_fit(fct.cosine,x,respE,p0=[0.8,3,-0.5])	#k,sigma,A
			pexpI,_ = curve_fit(fct.cosine,x,respI,p0=[0.8,3,-0.3])
			
		except Exception as e:
			print(e)
			pexpE = np.array([np.nan,np.nan,np.nan])
			pexpI = np.array([np.nan,np.nan,np.nan])
		
		deltaE_fit = see/np.abs(pexpE[0])*2*np.pi
		deltaI_fit = see/np.abs(pexpI[0])*2*np.pi				
	
		## simulation
		ax_E.plot(x,-respE*nE,'-',c=autm_s.to_rgba(it),label="E sim, t={}".format(it))
		ax_I.plot(x,-respI*nE,'-',c=blue_s.to_rgba(it),label="I sim, t={}".format(it))

		## cosine approx
		if False:
			ax_E.plot(x,-fct.cosine(x,*pexpE)*nE,'+',c=autm_s.to_rgba(it-80))
			ax_I.plot(x,-fct.cosine(x,*pexpI)*nE,'+',c=blue_s.to_rgba(it-80))				
		## Gaussian approximation of analytic expression
		if False:
			Gex_app,Gix_app = fct.greensfkt2_Gaussapprox(aee,aei,aie,aii,see,\
								sii,sii,config_dict["alpha"],np.array([it*network_params["dt"]]))
			Gex_app=Gex_app.flatten()
			Gix_app=Gix_app.flatten()
			xbins = np.linspace(-0.5,0.5,len(Gex_app),endpoint=False)/see*2*np.pi
			ax_E.plot(xbins,Gex_app/5000.,'+',c=autm_s.to_rgba(it))
			ax_I.plot(xbins,Gix_app/5000.,'+',c=blue_s.to_rgba(it))
		
		### full analytic expression
		if True:
			Gex,Gix,lmaxE,lmaxI = fct.greensfkt(aee,aei,aie,aii,see,sii,sii,\
									config_dict["alpha"],np.array([it*network_params["dt"]]))
			Gex=Gex.flatten()
			Gix=Gix.flatten()
			xbins = np.linspace(-0.5,0.5,len(Gex),endpoint=False)/see*2*np.pi
			ax_E.plot(xbins,np.fft.fftshift(Gex)/5000.,'+',c=autm_s.to_rgba(it),\
						label="E theory, t={}".format(it))
			ax_I.plot(xbins,np.fft.fftshift(Gix)/5000.,'+',c=blue_s.to_rgba(it),\
						label="I theory, t={}".format(it))

	
	axlim = 10.
	for ax in [ax_E,ax_I]:
		ax.legend(loc="best")
		ax.set_xlim(-axlim,axlim)
		ax.set_xlabel('Distance [see]')
	ax_E.set_ylabel('Exc response')
	ax_I.set_ylabel('Inh response')

	## activity at last timestep
	t = 50
	ax_EI.plot(x,(U[:nE,t]-U[0,t])*nE,'-r',label="Exc., t={}".format(t))
	ax_EI.plot(x,(U[nE:,t]-U[nE,t])*nE,'-b',label="Inh., t={}".format(t))
	ax_EI.legend(loc='best',fontsize=10)
	ax_EI.set_xlabel('Distance [see]')
	ax_EI.set_ylabel('Resp at last time point')
	ax_EI.set_xlim(-7,7)
	ax_EI.set_ylim(-0.01,0.04)

	plt.savefig(image_dir + "greensfkt_Fig{}.pdf".format(figure_no))
	plt.close(fig_greensfkt)




def plot_phase_diagram3d(figure_no,s2d,r2d,aee2d,Z01,Z2,Zr,config_dict):
	"""
	plot and save 3d phase diagram
	"""
	##generate figure
	fig = plt.figure()
	ax3d = fig.gca(projection='3d')
	ax3d.view_init(22,10)

	mode = config_dict["mode"]
	
	if config_dict["feature"] is None:
		ax3d.plot_surface(s2d,r2d,Z01[:,:,0],color='r',alpha=0.5,antialiased=True)#,rstride=2)
		ax3d.plot_surface(s2d,r2d,Z2,color='k',alpha=0.5,antialiased=True,rstride=5,zorder=2)
	
	elif config_dict["feature"] in ("ratio","ratio_norm"):
		if config_dict["feature"]=="ratio":
			vmin,vmax = 0.,1.5
		elif config_dict["feature"]=="ratio_norm":
			vmin,vmax = 0.,0.15
		elif config_dict["feature"]=="baseline":
			vmin,vmax = 0.,0.2
		for ii in range(Zr.shape[2]):
			ax3d.contourf(s2d[:,:,ii], r2d[:,:,ii], Zr[:,:,ii,0], 50, zdir='z',\
				 			offset=aee2d[0,0,ii], cmap='plasma',vmin=vmin,vmax=vmax)
	## SAVING
	ax3d.set_ylim(0,np.max(r2d))
	if mode in ("aei","exp","3"):
		ax3d.set_zlim(0,max(config_dict["aee"]))#+5
	# elif mode in ('3'):
	# 	ax3d.set_zlim(0,40)#(0,25)#
	elif mode in ('aii',):
		ax3d.set_zlim(0,2+config_dict["aii_fix"])
		ax3d.set_xlim(0,max(s))#5
	ax3d.set_zlabel(r'$a_{ee}$')
	ax3d.set_xlabel(r'$s=a_{ei}*a_{ie}/(1+a_{ii})$')
	ax3d.set_ylabel(r'$r=s_{ei}^2/s_{ee}^2$')
	pdf_name = image_dir + 'Fig{}_phase_diag3d.pdf'.format(figure_no)
	plt.savefig(pdf_name, format='pdf', dpi=200)
	plt.close()


def plot_phase_diagram2d(figure_no,s2d,r2d,Z01,Z2,Zr,minalpha,config_dict):
	"""
	plot and save 2d phase diagram (slices through 3d phase diagram at fixed
	values for aee, s and r)
	"""
	alpha = config_dict["alpha"]
	r = config_dict["r"]
	s = config_dict["s"]
	delta = config_dict["delta"]
	aii_fix = config_dict["aii_fix"]

	if not isinstance(alpha,np.ndarray):
		alpha = np.array([alpha])

	ncol,nrow = 4,1
	fig = plt.figure(figsize=(6*ncol,5*nrow))
	axes = []
	for i in range(ncol):
		axes.append(fig.add_subplot(nrow,ncol,1+i))

	if config_dict["feature"] is None:
		for i,ialpha in enumerate(alpha):
			ridx = np.argmin(np.abs(r-0.8))
			axes[0].set_title('r={:.2f}'.format(r[ridx]))
			axes[0].plot(s2d[ridx,:],Z01[ridx,:],'-',color='r')
			axes[0].plot(s2d[ridx,:],Z2[ridx,:,i],'-',color='k')
			axes[0].set_xlabel('s')
			axes[0].set_ylabel('aee')
			axes[0].axhline(y=2,xmin=0,xmax=max(s),ls='--',c='gray')
			axes[0].set_ylim(0,20)#45#6
			axes[0].set_xlim(0,20)
			
			sidx = np.argmin(np.abs(s-11.4))
			axes[1].set_title('s={:.1f}'.format(s[sidx]))
			axes[1].plot(r2d[:,sidx],Z01[:,sidx],'-',color='r')
			axes[1].plot(r2d[:,sidx],Z2[:,sidx,i],'-',color='k')
			axes[1].set_xlabel('r')
			axes[1].set_ylabel('aee')
			axes[1].set_ylim(0,15.)
			
			axes[2].set_title('a_ee={}'.format(delta+aii_fix))
			Z2[np.logical_not(np.isfinite(Z2[:,:,i])),i] = 0.0
			f01 = interp2d(s2d,r2d,Z01,kind='linear',copy=True)
			f2 = interp2d(s2d,r2d,Z2[:,:,i],kind='linear',copy=True)
			z01_intp = f01(s,r)
			z2_intp = f2(s,r)
			
			if aii_fix is not None:
				aee_value = delta+aii_fix
			else:
				aee_value = config_dict["aee"][0]
			axes[2].contour(z01_intp.T,[aee_value],colors=['r'])#1.95#15
			axes[2].contour(z2_intp.T,[aee_value],colors=['k'])#
			#axes[2].plot(r,1./r,'--',c='gray')
			axes[2].set_xlabel('r')
			axes[2].set_ylabel('s')
			axes[2].set_xticks(np.arange(len(r))[::10])
			axes[2].set_xticklabels(np.around(r[::10],2))
			axes[2].set_yticks(np.arange(len(s))[::5])
			axes[2].set_yticklabels(np.around(s[::5],2))

			axes[3].set_title('a_ee={}'.format(delta+aii_fix))
			axes[3].contour(z01_intp.T,[aee_value],colors=['r'])#1.95#15
			axes[3].contour(z2_intp.T,[aee_value],colors=['k'])			
			axes[3].set_xlabel('r')
			axes[3].set_ylabel(r'$a_{ei}*a_{ie}$')
			axes[3].set_xticks(np.arange(len(r))[::10])
			axes[3].set_xticklabels(np.around(r[::10],2))
			axes[3].set_yticks(np.arange(len(s))[::5])
			axes[3].set_yticklabels(np.around(s[::5]*(aii_fix+1),2))

			## RGB
			#Z_rgb = np.zeros(Z01.T.shape+(3,))
			#maxz = np.nanmax([np.concatenate([Z2,Z01])])
			#Z_rgb[:,:,0] = Z01.T/maxz
			#Z_rgb[:,:,2] = Z2.T/maxz
			#ax.imshow(Z_rgb[:,:6,:],interpolation='nearest',origin='lower',extent=[0,51,0,6],aspect=51//6)
			
			## DIFF
			#diff_matrix = (Z01-Z2)[:,:6].T
			#ax.imshow(np.clip(diff_matrix,0,np.nanmax(diff_matrix)),interpolation='nearest',\
			#cmap='binary',origin='lower',extent=[0,51,0,6],aspect=51//6)

	elif config_dict["feature"] in ("min_alpha","ratio","ratio_norm","baseline"):
		if config_dict["feature"]=="min_alpha":
			array2d  = minalpha[:,:,:,1]
			cmap = "viridis"
			vmin,vmax = 0,1
		elif config_dict["feature"] in ("ratio","ratio_norm","baseline"):
			array2d = Zr[:,:,:,0]
			cmap = "plasma"
			if config_dict["feature"]=="ratio":
				vmin,vmax = 0.3,0.8
			elif config_dict["feature"]=="ratio_norm":
				vmin,vmax = 0.,0.15
			elif config_dict["feature"]=="baseline":
				vmin,vmax = 0.,0.2

		aee_fixed = 11.4
		s_fixed = 11.4
		r_fixed = 0.8

		axes[0].set_title('r={:.1f}'.format(r[np.argmin(np.abs(r-r_fixed))]))
		im0=axes[0].imshow(array2d[np.argmin(np.abs(r-r_fixed)),:,:].T,interpolation='nearest',\
						cmap=cmap,vmin=vmin,vmax=vmax)
		plt.colorbar(im0,ax=axes[0])
		axes[0].set_xlabel('s')
		axes[0].set_xticks(np.arange(0,len(s),5))
		axes[0].set_xticklabels(np.around(s[::5],1))
		axes[0].set_ylabel('aee')
		axes[0].set_yticks(np.arange(0,len(config_dict["aee"]),3))
		axes[0].set_yticklabels(np.around(config_dict["aee"][::3],1))
		plt.gca().invert_yaxis()
		axes[0].set_ylim(0,config_dict["aee"][-1])
		
		axes[1].set_title('s={:.1f}'.format(s[np.argmin(np.abs(s-s_fixed))]))
		im1=axes[1].imshow(array2d[:,np.argmin(np.abs(s-s_fixed)),:].T,interpolation='nearest',\
						cmap=cmap,vmin=vmin,vmax=vmax)
		plt.colorbar(im1,ax=axes[1])
		axes[1].set_xlabel('r')
		axes[1].set_xticks(np.arange(0,len(r),5))
		axes[1].set_xticklabels(r[::5])
		axes[1].set_ylabel('aee')
		axes[1].set_yticks(np.arange(0,len(config_dict["aee"]),3))
		axes[1].set_yticklabels(np.around(config_dict["aee"][::3],1))
		plt.gca().invert_yaxis()
		axes[1].set_ylim(0,config_dict["aee"][-1])
		
		idx = np.argmin(np.abs(config_dict["aee"]-aee_fixed))
		axes[2].set_title('a_ee={:.1f}'.format(config_dict["aee"][idx]))
		im2=axes[2].imshow(array2d[:,:,idx].T,interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
		plt.colorbar(im2,ax=axes[2])
		axes[2].set_xlabel('r')
		axes[2].set_xticks(np.arange(0,len(r),5))
		axes[2].set_xticklabels(np.around(r[::5],1))
		axes[2].set_ylabel('s')
		axes[2].set_yticks(np.arange(0,len(s),5))
		axes[2].set_yticklabels(np.around(s[::5],1))
		axes[2].set_ylim(0,s[-1])
		plt.gca().invert_yaxis()

		idx = np.argmin(np.abs(config_dict["aee"]-aee_fixed))
		axes[3].set_title('a_ee={:.1f}'.format(config_dict["aee"][idx]))
		im2=axes[3].imshow(array2d[:,:,idx].T,interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
		plt.colorbar(im2,ax=axes[3])
		axes[3].set_xlabel('r')
		axes[3].set_xticks(np.arange(0,len(r),5))
		axes[3].set_xticklabels(np.around(r[::5],1))
		axes[3].set_ylabel(r'$a_{ei}*a_{ie}$')
		axes[3].set_yticks(np.arange(len(s))[::5])
		axes[3].set_yticklabels(np.around(s[::5]*(aii_fix+1),1))
		plt.gca().invert_yaxis()

	else:
		pass

	pdf_name = image_dir + 'Fig{}_phase_diag2d.pdf'.format(figure_no)
	plt.savefig(pdf_name, format='pdf', dpi=200)
	plt.close()



def plot_phase_diagram_area(figure_no,config_dict):
	"""
	plot size of relevant area (r<1.) of pattern forming regime in phase diagram
	"""
	alpha = config_dict["alpha"]
	r = config_dict["r"]
	s = config_dict["s"]
	delta = config_dict["delta"]
	aii_fix = config_dict["aii_fix"]
	aee = config_dict["aee"]
	mode = config_dict["mode"]

	if not isinstance(alpha,np.ndarray):
		alpha = np.array([alpha])

	if config_dict["alpha"] is None:
		def func_min(r,aii):
			"""condition det(k=0)>0; aee>s+1, written as fct of aii"""
			return (aii+delta-1)*(aii+1)
		def func_max(r,aii):
			"""det(kmin)<0; aee > (1+r)s**(1/(1+r))r**(-r/(1+r))
			written as function of aii"""
			return ( (aii+delta)/(1.+r)*r**(r/(1.+r)) )**(1.+r)*(1+aii)

		aii_fix_list = np.arange(0.1,16,0.1)
		upp_lim = 1.	## when sigma_e = sigma_i
		integrals = []
		for j,jaii_fix in enumerate(aii_fix_list):
			## find crossing point r* between two conditions
			low_lim = r[np.argmin(np.abs(func_max(r,jaii_fix)-func_min(r,jaii_fix)))]
			
			## integrate over area of both conditions from crossinpoint r* to r=1
			int_min,_ = quad(func_min,low_lim,upp_lim,args=(jaii_fix,))
			int_max,_ = quad(func_max,low_lim,upp_lim,args=(jaii_fix,))
			integrals.append(int_max-int_min)
		integrals = np.array(integrals)
		integrals[integrals<0] = np.nan
		fit_integrals = integrals[np.isfinite(integrals)]


		## find power law exponent for increase in area size
		y_fit = np.log(aii_fix_list[np.isfinite(integrals)])/np.log(10)
		plin,covlin = curve_fit(fct.linfct,y_fit,np.log(fit_integrals)/np.log(10),p0=[1,2.2])
		perr = np.sqrt(np.diag(covlin))
		## bootstrap fit error
		Nsur = 100
		data_size = fit_integrals.size
		sur_exponents = []
		for isur in range(Nsur):
			sur_integrals = np.random.choice(fit_integrals,size=data_size,replace=False)
			sur_plin,_ = curve_fit(fct.linfct,y_fit,np.log(fit_integrals)/np.log(10),p0=[1,2.2])
			sur_exponents.append(sur_plin[1])
		print('Bootstrap (N=100) Exponent: mean={}, std={}'.format(np.nanmean(sur_exponents),\
			 np.nanstd(sur_exponents)))
		print('Original data Exponent: mean={}, sem={}'.format(plin[1],perr[1]))
		
		nrow,ncol = 1,2
		fig = plt.figure(figsize=(ncol*6,nrow*5))
		ax = fig.add_subplot(nrow,ncol,1)
		ax.plot(aii_fix_list,integrals,'k')
		ax.set_xlabel('a_ii')
		ax.set_ylabel('Int(s1(r))-Int(s2(r))')
		ax.set_ylim(0,300)
		ax.set_xlim(0,16)

		ax = fig.add_subplot(nrow,ncol,2)
		ax.plot(np.log(aii_fix_list)/np.log(10),np.log(fit_integrals)/np.log(10),'sk',label="Data")
		ax.plot(np.log(aii_fix_list)/np.log(10),fct.linfct(np.log(aii_fix_list)/np.log(10),*plin),\
				'--r',label=r'Fit, exp={:.2f}$\pm${:.2f}'.format(plin[1],perr[1]))
		# ax.plot(np.log(aii_fix_list)/np.log(10),fct.linfct(np.log(aii_fix_list)/np.log(10),-2.72,3.13)+1,'--g',label='exp={:.2f},fact={:.2f}'.format(3.13,-2.72))
		ax.legend(loc='best')
		ax.set_xlabel('Log10 (a_ii)')
		ax.set_ylabel('Log10 (Int(s1(r))-Int(s2(r)))')
		# ax.set_ylim(0.000001,3)
		# ax.set_xlim(0.1,16)
		ax.legend(loc="best")

		pdf_name = image_dir + 'Fig{}_phase.pdf'.format(figure_no)
		plt.savefig(pdf_name, format='pdf', dpi=200)
		plt.close()

	else:
		alpha_list = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])#
		# alpha_list = np.array([0.9,1.])## alpha=0 doesnt allow for pf
		## calculate critical frequency kmax systematically across parameter settings
		max_lambda_alpha = fct.positive_lambda_alpha(s, alpha_list, aee, delta, mode)
		Z1_k_alpha = max_lambda_alpha[:,:,:,1]
		print('Z1_k_alpha',Z1_k_alpha.shape,Z1_k_alpha.size,np.sum(np.isfinite(Z1_k_alpha)))
		## calculate determinant at kmax
		print("Z1_k_alpha",np.nanmin(Z1_k_alpha),np.nanmax(Z1_k_alpha))
		print("s",np.nanmin(s),np.nanmax(s),np.nanmin(aee),np.nanmax(aee),mode)
		Z2_det_alpha = fct.neg_min_alpha(Z1_k_alpha, s, alpha_list, aee, delta, mode)
		print("Z2_det_alpha",np.nanmin(Z2_det_alpha),np.nanmax(Z2_det_alpha),Z2_det_alpha.size,\
			np.sum(np.isfinite(Z2_det_alpha)))
		Z2_alpha = np.empty((len(alpha_list),len(s)))*np.nan
		## for all values s,r take values aee for which det(kmax)<0
		for i,ir in enumerate(alpha_list):
			for j,js in enumerate(s):
				if np.sum(Z2_det_alpha[i,j,:]<0)>0:
					Z2_alpha[i,j] = aee[np.where(Z2_det_alpha[i,j,:]<0)[0][0]]	#aee(s)
		print('Z2_alpha',np.sum(np.isfinite(Z2_alpha)),Z2_alpha.shape)

		def func_min(r,aii):
			return (aii+delta-1)
		upp_lim = 1

		fig = plt.figure(figsize=(12,5))
		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,2)
		r_fix = 0.8
		r_fix_id = np.argmin([r-r_fix])
		
		aii_fix_list = np.arange(3,26,2.)#25
		##Z2_alpha_fixaii = neg_min_alpha_fixaii(Z1_k_alpha, s, alpha_list, aii_fix_list)
		
		diff = np.empty((len(aii_fix_list),len(alpha_list)))
		for j,jaii_fix in enumerate(aii_fix_list):
			# delta s: length of interval allowing pattern formation at fixed r=0.8 and aee=aii+delta
			aee_fix = jaii_fix + delta
			## condition 1 and 3 (indep of r)
			s_min = np.max([(aee_fix-1),aee_fix/(1.+r_fix)])
			## condition 4 at r=0.8
			s_max = s[np.nanargmin(np.abs(Z2_alpha - aee_fix),axis=1)]
			diff[j,:] = s_max - s_min
		
		freq = np.arange(0.01,5.,0.5)
		integrals = np.empty((len(aii_fix_list),len(alpha_list)))*np.nan
		for j,jaii_fix in enumerate(aii_fix_list):
			for k,kalpha in enumerate(alpha_list):
				
				aee_fix = jaii_fix + delta
				aii = (1-kalpha)*jaii_fix
				aii_a = kalpha*jaii_fix
				aeiaie = s[None,:,None]*(1+aii+aii_a)
				se = 1
				si = np.sqrt(r[:,None,None])
				
				neg_min_val = fct.detk(freq[None,None,:],aee_fix,aeiaie,aii,aii_a,se,si,si)	# r x s x k
				print('neg_min_val',neg_min_val.shape,freq.shape,r.shape,s.shape)
				has_neg_det = (np.sum(neg_min_val<0,axis=2)>0)*(neg_min_val[:,:,0]>0)
				neg_min_val[np.logical_not(has_neg_det),:] = np.nan
				
				fitx,fity = [],[]
				for lr in range(len(r)):
					try:
						this_s = s[np.where(np.isfinite(neg_min_val[lr,:,0]))[0][-1]]
						fitx.append(r[lr])
						fity.append(this_s)
					except:
						pass
				fitx = np.array(fitx)
				fity = np.array(fity)
				print('r[wx],s[wy]',kalpha,jaii_fix,fitx,fity)
				
				
				if True:
					figt = plt.figure(figsize=(12,6))
					ax = figt.add_subplot(121)
					im=ax.imshow(np.isfinite(neg_min_val[:,:,0]).astype(int),\
								 interpolation='nearest',cmap='binary')
					plt.colorbar(im,ax=ax)
					ax = figt.add_subplot(122)
					ax.plot(fitx,fity,'o-')
					plt.savefig(image_dir + 'test/neg_min_aii{}_alpha{}.pdf'.format(jaii_fix,kalpha),\
								dpi=100)
					plt.close(figt)
				
				print('fitx,fity',jaii_fix,kalpha,np.sum(np.isfinite(fitx)),\
					np.sum(np.isfinite(fity)),fitx.size,fity.size)
				func_max = interp1d(fitx,fity,kind='linear',copy=True)
				#func_max = func_max(fitx)
				
				if fitx[0]<upp_lim:
					low_lim = fitx[np.argmin(np.abs(func_max(fitx)-func_min(fitx,jaii_fix)))]
					print('low_lim',jaii_fix,kalpha,low_lim)
					int_min,_ = integrate.quad(func_min,low_lim,upp_lim,args=(jaii_fix,))
					
					int_max,_ = integrate.quad(func_max,low_lim,upp_lim)
					integrals[j,k] = int_max-int_min
				else:
					integrals[j,k] = np.nan
					
		print('integrals',integrals)
		integrals[integrals<0] = 0
		im=ax2.imshow(integrals,interpolation='nearest',cmap='binary',origin='lower',vmin=0,vmax=36)
		plt.colorbar(im,ax=ax2)
		ax2.set_xlabel('a_ii')
		ax2.set_ylabel('Alpha')
		
		diff[diff<0] = np.nan
		im=ax1.imshow(diff, interpolation='nearest',cmap='binary',origin='lower')
		plt.colorbar(im,ax=ax1)
		print('diff',np.nanmin(diff))
		try:
			ax1.contour(diff,[0,0.01],colors='m')
		except:
			pass
		ax1.set_title(r'$\Delta$s')
		ax2.set_title('Area')
		for ax in [ax1,ax2]:
			ax.set_yticks(np.arange(len(aii_fix_list))[::2])
			ax.set_yticklabels(aii_fix_list[::2])
			ax.set_xticks(np.arange(len(alpha_list))[::2])
			ax.set_xticklabels(alpha_list[::2])
			ax.set_xlabel('alpha')
			ax.set_ylabel('a_ii')
		
		pdf_name = image_dir + 'Fig{}.pdf'.format(figure_no)
		fig.savefig(pdf_name, format='pdf', dpi=300, bbox_inches='tight')