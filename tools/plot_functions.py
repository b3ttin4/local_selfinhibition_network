import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tools import functions as fct
from . import image_dir

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'



def plot_activity_patterns(figure_no,activity,lin_operator,config_dict,network_params):
	"""
	plot E and I final activity patterns
	"""
	see,sie,sei,sii = config_dict["conn_width_mean"]
	aee,aie,aei,aii = config_dict["conn_strength"]
	nE, nI = config_dict["system_size"],config_dict["system_size"]
	u = activity

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
		
		ax_E.plot(space,(u[:nE]),'-r',label=r"$u_E$")
		ax_I.plot(space,(u[nE:]),'-b',label=r"$u_I$")
		
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
		ax_ampl.set_ylim(bottom=0)
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
			ax.set_ylabel(r'Distance ($\sigma_E$)')
			ax.set_xlabel(r'Distance ($\sigma_E$)')
		ax_M.set_title('Recurrent connectivity')
		
		uE = u[:nE2].reshape(nE,nE)
		im=ax_E.imshow(uE,interpolation='nearest',cmap='binary')#,vmin=0,vmax=0.12)
		plt.colorbar(im,ax=ax_E)
		ax_E.set_title("Excitatory activity u_E")
		uI = u[nE2:].reshape(nE,nE)
		im=ax_I.imshow(uI,interpolation='nearest',cmap='binary')#,vmin=0,vmax=0.12)
		plt.colorbar(im,ax=ax_I)
		ax_I.set_title("Inhibitory activity u_I")
		
		im=ax_M.imshow(M1,interpolation="nearest",cmap="binary")
		plt.colorbar(im,ax=ax_M)


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
		ax.set_title('Spectrum(u_E)')
		im=ax_fftI.imshow(np.abs(fftuI)*nE*np.sqrt(2*np.pi),cmap="binary")
		plt.colorbar(im,ax=ax_fftI)
		ax.set_title('Spectrum(u_I)')
		
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

	
	plt.savefig(image_dir + "activity_Fig{}.pdf".format(figure_no))
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

	blue_s, bone_s, autm_s = fct.sequential_cmap(0,timesteps)

	ax_E = fig_greensfkt.add_subplot(231)
	ax_I = fig_greensfkt.add_subplot(232)
	
	ax_EI = fig_greensfkt.add_subplot(233)
	ax_DL = fig_greensfkt.add_subplot(234)
	ax_inp = fig_greensfkt.add_subplot(235)

	x = np.linspace(-1./2,1./2,nE,endpoint=False)/see
	k = np.arange(0,60,0.01)
	ew = fct.eigval1(k,aee,aei*aie,aii,see,sei,sii,network_params["tau"],\
						config_dict["alpha"])
	kmax = k[np.argmax(np.real(ew))]
	max_lambda = ew[np.argmax(np.real(ew))]

	
	lmax = 2*np.pi/kmax
	all_inpE, all_inpI = [],[]
	visbin_inp = 1
	for it in range(0,timesteps,visbin_inp):
		## Convolution response with connectivity
		inpEE = np.dot(M1[:nE,:nE],U[:nE,it]*nE - np.nanmean(U[:nE,0])*nE) *aee
		inpEI = np.dot(M1[nE:,:nE],U[nE:,it]*nE - np.nanmean(U[nE:,0])*nE) *aei
		all_inpE.append(inpEE)
		all_inpI.append(inpEI)
		
	ax_inp.plot(x,all_inpE[1],'--r',label=r"$I_{EE}, t=0$")
	ax_inp.plot(x,all_inpI[1],'--b',label=r"$I_{EI}, t=0$")
	ax_inp.plot(x,all_inpE[1]-all_inpI[1],'--k',label=r"$I_{EE}-I_{EI}, t=0$")
	
	ax_inp.plot(x,inpEE,'-r',label=r"$I_{EE}$"+",t={}".format(it))
	ax_inp.plot(x,inpEI,'-b',label=r"$I_{EI}$"+",t={}".format(it))
	ax_inp.plot(x,inpEE-inpEI,'-k',label=r"$I_{EE}-I_{EI}$"+",t={}".format(it))
	ax_inp.set_xlabel(r"Space ($\sigma_{EE}$")
	ax_inp.set_ylabel("Input to E")
	ax_inp.legend(loc="best",fontsize=10)

	diff,difffull,diff_fit,seeff,sieff,Aeff = [],[],[],[],[],[]
	seeff_th, sieff_th = [],[]
	visbin = 10
	start = 80
	print("timesteps",timesteps,U.shape)
	# for it in range(start,timesteps,visbin):
	for it in [timesteps//2,timesteps-1]:
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

		difffull.append( fct.deltasigma(k,aee,aie,aei,aii,see,sii,config_dict["alpha"],\
										it*network_params["dt"])/lmax )
		diff_fit.append((deltaI_fit-deltaE_fit)/lmax)
		
	time = np.linspace(start,timesteps,len(diff_fit))*network_params["dt"]
	ax_DL.plot(time,diff_fit,'-c',label='Fit to simulation')
	ax_DL.plot(time,difffull,'--',c='g',label='Theoretical analysis')
	ax_DL.legend(loc='best',fontsize=10)
	# ax_DL.set_ylim(0,0.05)
	ax_DL.set_xlabel('Time (tau)')
	ax_DL.set_ylabel('Diff variance/sigma_E^2')
	
	axlim = 10.
	for ax in [ax_E,ax_I]:
		ax.legend(loc="best")
		ax.set_xlim(-axlim,axlim)
		ax.set_xlabel('Distance [see]')
	ax_E.set_ylabel('Exc response')
	ax_I.set_ylabel('Inh response')

	## activity at last timestep
	ax_EI.plot(x,(U[:nE,-1]-U[0,-1])*nE,'-r',label="Exc., t={}".format(timesteps-1))
	ax_EI.plot(x,(U[nE:,-1]-U[nE,-1])*nE,'-b',label="Inh., t={}".format(timesteps-1))
	ax_EI.legend(loc='best',fontsize=10)
	ax_EI.set_xlabel('Distance [see]')
	ax_EI.set_ylabel('Resp at last time point')
	ax_EI.set_xlim(-7,7)


	plt.savefig(image_dir + "greensfkt_Fig{}.pdf".format(figure_no))
	plt.close(fig_greensfkt)