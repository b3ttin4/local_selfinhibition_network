#!/usr/bin/env python


import sys
import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy import linalg
from scipy.optimize import curve_fit
import scipy.integrate as integrate


from tools import Runge_Kutta_Fehlberg,get_EI_np as get_EI,save_activity,\
functions as fct, parameter_settings, network_params, plot_functions


if __name__=="__main__":
	#base_path = "./image/"

	figure_no = "1d"


	## get parameter settings chosen in figure figure_no
	config_dict = parameter_settings.parameter_settings(figure_no)
	conn_strength = config_dict["conn_strength"]
	conn_width_mean = config_dict["conn_width_mean"]
	alpha = config_dict["alpha"]
	runtime = config_dict["runtime"]
	dim = config_dict["dim"]
	system_size = config_dict["system_size"]


	nE = system_size
	nI = system_size
	if dim==1:
		n = nE+nI
	else:
		nE2 = nE*nE
		n = nE2 + nI*nI

	add_autapses = True
	pulse_input = False
	if figure_no in ('4a',"4ab","4abcd"):
		pulse_input = True

	## nonlinearity
	if network_params["nonlinearity_rule"]=='linear':
		def fio(x):
			return x
	elif network_params["nonlinearity_rule"]=='rectifier':
		def fio(x):
			x[x<0] = 0
			return x


	## time constants
	taue = 1
	taui = network_params["tau"]*taue
	taulist = np.copy(network_params["tau"])
	if network_params["tau"]!=1.:
		taulist = np.ones((n))*taue
		if dim==1:
			taulist[nE:] = taui
		else:
			taulist[nE2:] = taui


	## Constant input
	timesteps = int(taue/network_params["dt"]*runtime)
	inp = network_params["inpE"]*np.ones((n,timesteps))
	## pulse like input at timepoint 0
	if pulse_input:
		## delta pulse to exc units
		inp[nE//2,0] += 1.
	if dim==1:
		inp[nE:,:] = network_params["inpI"]
	else:
		inp[nE2:,:] = network_params["inpI"]



	network_params.update({
					'input_shape' 			: np.array([nE, nI]),
					'conn_strength'			: conn_strength,
					'conn_width_mean'		: conn_width_mean,
					'runtime'				: runtime,
					'alpha'					: alpha,
					'timesteps'				: timesteps,
					'profile_mode'			: "gaussian",#'exponential',
					})



	## connectivity matrix
	M1 = get_EI.get_EI(dim, conn_width_mean, conn_strength, nE,nI, alpha, \
						conv_params=network_params)

	# mee = M1[:nE**2,:nE**2]
	# mei = M1[:nE**2,nE**2:]
	# mie = M1[nE**2:,:nE**2]
	# mii = M1[nE**2:,nE**2:]
	# fig = plt.figure()
	# ax = fig.add_subplot(231)
	# ax.imshow(M1,interpolation="nearest",cmap="binary")
	# ax = fig.add_subplot(232)
	# ax.imshow(mee[:,nE*(nE-1)//2].reshape(nE,nE),interpolation="nearest",cmap="binary")
	# ax = fig.add_subplot(233)
	# ax.imshow(mie[:,nE*(nE-1)//2].reshape(nE,nE),interpolation="nearest",cmap="binary")
	# ax = fig.add_subplot(234)
	# ax.imshow(mei[:,nE*(nE-1)//2].reshape(nE,nE),interpolation="nearest",cmap="binary")
	# ax = fig.add_subplot(235)
	# ax.imshow(mii[:,nE*(nE-1)//2].reshape(nE,nE),interpolation="nearest",cmap="binary")
	# plt.show()


	ew,vr = linalg.eig(M1,right=True)
	max_ew = ew[np.argmax(np.real(ew))]
	print('max_ew',max_ew,np.nanmax(np.real(ew)));sys.stdout.flush()
	# exit()

	def rhs(v,t):
		t_inp = int(np.ceil(t) if np.ceil(t)<timesteps else np.ceil(t)-1)
		return 1./taulist*(-v + fio(np.dot(M1,v) + inp[:,t_inp]) )


	## Initial condition
	rng = np.random.RandomState(948465)
	uinit = np.abs(rng.rand(n)*0.1)
	if pulse_input:
		aee,aie,aei,aii = conn_strength
		ue0 = 1.*(aei-aii-1.)/((aii+1)*(aee-1)-aei*aie)
		ui0 = 1.*(aee-aie-1.)/((aii+1)*(aee-1)-aei*aie)
		uinit[:nE] = ue0
		uinit[nE:] = ui0
	u = uinit
	U = np.zeros((n,timesteps))
	## Running dynamics
	for t in range(timesteps):
		u = Runge_Kutta_Fehlberg.rkf5_1param(rhs, u, network_params["dt"], t)
		# u = Runge_Kutta_Fehlberg.euler_1param(rhs, u, network_params["dt"], t)
		U[:,t] = u
	print('Finished simulation.');sys.stdout.flush()


	## Plotting
	plot_functions.plot_activity_patterns(figure_no,U,M1,config_dict,network_params)

	if pulse_input:
		plot_functions.plot_greensfunction(figure_no,U,M1,config_dict,network_params)
	print('Finished plotting.');sys.stdout.flush()






