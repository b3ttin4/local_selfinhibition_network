import numpy as  np
import sys


def parameter_settings(figure_no):
	"""
	returns parameter settings for given figure panel
	"""
	## Figure 1
	if figure_no=='1b':
		aii = 0.0
		conn_strength = np.array([1.95+aii,1.4+aii,aii+0.9,aii])
		conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/200.
		alpha = 1.0
		runtime = 500
		dim = 1
		system_size = 256
	elif figure_no in ('1c','1d'):
		aii = 3.0
		conn_strength = np.array([1.95+aii,1.4+aii,aii+0.9,aii])
		conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/200.
		alpha = 1.0
		runtime = 500
		dim = 1
		system_size = 256
		if figure_no=="1d":
			dim = 2
			system_size = 50
	
	## Figure 3
	elif figure_no=='3g':
		conn_strength = np.array([11.4,11.5,10.9,10.])
		conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/60.
		alpha = 0.2
		runtime = 500
		dim = 2
		system_size = 40
	
	## Figure 4
	elif figure_no=='4a':
		conn_strength = np.array([11.4,11.,10.9,10.])
		conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/50.
		alpha = 1.
		runtime = 1
		dim = 1
		system_size = 256

	## Figure 5
	elif figure_no=='5a_left':
		conn_strength = np.array([22.8,30.5,22.2,21.4])
		conn_width_mean = np.array([3.3,3.3,2.7,2.7])/80.
		alpha = 1.0
		runtime = 50
		dim = 1
		system_size = 256
	elif figure_no=='5a_right':
		conn_strength = np.array([22.8,41.7,22.3,21.4])
		conn_width_mean = np.array([3.3,3.3,2.4,2.4])/80.
		alpha = 1.0
		runtime = 50
		dim = 1
		system_size = 256
		
	## Figure 6
	elif figure_no=="6a_top":
		conn_strength = np.array([11.4,11.5,10.9,10.])
		conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/60.
		alpha = 0.2
		runtime = 500
		dim = 2
		system_size = 40
	elif figure_no=="6a_bottom":
		conn_strength = np.array([11.4,10.8,10.9,10.])
		conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/75.
		alpha = 0.2
		runtime = 500
		dim = 2
		system_size = 40
	
	else:
		print("Selected figure not known! Exit!")
		sys.exit()
	

	# if dim==1:
	# 	system_size = 64
	# if dim==2:
	# 	system_size = 35

	config_dict = {
					"conn_strength" : conn_strength,
					"conn_width_mean" : conn_width_mean,
					"alpha" : alpha,
					"runtime" : runtime,
					"dim" : dim,
					"system_size" : system_size,
	}		
	return config_dict
		


## ======================= ======================= ================== ##
## ======================= EE, IE, EI, II =========================== ##

## ================= lateral inhibition ============================= ##
conn_strength = np.array([22.2,21.6,21.6,20.8])
conn_width_mean = np.array([3.3, 3.3, 4.6, 4.6])/60.

## ======  used in heterogeneous system  ============================ ##
##conn_strength = np.array([22.2,6.,31.,20.8])#aii=20.8#aei=24.5 no pf, det(0)>0
conn_strength = np.array([22.,21.6,21.6,20.8])#aii=20.8#aei=24.5
conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/100.#1.

### parameter settings for 2pop manuscript, full model (alpha=0.2) and for different crit wavelengths
conn_strength = np.array([11.4,11.,10.9,10.])
#conn_strength = np.array([11.8,11.,10.9,10])
#conn_strength = np.array([11.4,13.5,10.9,10])
conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/50.#2.9
### out of PF regime:
#conn_width_mean = np.array([3.3,3.3,1.2,1.2])/60.

