import numpy as  np
import sys


def parameter_settings(figure_no):
	"""
	returns parameter settings for given figure panel

	input:
	figure_no : figure number

	output:
	config_dict: dictionary of network parameters
	"""

	#############################################################
	## Parameter settings for numerical simulation of network
	if figure_no in ("1b","1c","1d","3g","4abcd","5ab_left","5ab_right","6a_top","6a_bottom",\
					 "6a_bottom_test"):
		## Figure 1
		if figure_no=='1b':
			aii = 0.0
			conn_strength = np.array([1.95+aii,1.4+aii,aii+0.9,aii])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/200.
			alpha = 1.0
			runtime = 500
			dim = 1
			system_size = 256
		elif figure_no in ('1c',):
			aii = 3.0
			conn_strength = np.array([1.95+aii,1.4+aii,aii+0.9,aii])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/200.
			alpha = 1.0
			runtime = 500
			dim = 1
			system_size = 256
		elif figure_no in ('1d',):
			aii = 3.0
			conn_strength = np.array([1.95+aii,1.4+aii,aii+0.9,aii])
			conn_width_mean = np.array([3.3, 3.3, 2.7, 2.7])/70.
			alpha = 1.0
			runtime = 500
			dim = 2
			system_size = 30
			
		## Figure 3
		elif figure_no=='3g':
			conn_strength = np.array([11.4,11.5,10.9,10.])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/60.
			alpha = 0.2
			runtime = 500
			dim = 2
			system_size = 30
		
		## Figure 4
		elif figure_no=='4abcd':
			conn_strength = np.array([11.4,11.,10.9,10.])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/50.
			alpha = 1.
			runtime = 5
			dim = 1
			system_size = 256


		## Figure 5
		elif figure_no=='5ab_left':
			# conn_strength = np.array([22.8,22.2,22.2,21.4])
			# conn_width_mean = np.array([3.3,3.3,2.7,2.7])/80.
			conn_strength = np.array([11.4,10.5,10.9,10.])
			conn_width_mean = np.array([3.3, 3.3, 1.2, 1.2])/75.
			alpha = 1.
			runtime = 30
			dim = 1
			system_size = 256
		elif figure_no=='5ab_right':
			# conn_strength = np.array([22.8,41.7,22.3,21.4])
			# conn_width_mean = np.array([3.3,3.3,2.35,2.35])/82.#82.
			conn_strength = np.array([11.4,21.5,10.9,10.])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/68.
			alpha = 1.0
			runtime = 50
			dim = 1
			system_size = 256
			
		## Figure 6
		elif figure_no=="6a_top":
			conn_strength = np.array([11.4,11.5,10.9,10.])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/75.
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
		elif figure_no=="6a_bottom_test":
			conn_strength = np.array([11.4,10.8,10.9,10.])
			conn_width_mean = np.array([3.3, 3.3, 2.9, 2.9])/60.
			alpha = 0.15
			runtime = 500
			dim = 2
			system_size = 40

		#############################################################
		config_dict = {
						"conn_strength" : conn_strength,
						"conn_width_mean" : conn_width_mean,
						"alpha" : alpha,
						"runtime" : runtime,
						"dim" : dim,
						"system_size" : system_size,
		}

	#############################################################
	## Parameter settings for numerical simulation of network
	elif figure_no in ("2a_left","2a_right","2b_zero_aii","2b_finite_aii","2c","2d","3b",\
						"3cd","3e","3f","5c_left","5c_right","5d","5e","5f"):
		## Figure 2
		if figure_no=="2a_left":
			mode = "3"
			show_feat = None
			plot_style = None
			npts = 40
			s = np.linspace(0.01,1,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(0.01,2,20)
			zmax = None
			aii_fix = None
			delta = None
			alpha = None
		elif figure_no=="2a_right":
			mode = "3"
			show_feat = None
			plot_style = None
			npts = 40
			s = np.linspace(0.01,4,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(0.01,2.5,20)
			zmax = None
			aii_fix = None
			delta = None
			alpha = None
		elif figure_no=="2b_zero_aii":
			mode = "3"
			show_feat = None
			plot_style = "cuts"
			npts = 40
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.array([1.95])
			zmax = None
			aii_fix = 0.0
			delta = 1.95
			alpha = None
		elif figure_no=="2b_finite_aii":
			mode = "3"
			show_feat = None
			plot_style = "cuts"
			npts = 40
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.array([4.95])
			zmax = None
			aii_fix = 3.0
			delta = 1.95
			alpha = None
		elif figure_no=="2c":
			mode = "3"
			show_feat = None
			plot_style = "cuts"
			npts = 40
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.array([11.95])
			zmax = None
			aii_fix = 10.
			delta = 1.95
			alpha = None
		elif figure_no=="2d":
			mode = "3"
			show_feat = None
			plot_style = "lineplot"
			npts = 40
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.array([11.95])
			zmax = None
			aii_fix = 10.
			delta = 1.95
			alpha = None

		## Figure 3
		elif figure_no=="3b":
			mode = "3"
			show_feat = None
			plot_style = None
			npts = 40
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(1.8,40.2,25)
			zmax = None
			aii_fix = None
			delta = 1.4
			alpha = np.array([0.5])
		elif figure_no=="3cd":
			mode = "3"
			show_feat = None
			plot_style = None
			npts = 100
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(1.8,40.2,200)
			zmax = None
			aii_fix = 10.
			delta = 1.4
			alpha = np.array([0.0,0.1,0.2,0.8])
		elif figure_no=="3e":
			mode = "3"
			show_feat = "min_alpha"
			plot_style = None
			npts = 20
			s = np.linspace(0.01,40,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(0.01,40,11)
			zmax = None
			aii_fix = 10.
			delta = 1.4
			alpha = np.linspace(0,1.,20)
		elif figure_no=="3f":
			mode = "3"
			show_feat = None
			plot_style = "lineplot"
			npts = 40
			s = np.linspace(1.01,50,npts+1)
			r = np.linspace(0.01,1.,npts)
			aee = np.linspace(2.,40,20)#5.2#60#120#4
			zmax = None
			aii_fix = 10.
			delta = 1.95
			alpha = np.linspace(0,1.,20)
				

		## Figure 5
		elif figure_no in ("5c_left","5c_right","5f"):
			mode = "3"
			if figure_no=="5d":
				show_feat = "ratio"
			elif figure_no=="5f":
				show_feat = "ratio_norm"
			plot_style = None
			npts = 40
			s = np.linspace(0.01,60,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(0.01,40,10)
			zmax = None
			aii_fix = 10.
			delta = 1.95
			if figure_no=="5c_left":
				alpha = np.array([1.])
			else:
				alpha = np.array([0.2])
		elif figure_no in ("5d","5e"):
			mode = "3"
			if figure_no=="5d":
				show_feat = "ratio"
			elif figure_no=="5e":
				show_feat = "baseline"
			plot_style = "cuts"
			npts = 40
			s = np.linspace(0.01,60,npts+1)
			r = np.linspace(0.01,2.,npts)
			aee = np.linspace(0.01,40,11)
			zmax = None
			aii_fix = 10.
			delta = 1.95
			alpha = np.array([1.])



		config_dict = {
						"mode" : mode,
						"feature" : show_feat,
						"plot_style" : plot_style,
						"num_points" : npts,
						"s" : s,
						"r" : r,
						"aee" : aee,
						"zmax" : zmax,
						"aii_fix" : aii_fix,
						"delta" : delta,
						"alpha" : alpha,
					}

	else:
		print("Selected figure '{}' not known! Exit!".format(figure_no))
		sys.exit()
	



		
	return config_dict
