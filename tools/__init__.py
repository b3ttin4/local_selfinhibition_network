import os

# base_path = os.environ["HOMe"] + "pattern_formation/image/"
base_path = "/home/bettina/physics/fias/projects/pattern_formation/publish/"
image_dir = base_path + "image/"
if not os.path.exists(image_dir):
	os.makedirs(image_dir)
	print("Created image path: {}".format(image_dir))

## fixed params
network_params = {
				"inpE" : 1.,
				"inpI" : 1.,
				"tau" : 1.,
				"nonlinearity_rule" : 'rectifier',		#'rectifier'	'linear'
				"dt" : 0.01,
				"add_autapses" : True,

}
