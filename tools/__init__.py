import os

home_dir = os.environ["HOME"]
current_user = os.environ["USER"]

if current_user=="bettina":
	base_path = home_dir + "/physics/fias/projects/pattern_formation/publish/"
elif current_user=="bh2757":
	base_path = "/burg/theory/users/bh2757/columbia/projects/pattern_formation/"
else:
	base_path = home_dir + "/projects/pattern_formation/"
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
