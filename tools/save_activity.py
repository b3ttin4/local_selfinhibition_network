import h5py
import numpy as np
import sys


def save_activity(activity, network_params, filename, folder_index, base_path, activity_key='activity',dim=2):
	filepath = base_path + 'pattern_formation/data{}d/'.format(dim)
	full_name = filepath + filename
	f = h5py.File(full_name,'a')

	f.create_dataset(folder_index + activity_key, data=activity)
	if activity_key=='activity':
		f.create_dataset(folder_index + 'nevents', data=activity.shape[0])
	
	for key in network_params.keys():
		f.create_dataset(folder_index + key, data=network_params[key])

	f.close()
	
		
	
def gimme_index(filename,base_path,dim=2):
	filepath = base_path + 'pattern_formation/data{}d/'.format(dim)
	full_name = filepath + filename
	print('save under:',full_name)
	try:
		f = h5py.File(full_name,'a')
		indices = [int(item) for item in sorted(f.keys())]
		max_index = np.max(indices)
		f.create_group('{}'.format(max_index+1))
		f.close()
	except Exception as e:
		#print(e)
		max_index = -1
	return max_index + 1
	
