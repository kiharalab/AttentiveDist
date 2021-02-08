import os
import numpy as np

def create_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)

def read_target_list(list_fn):

	targets = []

	f = open(list_fn, 'r')
	for line in f:
	    targets.append(line.strip())

	f.close()
	
	return targets

def distance_bin_2_contact(distance_bin_pred_matrix):

	#Combine pred of first 9 bins
		# 1 bin of <4
		# 8 bins between 4-8 with 0.5 interval

	distance_bin_pred_matrix = np.array(distance_bin_pred_matrix)

	assert len(distance_bin_pred_matrix) == 20
	prot_len = len(distance_bin_pred_matrix[0][0])

	contact_map_prob = np.zeros((prot_len,prot_len))

	for i in range(0,9):
		contact_map_prob +=  np.array(distance_bin_pred_matrix[i])

	return contact_map_prob
