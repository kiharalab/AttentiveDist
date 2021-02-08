import argparse
import heapq
import json
import numpy as np
import os
from os.path import join
import pickle as pkl

def load(fileName): 
	fileObject2 = open(fileName, 'rb') 
	modelInput = pkl.load(fileObject2) 
	fileObject2.close() 
	return modelInput

def read_json(fn):
    with open(fn) as f:
        return json.load(f)
        
def get_separation(mode):
	if mode == 'long':
		return 24

def get_real_contacts(contact_map,mode):
	contacts = {}
	separation = get_separation(mode)
	for i in range(len(contact_map)):
		for j in range(len(contact_map[0])):
			if abs(i - j) >= separation:
				if contact_map[i][j] == 1:
					if i+1 in contacts:
						contacts[i+1].append(j+1)
					else:
						contacts[i+1] = [j+1]
	return contacts

def contact_accuracy(real_contacts,predicted_contacts):
	correct = 0
	incorrect = 0
	#Compare the two dict
	for i in predicted_contacts:
		if i in real_contacts:
			c = predicted_contacts[i]
			for j in c:
				if j in real_contacts[i]:
					correct += 1
				else:
					incorrect += 1

		else:
			incorrect += len(predicted_contacts[i])

	accuracy = float(correct)/float(correct + incorrect)
	return accuracy,correct,incorrect

def load_test_contacts(mode):
	if mode == 'long':
		file = 'casp13_L_contacts.json'
	contacts = {}
	with open(file) as f:
		data = json.load(f)
	return data

def load_domains():
	domains = {}
	file = 'domains'
	f = open(file,'r')
	for row in f:
		r = row.strip()
		t = r.split()
		if t[1] == 'evaluated':
			tmp = t[0].split(':')
			target = tmp[0].split('-')[0]
			domain_num = tmp[0].split('-')[1]
			seq_num = tmp[1]
			if target not in domains:
				domains[target] = {}
			domains[target][domain_num] = seq_num
	return domains

def inside_domain(domain_aa,i):
	#Check if i is within a domain
	domain_seqeunces = domain_aa.split(',')
	for cur_range in domain_seqeunces:
		low = int(cur_range.split('-')[0])
		high = int(cur_range.split('-')[1])
		if int(i) >= low and int(i) <= high:
			return True #If i is within any range, it is accepted
	return False


def get_predicted_contacts(contact_map,mode,l_by, domain_aa):
	contacts = {}
	separation = get_separation(mode)
	domain_seqeunces = domain_aa.split(',')
	total_length = 0
	for cur_range in domain_seqeunces:
		low = int(cur_range.split('-')[0])
		high = int(cur_range.split('-')[1])
		length = high - low + 1
		total_length = total_length + length
	l = int(total_length / l_by)
	contact_heap = []
	for i in range(len(contact_map)):
		for j in range(len(contact_map[0])):
			if abs(i - j) >= int(separation):
				val = -contact_map[i][j] #Doing negative to create a max heap
				heapq.heappush(contact_heap,(val,i+1,j+1))
	while ( l > 0):
		if len(contact_heap) > 0:
			prob,r,c = heapq.heappop(contact_heap)
			if inside_domain(domain_aa,r) and inside_domain(domain_aa,c): #Check both amino acid are inside domain
				if c not in contacts or r not in contacts[c]: # For i,j check if j,i is not already included
					if r in contacts:
						contacts[r].append(c)
					else:
						contacts[r] = [c]
					l = l - 1
		else:
			contacts[99999] = []
			while(l>0):
				contacts[99999].append(l)
				l = l - 1
	return contacts

def dist_2_contact(distance_pred,start_bin,end_bin,method=None):
	distance_pred = np.array(distance_pred)
	prot_len = len(distance_pred[0][0])
	contact_map_prob = np.zeros((prot_len,prot_len))
	# Add till 8A bin
	for i in range(start_bin,end_bin+1):
		contact_map_prob +=  np.array(distance_pred[i])
	return contact_map_prob

def evaluate_casp13(name, pred_contact, domains, real_contacts_long):

	domain_precs = []

	if name in domains: #domains contains only those proteins which were evaluated in casp13 for contact pred
		protein_domains = domains[name]
		for i in protein_domains:
			accuracy_info = {}

			for l_by in [5,2,1]:
				for dist in ['long']:

					predicted_contacts = get_predicted_contacts(pred_contact,dist,l_by,protein_domains[i])
					actual_contacts = real_contacts_long[name+'-'+i]
					actual_contacts = {int(k):[int(i) for i in v] for k,v in actual_contacts.items()}
					accuracy,_,_ = contact_accuracy(actual_contacts,predicted_contacts)
					accuracy_name = str(dist) + " L/"+str(l_by)
					accuracy_info[accuracy_name] = np.round(accuracy,3)
			# print(name,i,accuracy_info)
			domain_precs.append(accuracy_info)
	return domain_precs

def get_targets(path):
	targets = os.listdir(path)
	targets = [t for t in targets if "_prediction.npz" in t]
	for i in range(len(targets)):
		targets[i] = targets[i].replace("_prediction.npz","")
	return targets

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Compute L/n precision for CASP13 targets')
	parser.add_argument('--path', type=str, default='', help='path of casp13 predictions')
	args = parser.parse_args()

	real_contacts_long = load_test_contacts('long')
	domains = load_domains()

	path = args.path

	targets = get_targets(path)
	
	precisions = {}
	for l_by in [5,2,1]:
		for dist in ['long']:
			precisions[str(dist) + " L/"+str(l_by)] = []

	for target in targets:
		pred_distance = np.load(join(path,target+'_prediction.npz'))["dist"]
		pred_contact = dist_2_contact(pred_distance,0,8,'zprogram')
		precision = evaluate_casp13(target, pred_contact, domains, real_contacts_long)
		for domain_prec in precision:
			for i in domain_prec:
				precisions[i].append(domain_prec[i])

	for pr_mode in precisions:
	    avg_precision = np.around(np.mean(precisions[pr_mode]), decimals=3)
	    print("Casp13 "+ str(pr_mode) + " = "+str(avg_precision))
