import numpy as np
from os.path import join
import torch
from torch.autograd import Variable
from torch import FloatTensor

def _outer_concatenation(features_1d):
	max_len = features_1d.size()[2]
	a = features_1d.unsqueeze(3).repeat(1,1, 1, max_len)
	b = features_1d.unsqueeze(2).repeat(1, 1,max_len, 1)
	outs = torch.cat([a, b], dim=-3)
	return outs

def get_input(target, feature_dir):

	features = np.load(join(feature_dir, '{}_features.npz'.format(target)),allow_pickle=True)
	features_data = features["features"].item()

	data = {}

	evalues = ["10-3","10-1","1","10"]
	for evalue in evalues:
	
		inputs_1d = []
		inputs_2d = []
		_f_1d = []
		_f_2d = []

		_f_1d.append(features_data[evalue]["pssm"])
		_f_1d.append(features_data[evalue]["spot1d"])
		_f_1d.append(features_data[evalue]["hmm"])
		_f_1d.append(features_data[evalue]["onehot"])
		_f_2d.append(features_data[evalue]["ccmpred"])
		_f_2d.append(features_data[evalue]["mi"])
		_f_2d.append(features_data[evalue]["potential"])

		f_1d = _f_1d.pop(0)
		for i in _f_1d:
			f_1d = np.concatenate((f_1d, i), axis=0)
		inputs_1d.append(f_1d)

		f_2d = _f_2d.pop(0)
		for i in _f_2d:
			f_2d = np.concatenate((f_2d, i), axis=0)
		inputs_2d.append(f_2d)

		inputs_1d = Variable(FloatTensor(inputs_1d))
		inputs_2d = Variable(FloatTensor(inputs_2d))
		outs_1d = _outer_concatenation(inputs_1d)
		concat_2d = torch.cat([inputs_2d, outs_1d],1)
		data[str(evalue)] = concat_2d

	
	data["name"] = target

	return data

def get_input_single(target, feature_dir):

	features = np.load(join(feature_dir, '{}_features.npz'.format(target)),allow_pickle=True)
	features_data = features["features"].item()

	data = {}

	inputs_1d = []
	inputs_2d = []
	_f_1d = []
	_f_2d = []

	_f_1d.append(features_data["10-3"]["pssm"])
	_f_1d.append(features_data["10-3"]["spot1d"])
	_f_2d.append(features_data["10-3"]["ccmpred"])
	_f_2d.append(features_data["10-3"]["mi"])
	_f_2d.append(features_data["10-3"]["potential"])

	f_1d = _f_1d.pop(0)
	for i in _f_1d:
		f_1d = np.concatenate((f_1d, i), axis=0)
	inputs_1d.append(f_1d)

	f_2d = _f_2d.pop(0)
	for i in _f_2d:
		f_2d = np.concatenate((f_2d, i), axis=0)
	inputs_2d.append(f_2d)

	inputs_1d = Variable(FloatTensor(inputs_1d))
	inputs_2d = Variable(FloatTensor(inputs_2d))
	outs_1d = _outer_concatenation(inputs_1d)
	concat_2d = torch.cat([inputs_2d, outs_1d],1)
	data["10-3"] = concat_2d
	data["name"] = target

	return data

	