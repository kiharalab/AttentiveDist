from collections import Counter
from math import log
import numpy as np
import os
from os import path
from os.path import join
import sys

def sigmoid(x):
	return 1/(1+np.exp(-x))

def ccmpred_parser(filename):
	data = np.loadtxt(filename)
	n, _ = data.shape
	matrix  = np.zeros((1,n,n),dtype = "float32")                       
	matrix[0, :, :] = data[:, :]
	return matrix

def mutualinfo_parser(filename):
	data = np.loadtxt(filename)
	n, _ = data.shape
	matrix  = np.zeros((1,n,n),dtype = "float32")                 
	matrix[0, :, :] = data[:, :]

	return matrix

def potential_parser(filename):
	data = np.loadtxt(filename)
	n, _ = data.shape
	matrix  = np.zeros((1,n,n),dtype = "float32")                 
	matrix[0, :, :] = data[:, :]
	return matrix

def pssm_parser(filename):
	#parse pssm file (generate by psiblast -Q)
	f = open(filename,'r')
	data = f.readlines()[3:-6]
	length = len(data)
	matrix = []
	for row in data:
		r = row.strip()
		tmp = r.split()
		aa_freq = tmp[2:2+20]
		for i in range(len(aa_freq)):
			aa_freq[i] = float(aa_freq[i])
		matrix.append(aa_freq)
	matrix = np.array(matrix)
	matrix = matrix.transpose()
	return sigmoid(matrix)

def spot1d_parser(filename):
	matrix = [[],[],[],[]]
	f = open(filename,'r')
	header = f.readline()
	length = 0
	for line in f:
		tmp = line.strip().split()
		asa = float(tmp[4])
		c = float(tmp[12])
		e = float(tmp[13])
		h = float(tmp[14])
		length += 1

		matrix[0].append(c)
		matrix[1].append(e)
		matrix[2].append(h)
		matrix[3].append(asa)
	matrix = np.array(matrix, dtype = "float32")
	return matrix

def get_seq(filename):
	f = open(filename,'r')
	f.readline()
	target_seq = "".join(s.strip() for s in f.readlines())
	f.close()
	return target_seq.strip()

def hmm_profile_parser(filename, sequence, asterisks_replace=0.0):

  with open(filename, 'r') as f:
    hhm_file = f.read()
  """Extracts information from the hmm file and replaces asterisks."""
  profile_part = hhm_file.split('#')[-1]
  profile_part = profile_part.split('\n')
  whole_profile = [i.split() for i in profile_part]
  # This part strips away the header and the footer.
  whole_profile = whole_profile[5:-2]
  gap_profile = np.zeros((len(sequence), 10))
  aa_profile = np.zeros((len(sequence), 20))
  count_aa = 0
  count_gap = 0
  for line_values in whole_profile:
    if len(line_values) == 23:
      # The first and the last values in line_values are metadata, skip them.
      for j, t in enumerate(line_values[2:-1]):
        aa_profile[count_aa, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_aa += 1
    elif len(line_values) == 10:
      for j, t in enumerate(line_values):
        gap_profile[count_gap, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_gap += 1
    elif not line_values:
      pass
    else:
      raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                       'got %d'%(line_values, len(line_values)))
  hmm_profile = np.hstack([aa_profile, gap_profile])
  assert len(hmm_profile) == len(sequence)
  hmm_profile = np.swapaxes(hmm_profile,0,1)
  return hmm_profile

def get_onehotencoding(sequence):
  aa = "ARNDCQEGHILKMFPSTWYV"
  seq_feature = np.zeros((20, len(sequence)))

  for i in range(len(sequence)):
    for j in range(len(aa)):
      if sequence[i] == aa[j]:
        seq_feature[j][i] = 1
  return seq_feature
