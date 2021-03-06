import argparse
import numpy as np
import os
from os.path import join

import parse_feature

contact_potential = [
	[-0.20, 0.27, 0.24, 0.30,-0.26, 0.21, 0.43,-0.03, 0.21,-0.35,-0.37, 0.20,-0.23,-0.33, 0.07, 0.15, 0.00,-0.40,-0.15,-0.38 ],
	[ 0.27, 0.13, 0.02,-0.71, 0.32,-0.12,-0.75, 0.14, 0.04, 0.18, 0.09, 0.50, 0.17, 0.08,-0.02, 0.12, 0.00,-0.41,-0.37, 0.17 ],
	[ 0.24, 0.02,-0.04,-0.12, 0.28,-0.05,-0.01, 0.10, 0.10, 0.55, 0.36,-0.14, 0.32, 0.29, 0.13, 0.14, 0.00,-0.09, 0.01, 0.39 ],
	[ 0.30,-0.71,-0.12, 0.27, 0.38, 0.12, 0.40, 0.17,-0.22, 0.54, 0.62,-0.69, 0.62, 0.48, 0.25, 0.01, 0.00, 0.06,-0.07, 0.66 ],
	[-0.26, 0.32, 0.28, 0.38,-1.34, 0.04, 0.46,-0.09,-0.19,-0.48,-0.50, 0.35,-0.49,-0.53,-0.18, 0.09, 0.00,-0.74,-0.16,-0.51 ],
	[ 0.21,-0.12,-0.05, 0.12, 0.04, 0.14, 0.10, 0.20, 0.22, 0.14, 0.08,-0.20,-0.01,-0.04,-0.05, 0.25, 0.00,-0.11,-0.18, 0.17 ],
	[ 0.43,-0.75,-0.01, 0.40, 0.46, 0.10, 0.45, 0.48,-0.11, 0.38, 0.37,-0.87, 0.24, 0.34, 0.26, 0.10, 0.00,-0.15,-0.16, 0.41 ],
	[-0.03, 0.14, 0.10, 0.17,-0.09, 0.20, 0.48,-0.20, 0.23, 0.21, 0.14, 0.12, 0.08, 0.11,-0.01, 0.10, 0.00,-0.24,-0.04, 0.04 ],
	[ 0.21, 0.04, 0.10,-0.22,-0.19, 0.22,-0.11, 0.23,-0.33, 0.19, 0.10, 0.26,-0.17,-0.19,-0.05, 0.15, 0.00,-0.46,-0.21, 0.18 ],
	[-0.35, 0.18, 0.55, 0.54,-0.48, 0.14, 0.38, 0.21, 0.19,-0.60,-0.79, 0.21,-0.60,-0.65, 0.05, 0.35, 0.00,-0.65,-0.33,-0.68 ],
	[-0.37, 0.09, 0.36, 0.62,-0.50, 0.08, 0.37, 0.14, 0.10,-0.79,-0.81, 0.16,-0.68,-0.78,-0.08, 0.26, 0.00,-0.70,-0.44,-0.80 ],
	[ 0.20, 0.50,-0.14,-0.69, 0.35,-0.20,-0.87, 0.12, 0.26, 0.21, 0.16, 0.38, 0.22, 0.11, 0.12, 0.10, 0.00,-0.28,-0.40, 0.16 ],
	[-0.23, 0.17, 0.32, 0.62,-0.49,-0.01, 0.24, 0.08,-0.17,-0.60,-0.68, 0.22,-0.56,-0.89,-0.16, 0.32, 0.00,-0.94,-0.51,-0.47 ],
	[-0.33, 0.08, 0.29, 0.48,-0.53,-0.04, 0.34, 0.11,-0.19,-0.65,-0.78, 0.11,-0.89,-0.82,-0.19, 0.10, 0.00,-0.78,-0.49,-0.67 ],
	[ 0.07,-0.02, 0.13, 0.25,-0.18,-0.05, 0.26,-0.01,-0.05, 0.05,-0.08, 0.12,-0.16,-0.19,-0.07, 0.17, 0.00,-0.73,-0.40,-0.08 ],
	[ 0.15, 0.12, 0.14, 0.01, 0.09, 0.25, 0.10, 0.10, 0.15, 0.35, 0.26, 0.10, 0.32, 0.10, 0.17, 0.13, 0.00, 0.07, 0.07, 0.25 ],
	[ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ],
	[-0.40,-0.41,-0.09, 0.06,-0.74,-0.11,-0.15,-0.24,-0.46,-0.65,-0.70,-0.28,-0.94,-0.78,-0.73, 0.07, 0.00,-0.74,-0.55,-0.62 ],
	[-0.15,-0.37, 0.01,-0.07,-0.16,-0.18,-0.16,-0.04,-0.21,-0.33,-0.44,-0.40,-0.51,-0.49,-0.40, 0.07, 0.00,-0.55,-0.27,-0.27 ],
	[-0.38, 0.17, 0.39, 0.66,-0.51, 0.17, 0.41, 0.04, 0.18,-0.68,-0.80, 0.16,-0.47,-0.67,-0.08, 0.25, 0.00,-0.62,-0.27,-0.72 ]
	];

def get_pairwise_potential(i,j):
	aa = "ARNDCQEGHILKMFPSTWYV"
	return contact_potential[aa.index(i)][aa.index(j)]
		
def compute_potential(seq):
	matrix = [[0 for i in range(len(seq))] for j in range(len(seq))]
	for i in range(len(seq)):
		for j in range(len(seq)):
			matrix[i][j] = get_pairwise_potential(seq[i],seq[j])
	matrix = np.array(matrix)
	return matrix
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Preprocessing')
	parser.add_argument('--fasta', type=str, required=True, help='fasta file')
	parser.add_argument('--out', type=str, required=True, help='output file')
	args = parser.parse_args()

	fasta_file = args.fasta
	out_file = args.out

	sequence = parse_feature.get_seq(fasta_file)
	potential = compute_potential(sequence)
	np.savetxt(out_file, potential, fmt='%1.2f')

	


