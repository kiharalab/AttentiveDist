import argparse
import os
from math import log
from collections import Counter
import numpy as np

def compute_mutual_info(msa_file):
    print("Started computing MI")
    f = open(msa_file,'r')
    msa = []
    for row in f:
        if '>' not in row:
            r = ''.join(x for x in row.strip() if not x.islower())
            msa.append(list(r))
    matrix = [[0 for i in range(len(msa[0]))] for j in range(len(msa[0]))]
    
    for i in range(len(msa[0])):
        for j in range(len(msa[0])):
            
            if i == j:
                matrix[i][j] = 0
            
            else:
    
                Pi = Counter(sequence[i] for sequence in msa)
                sumi = sum(Pi[aa_i] for aa_i in Pi)
                for aa_i in Pi:
                    Pi[aa_i] = Pi[aa_i]/sumi

                Pj = Counter(sequence[j] for sequence in msa)
                sumj = sum(Pj[aa_j] for aa_j in Pj)
                for aa_j in Pj:
                    Pj[aa_j] = Pj[aa_j]/sumj

                Pij = Counter((sequence[i],sequence[j]) for sequence in msa)
                sumij = sum(Pij[aa_ij] for aa_ij in Pij)
                for aa_ij in Pij:
                    Pij[aa_ij] = Pij[aa_ij]/sumij 

                matrix[i][j] = (sum(Pij[(x,y)]*log(Pij[(x,y)]/(Pi[x]*Pj[y])) for x,y in Pij))

    matrix = np.array(matrix)
    return matrix
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--msa', type=str, required=True, help='msa file')
    parser.add_argument('--out', type=str, required=True, help='output file')
    args = parser.parse_args()

    msa = args.msa
    out = args.out

    mi = compute_mutual_info(msa)
    np.savetxt(out, mi, fmt='%1.2f')
    print("MI computed")







        