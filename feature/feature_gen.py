import argparse
import numpy as np
from os.path import join

import parse_feature

def generate_feature(target, evalue, input_dir):

    features = {}

    fasta_file = "%s/%s.fasta"%(target, target)
    sequence = parse_feature.get_seq(join(input_dir,fasta_file))
    features['sequence'] = sequence

    pssm_file = "%s/%s/%s.pssm"%(target, evalue, target)
    pssm = parse_feature.pssm_parser(join(input_dir,pssm_file))
    features['pssm'] = pssm

    hmm_file = "%s/%s/%s.hmmprofile"%(target, evalue, target)
    hmm = parse_feature.hmm_profile_parser(join(input_dir,hmm_file), sequence)
    features['hmm'] = hmm

    spot1d_file = "%s/%s.spot1d"%(target, target)
    spot1d = parse_feature.spot1d_parser(join(input_dir,spot1d_file))
    features['spot1d'] = spot1d

    onehot_encoding = parse_feature.get_onehotencoding(sequence)
    features['onehot'] = onehot_encoding

    ccmpred_file = "%s/%s/%s.ccmpred"%(target, evalue, target)
    ccmpred = parse_feature.ccmpred_parser(join(input_dir,ccmpred_file))
    features['ccmpred'] = ccmpred

    mutualinfo_file = "%s/%s/%s.mutualinfo"%(target, evalue, target)
    mutualinfo = parse_feature.mutualinfo_parser(join(input_dir,mutualinfo_file))
    features['mi'] = mutualinfo

    potential_file = "%s/%s.potential"%(target, target)
    potential = parse_feature.potential_parser(join(input_dir,potential_file))
    features['potential'] = potential

    return features

def main(args):

    target = args.target
    input_dir = args.input_dir
    features = {}
    evalues = ["10-3","10-1","1","10"]
    for evalue in evalues:
        features[evalue] = generate_feature(target, evalue, input_dir)

    feature_file = "%s/%s_features"%(target, target)
    np.savez(join(input_dir, feature_file), features=features)
    print("Features computed for target %s"%(target))
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--target', type=str, default="", help='target protein name')
    parser.add_argument('--input_dir', type=str, default="../input", help='directory containing input files')
    args = parser.parse_args()
    main(args)

