# AttentiveDist

AttentiveDist is protein inter-residue distance prediction method based on deep learning. It combines information from different Multiple Sequnce Alignment (MSA's) using an attention mechanism.
The model is trained in multi-task fashion, predicting backbone and orientation angles as well.

**Reference**\
Aashish Jain, Genki Terashi, Yuki Kagaya, Sai Raghavendra Maddhuri Venkata Subramaniya, Charles Christoffer, and Daisuke Kihara, *Analyzing Effect of Quadruple Multiple Sequence Alignments on Deep Learning based Protein Inter-Residue Distance Prediction.* (In submission, 2021)

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)
Contact: Daisuke Kihara ([dkihara@purdue.edu](mailto:dkihara@purdue.edu))

## Setup
1. Clone the repository
2. Install required packages
	`pip install -r requirements.txt`

Typical installation time is around few minutes.

## Input

The inputs are generated using the following methods:

**DeepMSA** - generating a3m msa file\
**PSI-BLAST** - pssm file using the msa\
**HH-suite** - run hhmake to generate hmm profile using the msa\
**SPOT1D** - spot1d predicts the secondary structure and the solvent accessible surface area\
**CCMPRED** - pseudo-likelihood maximization\
**Mutual Information** - Computes MSA mutual information for pairwise resiudes\
`python feature/mutual_info.py --msa [MSA file] --out [Output file name]`\
Example: `python feature/mutual_info.py --msa example/input/T0997/10-3/T0997.a3m --out T0997.mutualinfo`\
**Contact Potential** - Computes contact potential for pairwise residues\
`python feature/potential.py --fasta [Fasta file] --out [Output file name]`\
Example: `python feature/potential.py --fasta example/input/T0997/T0997.fasta --out T0997.potential`

The inputs from DeepMSA, PSI-BLAST, CCMPRED and HH-suite are comupted using four e-values 0.001, 0.1, 1 and 10. Rest of the methods were run with standard parameters.

The inputs for CASP13 FM and TBM targets are provided. To download them:
```
wget http://kiharalab.org/attentivedist_data/input.tar.gz
tar -zxvf input.tar.gz	
```


## Featurization
All the inputs are converted into features using `feature_gen.py` script.
```
python feature/feature_gen.py --target [Target protein name] --input_dir [Input directory path]
```
Example:
`python feature/feature_gen.py --target T0997 --input_dir example/input/`

The target name should be same as used in the input directory. Please refer to the example input for how to structure the input files.

## Prediction
To run the model and generate inter-residue distance prediction we provide the script `predict.py`

For single target prediction
```
python predict.py --target [Target protein name] --input_dir [Input directory path] --out [Output directory path] --cuda
```
For multiple target prediction
```
python predict.py --target [File containing target names] --input_dir [Input directory path] --out [Output directory path] --cuda
```
Note:
1. Run the `feature_gen.py` and generate the feature before prediction.
2. The prediction will be saved as `[Target protein name]_prediction.npz`. It contains distance, backbone angle and orientation predictions and attention maps for the 4 MSA in the order 0.001, 0.1, 1 and 10. To get the predictions please refer to eval/eval_casp13.py code. Attention maps are stored under key "attention_maps".
3. Use `--cuda` if GPU is avaliable
4. Expected runtime (given inputs are present):
	GPU: ~10 sec,
	CPU: 1-2 minutes

Example:
```	
python predict.py --target T0997 --input_dir example/input/ --out example/output/ --cuda
```

## Evaluation
Code for CASP13 evaluation is in the `eval` folder.
We provide the predictions for CASP13 FM and TBM/FM targets. To do evaluation:
```
cd eval
wget http://kiharalab.org/attentivedist_data/casp13_pred.tar.gz
tar -zxvf casp13_pred.tar.gz
python eval_casp13.py --path ./casp13_predictions
```
Output (43 CASP13 FM and TBM/FM domains):
```
Casp13 long L/5 = 0.746
Casp13 long L/2 = 0.624
Casp13 long L/1 = 0.493
```

## Other Predictions
To generate sidechain center distance prediction
```
python other_predict.py --mode sce --target [Target protein name] --input_dir [Input directory path] --out [Output directory path] --cuda
```
Example:`python other_predict.py --mode sce --target T0997 --input_dir example/input/ --out example/output/ --cuda`

To generate N-O bond distance prediction
```
python other_predict.py --mode n_o --target [Target protein name] --input_dir [Input directory path] --out [Output directory path] --cuda
```
Example:`python other_predict.py --mode n_o --target T0997 --input_dir example/input/ --out example/output/ --cuda`

## 3D Structure Modeling by constraints

To generate the structures, we first need to predict the reference probability.
```
python predict_ref.py --target [Target protein name] --input_dir [Input directory path] --out [Output directory path]
```
Example:`python predict_ref.py --target T0997 --input_dir example/input/ --out example/output/`

This generates the final npz file which is used for modeling.

The modeling the done using pyRosetta. We modified the modeling part of trRosetta code. The modified code and LICENSE are available in modeling folder.
Please follow the instruction in modeling/README.md
