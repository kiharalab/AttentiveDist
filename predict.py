import argparse
import numpy as np
import os
from os.path import join, isfile
import torch
import torch.nn.functional as F

from dataset import get_input
from models.attentivedist import AttentiveDist
import util

def load_model(args, mode):

	if mode == "attn":
		attention = True
		num_blocks = 5
		blocks_after_attn = 40
	else:
		attention = False
		num_blocks = 40
		blocks_after_attn = 0

	model = AttentiveDist(in_channels=151, out_channels_dist=20, out_channels_angle=72, channels=64, num_blocks=num_blocks, dropout=0.1, 
					  dilation_list=[1,2,4], pool='Max', out_channels_omega=25 ,out_channels_theta=25 ,out_channels_phi=13,
					  attention=attention, blocks_after_attn=blocks_after_attn)
	if args.cuda:
		model.cuda("cuda")
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	if mode == "attn":
		checkpoint = torch.load(join(args.trained_models,"model_"+str(mode)),map_location=device)
		model.load_state_dict(checkpoint['state_dict'])
	else:
		model.load_state_dict(torch.load(join(args.trained_models,"model_"+str(mode)),map_location=device))
	model.eval()

	return model


def pred(args, target, model, mode):

	model.eval()

	target_dir = join(args.input_dir, target)
	input_features = get_input(target, target_dir)

	if args.cuda:
		input_features["10-3"] = input_features["10-3"].cuda()
		input_features["10-1"] = input_features["10-1"].cuda()
		input_features["1"] = input_features["1"].cuda()
		input_features["10"] = input_features["10"].cuda()

	if mode == "attn":
		dist_outs,omega_outs,theta_outs,orientation_phi_outs,phi_outs,psi_outs = model(input_features["10-3"],input_features["10-1"],input_features["1"],input_features["10"])
	else:
		dist_outs,omega_outs,theta_outs,orientation_phi_outs,phi_outs,psi_outs = model(input_features[mode])

	#Distance
	dist_outs = F.softmax(0.5 * (dist_outs + dist_outs.transpose(2,3)), dim=1)
	dist_pred = np.array(dist_outs.data.cpu())
	dist_pred = dist_pred[0]

	#Backbone angles
	phi_outs = F.softmax(phi_outs,dim=1)
	phi_pred = np.array(phi_outs.data.cpu())
	phi_pred = phi_pred[0]
	psi_outs = F.softmax(psi_outs,dim=1)
	psi_pred = np.array(psi_outs.data.cpu())
	psi_pred = psi_pred[0]

	#Orientation angles
	omega_outs = F.softmax(0.5 * (omega_outs + omega_outs.transpose(2,3)), dim=1)
	omega_pred = np.array(omega_outs.data.cpu())
	omega_pred = omega_pred[0]
	theta_outs = F.softmax(theta_outs,dim=1)
	theta_pred = np.array(theta_outs.data.cpu())
	theta_pred = theta_pred[0]
	orientation_phi_outs = F.softmax(orientation_phi_outs,dim=1)
	orientation_phi_pred = np.array(orientation_phi_outs.data.cpu())
	orientation_phi_pred = orientation_phi_pred[0]

	pred = {}
	pred["dist"] = dist_pred
	pred["phi"] = phi_pred
	pred["psi"] = psi_pred
	pred["omega"] = omega_pred
	pred["theta"] = theta_pred
	pred["orientation_phi"] = orientation_phi_pred

	return pred

def pred_attentivedist(args, models, target):
	pred_attn = pred(args, target, models["attn"], mode="attn")
	pred_10_3 = pred(args, target, models["10-3"], mode="10-3")
	pred_10_1 = pred(args, target, models["10-1"], mode="10-1")
	pred_1 = pred(args, target, models["1"], mode="1")
	pred_10 = pred(args, target, models["10"], mode="10")

	pred_attentive_dist = {}

	for k in pred_attn:
		pred_attentive_dist[k] = (pred_attn[k] + pred_10_3[k] + pred_10_1[k] + pred_1[k] + pred_10[k])/5
 
	np.savez(join(args.out, target + '_prediction'),
			dist=pred_attentive_dist["dist"],
			phi=pred_attentive_dist["phi"],
			psi=pred_attentive_dist["psi"],
			omega=pred_attentive_dist["omega"],
			theta=pred_attentive_dist["theta"],
			orientation_phi=pred_attentive_dist["orientation_phi"])

	print("AttentiveDist prediction done for %s"%(target))


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Use trained model to predict')
	parser.add_argument('--target', type=str, required=True, default="", help='target protein name')
	parser.add_argument('--trained_models', type=str, default="./trained_models", help='Model to load')
	parser.add_argument('--input_dir', type=str, default="./input", help='directory containing features')
	parser.add_argument('--out', type=str, default="./", help='directory to save the output')
	parser.add_argument("--cuda", action='store_true', help="Use GPU for prediction")
	args = parser.parse_args()

	target = args.target

	modes = ["attn", "10-3", "10-1", "1", "10"]
	models = {}
	for mode in modes:
		models[mode] = load_model(args, mode)

	if isfile(target):
		targets = util.read_target_list(target)
	else:
		targets = [target]

	for target in targets:
		pred_attentivedist(args, models, target)


