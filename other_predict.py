import argparse
import numpy as np
import os
from os.path import join, isfile
import torch
import torch.nn.functional as F

from dataset import get_input_single
from models.n_o import NOModel
from models.resnet import ResNet
import util

def load_model(args):
	mode = args.mode

	if mode == 'n_o':
		print("Getting N-O distance prediction for %s"%(target))
		model = NOModel(in_channels=51, out_channels_n_o=38, channels=64, num_blocks=25, dropout=0.1, dilation_list=[1,2,4])
		model_name = 'model_n_o'
	elif mode == 'sce':
		print("Getting Sidechain Center distance prediction for %s"%(target))
		model = ResNet(in_channels=51, out_channels=38, channels=64, num_blocks=25, dropout=0.1, dilation_list=[1,2,4])
		model_name = 'model_sce'
	else:
		raise Exception("%s is not a valid mode"%(str(mode)))

	if args.cuda:
		model.cuda("cuda")
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model.load_state_dict(torch.load(join(trained_models, model_name),map_location=device))
	model.eval()

	return model


def pred_n_o(args, target, model):

	model.eval()

	target_dir = join(args.input_dir, target)
	input_features = get_input_single(target, target_dir)

	if args.cuda:
		input_features["10-3"] = input_features["10-3"].cuda()

	n_o_outs, _ = model(input_features["10-3"])
	n_o_outs = F.softmax(n_o_outs,dim=1)
	n_o_pred = np.array(n_o_outs.data.cpu())
	n_o_pred = n_o_pred[0]

	np.save(join(args.out, target + '_n_o'),n_o_pred)

	print("N-O prediction done for %s"%(target))

def pred_sce(args, target, model):

	model.eval()

	target_dir = join(args.input_dir, target)
	input_features = get_input_single(target, target_dir)

	if args.cuda:
		input_features["10-3"] = input_features["10-3"].cuda()

	network_outputs = model(input_features["10-3"])
	network_outputs = F.softmax(network_outputs,dim=1)
	pred = np.array(network_outputs.data.cpu())
	pred = pred[0]

	np.save(join(args.out, target + '_sce'),pred)

	print("Sidechain Center prediction done for %s"%(target))

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Use trained model to predict')
	parser.add_argument('--target', type=str, required=True, default="", help='target protein name')
	parser.add_argument('--mode', type=str, required=True, default="sce", help='prediction mode, choose from n_o, sce')
	parser.add_argument('--trained_models', type=str, default="./trained_models", help='Model to load')
	parser.add_argument('--input_dir', type=str, default="./input", help='directory containing features')
	parser.add_argument('--out', type=str, default="./", help='directory to save the output')
	parser.add_argument("--cuda", action='store_true', help="Use GPU for prediction")
	args = parser.parse_args()

	target = args.target
	mode = args.mode
	trained_models = args.trained_models

	if isfile(target):
		targets = util.read_target_list(target)
	else:
		targets = [target]

	model = load_model(args)

	for target in targets:
		if mode == 'n_o':
			pred_n_o(args, target, model)
		elif mode == 'sce':
			pred_sce(args, target, model)
		else:
			raise Exception("%s is not a valid mode"%(str(mode)))


