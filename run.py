import torch
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from triangle_gauss import *
from utilities import *
from problems import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="aaa")
# # N_partitions = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

parser.add_argument("--partition", type=float, default=0.05, help="min size of element")
parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
parser.add_argument("--hidden_size", type=int, default=20, help="hidden size")
args = parser.parse_args()

P_dgnet = Possion2d_dg(boundary_type='polygon', partition=args.partition, num_layers=args.num_layers, hidden_size=args.hidden_size, act='tanh')
for i in range(2):
    P_dgnet.train()
    P_dgnet.load()

