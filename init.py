import argparse
from src.fastvss import FastVSS
import torch
import random
import os
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="Initialize FastVSS Model")
parser.add_argument("n_dimensions", type=int, help="Number of dimensions for FastVSS model")

args = parser.parse_args()
device = torch.device("cpu")  # Change this if you want to use GPU

pdf = pd.read_csv("WANDS/product.csv", sep="\t")
qdf = pd.read_csv("WANDS/query.csv", sep="\t")
ldf = pd.read_csv("WANDS/label.csv", sep="\t")

model = FastVSS(
    n_dimensions=args.n_dimensions,
    n_label=3,
    product_df=pdf,
    query_df=qdf,
    label_df=ldf,
    verbose=True,
    device=device,
)

print(f"Model initialized with {args.n_dimensions} dimensions.")
