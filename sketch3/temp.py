import torch
import numpy as np

from modular_splicing.models_for_testing.list import LSSI
from modular_splicing.dataset.h5_dataset import H5Dataset

acceptor, donor = [x.model.eval() for x in LSSI]
acceptor.conv_layers[0].clipping = "none"
donor.conv_layers[0].clipping = "none"


print(acceptor)
print(donor)

data_path = "dataset_train_all.h5"

# dataset = genome
data = H5Dataset(
            path=data_path,
            cl=400,
            # **dataset_kwargs,
            cl_max=10_000,
            sl=5000,
            iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
            datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
            post_processor_spec=dict(type="IdentityPostProcessor"),
            )

clip = 200 # x starts and ends with padding
# remove window of overlap to concatenate

for datapoint in data:
    x, y = datapoint["inputs"]["x"], datapoint["outputs"]["y"]
    # x is one-hot encoded, 5400 by 4 (acgt)
    # 0 acceptor, 2 donor
    # predicts locations of acceptors (yp_acc = q matrix)

# yp_acc can be treated as col in q matrix

yp_acc = acceptor(torch.tensor(x[None]).float().cuda()).log_softmax(-1)[0,clip:-clip,1].detach().cpu().numpy()
yp_don = donor(torch.tensor(x[None]).float().cuda()).log_softmax(-1)[0,clip:-clip,2].detach().cpu().numpy()
# yp_null = 1 - yp_acc - yp_don



print(yp_acc[y == 1])
print(yp_don[y == 2])
                      
