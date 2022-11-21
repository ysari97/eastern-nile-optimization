import os
import sys

import pandas as pd
import numpy as np

from model_nile import ModelNile

model = ModelNile()

outflow = model.object_by_name("BlueNile").streamflow[0]

outflow_deque = list(model.model_topology.out_edges([0], data="flow"))[0][-1]
outflow_deque.append(outflow)
my_list = list(model.model_topology.in_edges([1], data="flow"))
print([inflow[-1][0] for inflow in my_list])

# my_list[-1][-1].append(5)
# # my_list[-1][-1].append(9)
# print([inflow[-1] for inflow in my_list])
