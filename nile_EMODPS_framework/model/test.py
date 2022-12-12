import os
import sys

import pandas as pd
import numpy as np

from model_nile import ModelNile

model = ModelNile()

print(model.overarching_policy.get_total_parameter_count())
