

import config as cfg
from utils import ImageDataset, CNN_Sequential

model = CNN_Sequential(3, 28, 10)
print(model)