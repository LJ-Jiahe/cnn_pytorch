

import config as cfg
from utils import ImageDataset



training_dataset = ImageDataset(data_dir=cfg.data_dir,
                                input_dir=cfg.training_input_dir,
                                target_dir=cfg.training_target_dir,
                                transform=cfg.transform)