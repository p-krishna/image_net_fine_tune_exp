"""
All hyperparameters, paths, label maps, experiment definitions
"""

import os


# DATASET PATHS — edit these to match your system

DATASET_ROOT = "data"
TRAIN_CSV    = os.path.join( DATASET_ROOT, "Train.csv" )
TEST_CSV     = os.path.join( DATASET_ROOT, "Test.csv" )


# OUTPUT DIRECTORIES ( relative to where you run main.py )

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"


# LABEL REMAPPING
# Original 43 classes → 3 shape categories
# Excluded labels ( 6,12,14,32,41,42 ) are dropped entirely

LABEL_MAP = {
    # CIRCLES → 0
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    7: 0, 8: 0, 9: 0, 10: 0,
    15: 0, 16: 0, 17: 0,
    # TRIANGLES → 1
    11: 1, 13: 1,
    18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1,
    24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
    30: 1, 31: 1,
    # BLUE SIGNS → 2
    33: 2, 34: 2, 35: 2, 36: 2, 37: 2,
    38: 2, 39: 2, 40: 2,
    # EXCLUDED: 6, 12, 14, 32, 41, 42 — omitted → rows dropped at load time
}

CLASS_NAMES = ["Circles", "Triangles", "Blue Signs"]
NUM_CLASSES  = 3
IMAGE_SIZE   = 224   # standard input for all 4 pretrained models


# IMAGENET NORMALIZATION STATS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# EXPERIMENTS

EXPERIMENTS = [
    {"id": "exp1", "lr": 1e-3, "batch_size": 16, "max_epochs":  5, "augment": False},
    {"id": "exp2", "lr": 1e-5, "batch_size": 32, "max_epochs": 15, "augment": True},
]


# MODELS TO TRAIN ( in order )

MODELS = ["alexnet", "vgg16", "resnet50", "efficientnet_b0"]


# DATALOADER WORKERS
# Linux/Mac: 2-4   |   Windows: 0

NUM_WORKERS = 2