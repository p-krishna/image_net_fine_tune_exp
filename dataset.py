# dataset.py — GTSRBDataset, label remapping, DataLoader factory

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import DATASET_ROOT, LABEL_MAP, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS

# Auto-detect: use sampled CSVs if they exist, otherwise fall back to full
_sampled_train = os.path.join( DATASET_ROOT, "Train_sampled.csv" )
_sampled_test  = os.path.join( DATASET_ROOT, "Test_sampled.csv" )
_full_train    = os.path.join( DATASET_ROOT, "Train.csv" )
_full_test     = os.path.join( DATASET_ROOT, "Test.csv" )

TRAIN_CSV = _sampled_train if os.path.exists( _sampled_train ) else _full_train
TEST_CSV  = _sampled_test  if os.path.exists( _sampled_test )  else _full_test

print( f"  [dataset] Using train: {os.path.basename( TRAIN_CSV )}" )
print( f"  [dataset] Using test : {os.path.basename( TEST_CSV )}" )


class GTSRBDataset( Dataset ):
    """
    Reads Train.csv or Test.csv, remaps ClassId → 3-class label,
    drops excluded labels, and serves ( image_tensor, label ) pairs.
    """

    def __init__( self, csv_path: str, transform=None ):
        df = pd.read_csv( csv_path )
        # Keep only rows whose ClassId exists in LABEL_MAP
        df = df[df["ClassId"].isin( LABEL_MAP )].reset_index( drop=True )
        # Map original class → new 3-class label
        df["NewLabel"] = df["ClassId"].map( LABEL_MAP )
        self.df        = df
        self.transform = transform

    def __len__( self ):
        return len( self.df )

    def __getitem__( self, idx ):
        row      = self.df.iloc[idx]
        img_path = os.path.join( DATASET_ROOT, row["Path"] )
        label    = int( row["NewLabel"] )
        image    = Image.open( img_path ).convert( "RGB" )
        if self.transform:
            image = self.transform( image )
        return image, label

class GTSRBDatasetCached( Dataset ):
    """
    On first run: loads images, applies full transform, saves tensors to cache_dir.
    On subsequent runs: loads cached .pt tensors directly — no image I/O overhead.
    """

    def __init__( self, csv_path: str, transform, cache_dir: str ):
        df = pd.read_csv( csv_path )
        df = df[df["ClassId"].isin( LABEL_MAP )].reset_index( drop=True )
        df["NewLabel"] = df["ClassId"].map( LABEL_MAP )
        self.df        = df
        self.transform = transform
        self.cache_dir = cache_dir
        os.makedirs( cache_dir, exist_ok=True )
        print( f"  [Cache] Dir: {cache_dir}" )

    def __len__( self ):
        return len( self.df )

    def __getitem__( self, idx ):
        row        = self.df.iloc[idx]
        label      = int( row["NewLabel"] )
        cache_path = os.path.join( 
            self.cache_dir,
            row["Path"].replace( "/", "_" ).replace( "\\", "_" ) + ".pt"
         )

        if os.path.exists( cache_path ):
            tensor = torch.load( cache_path, weights_only=True )
        else:
            img_path = os.path.join( DATASET_ROOT, row["Path"] )
            image    = Image.open( img_path ).convert( "RGB" )
            tensor   = self.transform( image )
            torch.save( tensor, cache_path )

        return tensor, label

def get_transforms( augment: bool ):
    """
    augment=False → resize + normalize only
    augment=True  → adds RandXReflection, RandXTranslation, RandYTranslation
    """
    if augment:
        return T.Compose( [
            T.RandomHorizontalFlip( p=0.5 ),                    # RandXReflection
            T.RandomAffine( degrees=0, translate=( 0.1, 0.0 ) ),  # RandXTranslation
            T.RandomAffine( degrees=0, translate=( 0.0, 0.1 ) ),  # RandYTranslation
            T.ToTensor(),
            T.Normalize( mean=IMAGENET_MEAN, std=IMAGENET_STD ),
        ] )
    return T.Compose( [
        T.ToTensor(),
        T.Normalize( mean=IMAGENET_MEAN, std=IMAGENET_STD ),
    ] )


def get_dataloaders( batch_size: int, augment: bool ):
    train_transform = get_transforms( augment=augment )
    test_transform  = get_transforms( augment=False )
    train_dataset = GTSRBDatasetCached( 
        TRAIN_CSV, train_transform,
        cache_dir=os.path.join( "cache", "train_aug" if augment else "train" )
     )
    test_dataset = GTSRBDatasetCached( 
        TEST_CSV, test_transform,
        cache_dir=os.path.join( "cache", "test" )
     )

    train_loader = DataLoader( 
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,  # preload next batch
        persistent_workers=True if NUM_WORKERS > 0 else False,
     )
    test_loader = DataLoader( 
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=False,
     )
    return train_loader, test_loader, train_dataset, test_dataset