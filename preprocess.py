# preprocess.py — run ONCE before training
# Resizes all GTSRB images to 224×224 and saves into a new folder
# After this, set DATASET_ROOT in config.py to the new folder

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import DATASET_ROOT, TRAIN_CSV, TEST_CSV, LABEL_MAP, IMAGE_SIZE

OUTPUT_ROOT = DATASET_ROOT + "_resized"

def preprocess_split( csv_path, split_name ):
    df = pd.read_csv( csv_path )
    df = df[df["ClassId"].isin( LABEL_MAP )].reset_index( drop=True )
    print( f"\n  Processing {split_name}: {len( df )} images → {OUTPUT_ROOT}" )

    for _, row in tqdm( df.iterrows(), total=len( df ), desc=split_name ):
        src_path = os.path.join( DATASET_ROOT, row["Path"] )
        dst_path = os.path.join( OUTPUT_ROOT,  row["Path"] )
        os.makedirs( os.path.dirname( dst_path ), exist_ok=True )

        if not os.path.exists( dst_path ):   # skip already processed
            img = Image.open( src_path ).convert( "RGB" )
            img = img.resize( ( IMAGE_SIZE, IMAGE_SIZE ), Image.BILINEAR )
            img.save( dst_path )

    # Copy CSV as-is ( paths are relative, still valid )
    csv_name    = os.path.basename( csv_path )
    dst_csv     = os.path.join( OUTPUT_ROOT, csv_name )
    df_orig     = pd.read_csv( csv_path )
    df_orig.to_csv( dst_csv, index=False )
    print( f"  Saved CSV → {dst_csv}" )

if __name__ == "__main__":
    preprocess_split( TRAIN_CSV, "Train" )
    preprocess_split( TEST_CSV,  "Test" )
    print( f"\n  Done. Now set in config.py:" )
    print( f"  DATASET_ROOT = \"{OUTPUT_ROOT}\"" )