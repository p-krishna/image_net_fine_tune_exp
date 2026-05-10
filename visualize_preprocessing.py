# visualize_preprocessing.py
# Saves each preprocessing step as a separate image file.
# Picks the LARGEST image ( by pixel area ) from the folder you specify.
#
# USAGE:
#   python visualize_preprocessing.py --folder /path/to/gtsrb/Train/0
#   python visualize_preprocessing.py --folder /path/to/gtsrb/Train/18

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESULTS_DIR


# ── Find largest image in folder ─────────────────────────────

def find_largest_image( folder: str ) -> str:
    """
    Scans folder for image files and returns the path of the
    largest one by pixel area ( width × height ).
    """
    EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
    candidates = [
        os.path.join( folder, f )
        for f in os.listdir( folder )
        if os.path.splitext( f )[1].lower() in EXTS
    ]
    if not candidates:
        raise FileNotFoundError( f"No image files found in: {folder}" )

    def area( path ):
        with Image.open( path ) as im:
            return im.width * im.height

    largest = max( candidates, key=area )
    with Image.open( largest ) as im:
        print( f"  Selected : {largest}" )
        print( f"  Size     : {im.width}×{im.height} px  ( {im.width*im.height:,} pixels )" )
    return largest


# ── Individual step savers ────────────────────────────────────

def save_step( img_array: np.ndarray, step_num: int, step_name: str,
              out_dir: str, extra_info: str = "" ):
    """Saves a single step image as step_N_<name>.png"""
    os.makedirs( out_dir, exist_ok=True )

    # Normalize float arrays for display
    if img_array.dtype != np.uint8:
        display = np.clip( img_array, 0, 1 )
    else:
        display = img_array

    fig, ax = plt.subplots( figsize=( 4, 4 ) )
    ax.imshow( display )
    ax.axis( "off" )

    title = f"Step {step_num}: {step_name}"
    if extra_info:
        title += f"\n{extra_info}"
    ax.set_title( title, fontsize=11, fontweight="bold", pad=10 )

    fname = f"step_{step_num:02d}_{step_name.lower().replace( ' ', '_' )}.png"
    path  = os.path.join( out_dir, fname )
    plt.savefig( path, dpi=150, bbox_inches="tight" )
    plt.close()
    print( f"  Saved → {path}" )
    return path


# ── Full pipeline ─────────────────────────────────────────────

def run_pipeline( folder: str ):
    img_path = find_largest_image( folder )
    img      = Image.open( img_path ).convert( "RGB" )
    folder_name = os.path.basename( os.path.normpath( folder ) )
    out_dir  = os.path.join( RESULTS_DIR, f"preprocessing_{folder_name}" )

    print( f"\n  Output dir: {out_dir}\n" )

    # ── Step 1: Original ──────────────────────────────────────
    arr_original = np.array( img )
    save_step( 
        arr_original, 1, "Original",
        out_dir,
        f"{img.width}×{img.height} px  |  dtype: uint8  |  range: 0–255"
     )

    # ── Step 2: Resize ────────────────────────────────────────
    resized     = img.resize( ( IMAGE_SIZE, IMAGE_SIZE ), Image.BILINEAR )
    arr_resized = np.array( resized )
    save_step( 
        arr_resized, 2, "Resize",
        out_dir,
        f"{IMAGE_SIZE}×{IMAGE_SIZE} px  |  dtype: uint8  |  range: 0–255"
     )

    # ── Step 3: Horizontal Flip ( RandXReflection ) ─────────────
    flipped     = T.RandomHorizontalFlip( p=1.0 )( resized )
    arr_flipped = np.array( flipped )
    save_step( 
        arr_flipped, 3, "Horizontal_Flip",
        out_dir,
        "RandXReflection ( p=1.0 forced for demo )"
     )

    # ── Step 4: X Translation ( RandXTranslation ) ──────────────
    x_trans     = T.RandomAffine( degrees=0, translate=( 0.15, 0.0 ) )( resized )
    arr_xtrans  = np.array( x_trans )
    save_step( 
        arr_xtrans, 4, "X_Translation",
        out_dir,
        "RandXTranslation  |  max shift: ±15% of width"
     )

    # ── Step 5: Y Translation ( RandYTranslation ) ──────────────
    y_trans     = T.RandomAffine( degrees=0, translate=( 0.0, 0.15 ) )( resized )
    arr_ytrans  = np.array( y_trans )
    save_step( 
        arr_ytrans, 5, "Y_Translation",
        out_dir,
        "RandYTranslation  |  max shift: ±15% of height"
     )

    # ── Step 6: ToTensor ──────────────────────────────────────
    tensor      = T.ToTensor()( resized )                    # [C,H,W] float32 0–1
    arr_tensor  = tensor.permute( 1, 2, 0 ).numpy()          # HWC for imshow
    save_step( 
        arr_tensor, 6, "ToTensor",
        out_dir,
        f"dtype: float32  |  range: {arr_tensor.min():.3f} – {arr_tensor.max():.3f}"
     )

    # ── Step 7: Normalize ─────────────────────────────────────
    normalized   = T.Normalize( mean=IMAGENET_MEAN, std=IMAGENET_STD )( tensor )
    arr_norm     = normalized.permute( 1, 2, 0 ).numpy()     # raw normalized values
    # Denormalize only for display ( so image is still visible )
    mean_np      = np.array( IMAGENET_MEAN )
    std_np       = np.array( IMAGENET_STD )
    arr_norm_vis = np.clip( arr_norm * std_np + mean_np, 0, 1 )

    save_step( 
        arr_norm_vis, 7, "Normalize",
        out_dir,
        f"ImageNet mean/std  |  actual range: {arr_norm.min():.3f} – {arr_norm.max():.3f}\n"
        f"mean={IMAGENET_MEAN}  std={IMAGENET_STD}"
     )

    print( f"\n  All 7 steps saved to: {out_dir}/" )
    print( f"  Files: step_01_original.png  →  step_07_normalize.png\n" )


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        description="Save each preprocessing step as a separate image file.\n"
                    "Automatically picks the largest image in the specified folder.",
        formatter_class=argparse.RawTextHelpFormatter,
     )
    parser.add_argument( 
        "--folder", required=True,
        help=( 
            "Path to a folder containing traffic sign images.\n"
            "Examples:\n"
            "  --folder /path/to/gtsrb/Train/0    ( circle signs )\n"
            "  --folder /path/to/gtsrb/Train/18   ( triangle signs )\n"
            "  --folder /path/to/gtsrb/Train/33   ( blue signs )"
         )
     )
    args = parser.parse_args()

    if not os.path.isdir( args.folder ):
        raise NotADirectoryError( f"Not a valid folder: {args.folder}" )

    run_pipeline( args.folder )