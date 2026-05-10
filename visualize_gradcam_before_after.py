# visualize_gradcam_before_after.py
#
# Generates Grad-CAM heatmaps for a single image using:
#   - Pretrained weights BEFORE fine-tuning ( random head )
#   - Fine-tuned weights AFTER exp1 ( epoch 5 checkpoint )
# Saves each output as a separate file.
#
# USAGE:
#   python visualize_gradcam_before_after.py --image /path/to/sign.png --model alexnet
#   python visualize_gradcam_before_after.py --image /path/to/sign.png --model vgg16

import os
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from config import ( 
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    RESULTS_DIR, CLASS_NAMES, CHECKPOINT_DIR
 )

NUM_CLASSES = len( CLASS_NAMES )


# ─────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────

def build_model( model_name: str, freeze_backbone: bool = True ) -> nn.Module:
    if model_name == "alexnet":
        model = models.alexnet( weights=models.AlexNet_Weights.DEFAULT )
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.classifier[6] = nn.Linear( 4096, NUM_CLASSES )

    elif model_name == "vgg16":
        model = models.vgg16( weights=models.VGG16_Weights.DEFAULT )
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.classifier[6] = nn.Linear( 4096, NUM_CLASSES )

    else:
        raise ValueError( f"Unsupported model: {model_name}" )

    return model


# ─────────────────────────────────────────────
# LAST CONV LAYER SELECTOR
# ─────────────────────────────────────────────

def get_last_conv( model_name: str, model: nn.Module ) -> nn.Module:
    """Returns the last conv layer — the target for Grad-CAM hooks."""
    if model_name == "alexnet":
        return model.features[10]   # Conv5: 256 filters, 3×3
    elif model_name == "vgg16":
        return model.features[28]   # Block5 Conv3: 512 filters, 3×3


# ─────────────────────────────────────────────
# GRAD-CAM CORE
# ─────────────────────────────────────────────

def _disable_inplace_relu( model: nn.Module ):
    """
    AlexNet/VGG16 use ReLU( inplace=True ) by default.
    This conflicts with backward hooks — inplace ops corrupt the
    gradient view that the hook needs. Disable before Grad-CAM.
    """
    for module in model.modules():
        if isinstance( module, nn.ReLU ):
            module.inplace = False


def compute_gradcam( model: nn.Module, tensor: torch.Tensor,
                    last_conv: nn.Module ) -> tuple:
    model.eval()

    # Fix: disable all inplace ReLUs before hooking
    _disable_inplace_relu( model )

    # Temporarily unfreeze last conv for gradient flow
    for p in last_conv.parameters():
        p.requires_grad_( True )

    gradients       = []
    activations_out = []

    fh = last_conv.register_forward_hook( 
        lambda m, i, o: activations_out.append( o.detach() ) )
    bh = last_conv.register_full_backward_hook( 
        lambda m, gi, go: gradients.append( go[0].detach() ) )

    output     = model( tensor )
    pred_class = output.argmax( dim=1 ).item()
    confidence = torch.softmax( output, dim=1 )[0, pred_class].item()

    model.zero_grad()
    output[0, pred_class].backward()

    fh.remove()
    bh.remove()

    # Refreeze
    for p in last_conv.parameters():
        p.requires_grad_( False )

    if not gradients:
        raise RuntimeError( 
            "Grad-CAM backward hook did not fire after unfreezing and "
            "disabling inplace ReLU — check model architecture." )

    grads   = gradients[0].squeeze( 0 )
    acts    = activations_out[0].squeeze( 0 )
    weights = grads.mean( dim=( 1, 2 ) )

    cam = sum( w * a for w, a in zip( weights, acts ) )
    cam = torch.clamp( cam, min=0 )
    cam = cam / ( cam.max() + 1e-8 )

    return cam.numpy(), pred_class, confidence


# ─────────────────────────────────────────────
# OVERLAY BUILDER
# ─────────────────────────────────────────────

def make_overlay( img_np: np.ndarray, cam_np: np.ndarray ):
    """Blends Grad-CAM heatmap onto the original image."""
    cam_img     = Image.fromarray( ( cam_np * 255 ).astype( np.uint8 ) )
    cam_resized = np.array( 
        cam_img.resize( ( img_np.shape[1], img_np.shape[0] ), Image.BILINEAR )
     ) / 255.0
    heatmap = cm.jet( cam_resized )[:, :, :3]
    overlay = 0.5 * ( img_np / 255.0 ) + 0.5 * heatmap
    return np.clip( overlay, 0, 1 ), cam_resized


# ─────────────────────────────────────────────
# SAVE ONE PANEL
# ─────────────────────────────────────────────

def save_image( arr: np.ndarray, path: str, title: str, cmap=None ):
    fig, ax = plt.subplots( figsize=( 4, 4 ) )
    ax.imshow( arr, cmap=cmap )
    ax.axis( "off" )
    ax.set_title( title, fontsize=10, fontweight="bold", pad=8 )
    plt.tight_layout()
    plt.savefig( path, dpi=150, bbox_inches="tight" )
    plt.close()
    print( f"  Saved → {path}" )


# ─────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────

def run( image_path: str, model_name: str ):

    # ── Locate exp1 checkpoint ────────────────────────────────
    ckpt_dir = os.path.join( CHECKPOINT_DIR, f"{model_name}_exp1" )
    if not os.path.isdir( ckpt_dir ):
        raise FileNotFoundError( 
            f"Checkpoint folder not found: {ckpt_dir}\n"
            f"Make sure exp1 for {model_name} is complete." )

    ckpt_files = sorted( 
        glob.glob( os.path.join( ckpt_dir, "epoch_*.pt" ) ),
        key=lambda f: int( os.path.basename( f )
                          .replace( "epoch_", "" ).replace( ".pt", "" ) )
     )
    if not ckpt_files:
        raise FileNotFoundError( f"No .pt files in {ckpt_dir}" )

    ckpt_path = ckpt_files[-1]
    print( f"  Checkpoint : {ckpt_path}" )

    # ── Preprocess image ──────────────────────────────────────
    transform = T.Compose( [
        T.Resize( ( IMAGE_SIZE, IMAGE_SIZE ) ),
        T.ToTensor(),
        T.Normalize( mean=IMAGENET_MEAN, std=IMAGENET_STD ),
    ] )
    img_pil     = Image.open( image_path ).convert( "RGB" )
    img_resized = np.array( img_pil.resize( ( IMAGE_SIZE, IMAGE_SIZE ) ) )
    tensor      = transform( img_pil ).unsqueeze( 0 )

    # ── Output directory ──────────────────────────────────────
    img_stem = os.path.splitext( os.path.basename( image_path ) )[0]
    out_dir  = os.path.join( RESULTS_DIR,
                            f"gradcam_before_after_{model_name}_{img_stem}" )
    os.makedirs( out_dir, exist_ok=True )
    print( f"  Output dir : {out_dir}\n" )

    # Original image
    save_image( 
        img_resized,
        os.path.join( out_dir, "00_original.png" ),
        f"Original ( {img_pil.width}×{img_pil.height} → resized {IMAGE_SIZE}×{IMAGE_SIZE} )"
     )

    # ════════════════════════════════════════════
    # BEFORE — pretrained backbone, random head
    # ════════════════════════════════════════════
    print( "  [BEFORE] Building model with random classification head..." )
    model_before     = build_model( model_name, freeze_backbone=True )
    model_before.eval()
    last_conv_before = get_last_conv( model_name, model_before )
    cam_before, pred_before, conf_before = compute_gradcam( 
        model_before, tensor.clone(), last_conv_before )
    overlay_before, heatmap_before = make_overlay( img_resized, cam_before )

    print( f"  [BEFORE] Pred: {CLASS_NAMES[pred_before]} ( {conf_before*100:.1f}% )" )

    save_image( heatmap_before,
               os.path.join( out_dir, "01_before_heatmap.png" ),
               f"BEFORE — Heatmap\nPred: {CLASS_NAMES[pred_before]} ( {conf_before*100:.1f}% )",
               cmap="jet" )
    save_image( overlay_before,
               os.path.join( out_dir, "02_before_overlay.png" ),
               f"BEFORE — Overlay\nPred: {CLASS_NAMES[pred_before]} ( {conf_before*100:.1f}% )" )

    # ════════════════════════════════════════════
    # AFTER — fine-tuned exp1 checkpoint
    # ════════════════════════════════════════════
    print( f"\n  [AFTER ] Loading fine-tuned checkpoint..." )
    model_after = build_model( model_name, freeze_backbone=True )
    ckpt        = torch.load( ckpt_path, map_location="cpu" )
    model_after.load_state_dict( ckpt["model_state_dict"] )
    model_after.eval()
    last_conv_after = get_last_conv( model_name, model_after )
    cam_after, pred_after, conf_after = compute_gradcam( 
        model_after, tensor.clone(), last_conv_after )
    overlay_after, heatmap_after = make_overlay( img_resized, cam_after )

    print( f"  [AFTER ] Pred: {CLASS_NAMES[pred_after]} ( {conf_after*100:.1f}% )" )

    save_image( heatmap_after,
               os.path.join( out_dir, "03_after_heatmap.png" ),
               f"AFTER exp1 — Heatmap\nPred: {CLASS_NAMES[pred_after]} ( {conf_after*100:.1f}% )",
               cmap="jet" )
    save_image( overlay_after,
               os.path.join( out_dir, "04_after_overlay.png" ),
               f"AFTER exp1 — Overlay\nPred: {CLASS_NAMES[pred_after]} ( {conf_after*100:.1f}% )" )

    # ════════════════════════════════════════════
    # SIDE-BY-SIDE SUMMARY ( single figure for report )
    # ════════════════════════════════════════════
    fig, axes = plt.subplots( 2, 3, figsize=( 13, 9 ) )

    panels = [
        ( 0, 0, img_resized,     None,  "Original Image" ),
        ( 0, 1, heatmap_before, "jet",  f"BEFORE — Heatmap\n"
                                        f"Pred: {CLASS_NAMES[pred_before]} ( {conf_before*100:.1f}% )" ),
        ( 0, 2, overlay_before,  None,  f"BEFORE — Overlay\n"
                                        f"Pred: {CLASS_NAMES[pred_before]} ( {conf_before*100:.1f}% )" ),
        ( 1, 0, img_resized,     None,  "Original Image" ),
        ( 1, 1, heatmap_after,  "jet",  f"AFTER exp1 — Heatmap\n"
                                        f"Pred: {CLASS_NAMES[pred_after]} ( {conf_after*100:.1f}% )" ),
        ( 1, 2, overlay_after,   None,  f"AFTER exp1 — Overlay\n"
                                        f"Pred: {CLASS_NAMES[pred_after]} ( {conf_after*100:.1f}% )" ),
    ]

    for row, col, arr, cmap, title in panels:
        ax = axes[row, col]
        ax.imshow( arr, cmap=cmap )
        ax.set_title( title, fontsize=9, pad=6 )
        ax.axis( "off" )

    for row, label in enumerate( ["BEFORE\n( random head )", "AFTER\n( exp1 · 5 epochs )"] ):
        axes[row, 0].set_ylabel( label, fontsize=11, fontweight="bold",
                                labelpad=10, rotation=90, va="center" )

    fig.suptitle( 
        f"Grad-CAM: Before vs After Fine-Tuning  |  {model_name.upper()}\n"
        f"Image: {os.path.basename( image_path )}",
        fontsize=13, fontweight="bold", y=1.01
     )
    plt.tight_layout()
    summary_path = os.path.join( out_dir, "05_summary_side_by_side.png" )
    plt.savefig( summary_path, dpi=150, bbox_inches="tight" )
    plt.close()

    print( f"\n  Saved → {summary_path}" )
    print( f"\n  All files in: {out_dir}/" )
    print( f"  ├── 00_original.png" )
    print( f"  ├── 01_before_heatmap.png" )
    print( f"  ├── 02_before_overlay.png" )
    print( f"  ├── 03_after_heatmap.png" )
    print( f"  ├── 04_after_overlay.png" )
    print( f"  └── 05_summary_side_by_side.png" )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        description="Grad-CAM before vs after exp1 fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter,
     )
    parser.add_argument( "--image", required=True,
                        help="Path to a single traffic sign image" )
    parser.add_argument( "--model", required=True,
                        choices=["alexnet", "vgg16"] )
    args = parser.parse_args()

    if not os.path.isfile( args.image ):
        raise FileNotFoundError( f"Image not found: {args.image}" )

    run( args.image, args.model )