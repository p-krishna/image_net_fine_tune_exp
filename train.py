# train.py — Training loop with per-epoch checkpointing and auto-resume

import os
import glob
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from config import CHECKPOINT_DIR


def get_checkpoint_dir( model_name: str, exp_id: str ) -> str:
    path = os.path.join( CHECKPOINT_DIR, f"{model_name}_{exp_id}" )
    os.makedirs( path, exist_ok=True )
    return path


def get_latest_checkpoint( ckpt_dir: str ):
    """Returns ( latest_epoch, filepath ) or ( 0, None ) if none found."""
    files = glob.glob( os.path.join( ckpt_dir, "epoch_*.pt" ) )
    if not files:
        return 0, None

    def epoch_num( f ):
        return int( os.path.basename( f ).replace( "epoch_", "" ).replace( ".pt", "" ) )

    files.sort( key=epoch_num )
    return epoch_num( files[-1] ), files[-1]


def save_checkpoint( ckpt_dir: str, epoch: int, model, optimizer, history: dict ):
    path = os.path.join( ckpt_dir, f"epoch_{epoch}.pt" )
    torch.save( {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history":              history,
    }, path )
    # Keep only last 2 checkpoints to save disk space
    files = sorted( 
        glob.glob( os.path.join( ckpt_dir, "epoch_*.pt" ) ),
        key=lambda f: int( os.path.basename( f ).replace( "epoch_", "" ).replace( ".pt", "" ) )
     )
    for old_file in files[:-2]:
        os.remove( old_file )


def train_one_epoch( model, loader, criterion, optimizer, device ):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to( device ), labels.to( device )
        optimizer.zero_grad()
        outputs = model( images )
        loss    = criterion( outputs, labels )
        loss.backward()
        optimizer.step()

        time.sleep( 0.02 )   # 20ms pause — CPU drops to ~60-70% avg utilization

        total_loss += loss.item() * images.size( 0 )
        correct    += ( outputs.argmax( dim=1 ) == labels ).sum().item()
        total      += images.size( 0 )

    return total_loss / total, correct / total


def run_training( model_name: str, exp: dict, model, train_loader, device ) -> dict:
    exp_id     = exp["id"]
    max_epochs = exp["max_epochs"]
    lr         = exp["lr"]

    ckpt_dir  = get_checkpoint_dir( model_name, exp_id )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=lr )

    history     = {"train_loss": [], "train_acc": [], "epoch_times": []}
    start_epoch, ckpt_file = get_latest_checkpoint( ckpt_dir )

    if ckpt_file:
        print( f"  [Resume] Loading checkpoint: {ckpt_file}" )
        ckpt = torch.load( ckpt_file, map_location=device )
        model.load_state_dict( ckpt["model_state_dict"] )
        optimizer.load_state_dict( ckpt["optimizer_state_dict"] )
        history = ckpt.get( "history", history )
        print( f"  [Resume] Continuing from epoch {start_epoch + 1}/{max_epochs}" )
    else:
        print( f"  [Start ] No checkpoint — training from scratch" )

    model.to( device )

    for epoch in range( start_epoch + 1, max_epochs + 1 ):
        t0 = time.time()
        loss, acc = train_one_epoch( model, train_loader, criterion, optimizer, device )
        elapsed   = time.time() - t0

        history["train_loss"].append( loss )
        history["train_acc"].append( acc )
        history["epoch_times"].append( elapsed )

        print( f"  Epoch {epoch:02d}/{max_epochs} | "
              f"Loss: {loss:.4f} | Acc: {acc*100:.2f}% | Time: {elapsed:.1f}s" )

        save_checkpoint( ckpt_dir, epoch, model, optimizer, history )

    return history