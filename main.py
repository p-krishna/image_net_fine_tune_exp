#!/usr/bin/env python3
"""
main.py — Entry point: sequential auto-run of all ( model x experiment ) pairs

Project        : Transfer Learning for Traffic Sign Recognition
Task           : Implement transfer learning pipeline for GTSRB dataset using PyTorch
Python Version : 3.14.2
Dependencies   : torch, torchvision, matplotlib, pandas, tqdm

Description    :
    * Loads pretrained models (AlexNet, VGG16, ResNet50, EfficientNet-B0)
    * Freezes backbone layers, replaces head for 43 classes
    * Trains on GTSRB with various hyperparameters and data augmentation
    * Evaluates on test set, generates confusion matrix and training curves

Usage          :
    * Download GTSRB dataset and place in 'data' folder
    * Run script: python preprocess.py
    * Run script: python sampler.py 3200
    * Run script: python main.py
    * Checkpoints and images saved in 'results' folder

Coding conventions :
    * PEP 8 standards
    * space around parantheses, operators, and after commas
"""

import os
import time
import torch
from sklearn.metrics import classification_report

from config import MODELS, EXPERIMENTS, CHECKPOINT_DIR, RESULTS_DIR, CLASS_NAMES
from dataset import get_dataloaders
from model   import build_model, count_trainable_params
from train   import run_training, get_checkpoint_dir, get_latest_checkpoint
from evaluate import evaluate_model
from visualize import ( 
    plot_class_distribution, plot_confusion_matrix,
    plot_training_curves, save_results_table,
 )

torch.set_num_threads( 4 )        # use only 4 of 8 threads for tensor ops
torch.set_num_interop_threads( 2 )

def is_complete( model_name, exp ):
    ckpt_dir = get_checkpoint_dir( model_name, exp["id"] )
    latest, _ = get_latest_checkpoint( ckpt_dir )
    return latest >= exp["max_epochs"]


def format_time( s ):
    m, s = divmod( int( s ), 60 )
    h, m = divmod( m, 60 )
    return f"{h}h {m}m {s}s" if h else ( f"{m}m {s}s" if m else f"{s}s" )


def print_header( text ):
    print( f"\n{'═'*65}\n  {text}\n{'═'*65}" )


def print_status_summary():
    print_header( "GTSRB Transfer Learning — Experiment Status" )
    done = 0
    for model_name in MODELS:
        for exp in EXPERIMENTS:
            ckpt_dir = get_checkpoint_dir( model_name, exp["id"] )
            latest, _ = get_latest_checkpoint( ckpt_dir )
            max_ep = exp["max_epochs"]
            if latest >= max_ep:
                status = f"✓ Complete ( {max_ep}/{max_ep} epochs )"; done += 1
            elif latest > 0:
                status = f"⟳ In progress ( {latest}/{max_ep} epochs )"
            else:
                status = "○ Not started"
            print( f"  {model_name:<18} {exp['id']}  →  {status}" )
    print( f"\n  Progress: {done}/{len( MODELS )*len( EXPERIMENTS )} complete\n" )


def main():
    os.makedirs( CHECKPOINT_DIR, exist_ok=True )
    os.makedirs( RESULTS_DIR,    exist_ok=True )
    device = torch.device( "cpu" )
    print( f"\n  Device: {device}" )

    print_status_summary()

    # Class distribution ( once )
    dist_plot = os.path.join( RESULTS_DIR, "class_distribution.png" )
    if not os.path.exists( dist_plot ):
        print( "  [Setup] Generating class distribution plot..." )
        _, _, train_ds, test_ds = get_dataloaders( batch_size=32, augment=False )
        plot_class_distribution( train_ds, test_ds )
    else:
        print( "  [Setup] Class distribution plot already exists — skipping" )

    all_records = []

    for model_name in MODELS:
        for exp in EXPERIMENTS:
            exp_id = exp["id"]
            label  = ( f"{model_name.upper()} | {exp_id} | "
                      f"lr={exp['lr']} bs={exp['batch_size']} "
                      f"ep={exp['max_epochs']} aug={'Yes' if exp['augment'] else 'No'}" )
            print_header( label )

            if is_complete( model_name, exp ):
                print( "  [Skip] Already complete — loading for evaluation" )
                ckpt_dir = get_checkpoint_dir( model_name, exp_id )
                _, ckpt_file = get_latest_checkpoint( ckpt_dir )
                ckpt    = torch.load( ckpt_file, map_location=device )
                history = ckpt.get( "history", {} )
                total_time = sum( history.get( "epoch_times", [0] ) )

                model = build_model( model_name )
                model.load_state_dict( ckpt["model_state_dict"] )
                _, test_loader, _, _ = get_dataloaders( exp["batch_size"], augment=False )
                results = evaluate_model( model, test_loader, device )
            else:
                model = build_model( model_name )
                print( f"  [Model] Trainable params: {count_trainable_params( model ):,}" )
                train_loader, test_loader, _, _ = get_dataloaders( 
                    exp["batch_size"], exp["augment"] )

                t0      = time.time()
                history = run_training( model_name, exp, model, train_loader, device )
                total_time = time.time() - t0
                print( f"\n  [Done] Training complete in {format_time( total_time )}" )

                print( "  [Eval] Running on test set..." )
                results = evaluate_model( model, test_loader, device )
                print( f"  [Result] Test Accuracy: {results['accuracy']*100:.2f}%" )
                print( classification_report( 
                    results["all_labels"], results["all_preds"],
                    target_names=CLASS_NAMES, zero_division=0 ) )

            plot_confusion_matrix( results["confusion_matrix"], model_name, exp_id,
                                  results["accuracy"] )
            plot_training_curves( history, model_name, exp_id )

            all_records.append( {
                "model": model_name, "exp_id": exp_id,
                "lr": exp["lr"], "batch_size": exp["batch_size"],
                "max_epochs": exp["max_epochs"], "augment": exp["augment"],
                "test_accuracy": results["accuracy"],
                "total_time_sec": total_time,
            } )

    print_header( "FINAL RESULTS SUMMARY" )
    save_results_table( all_records )
    print( "  All experiments complete.\n" )


if __name__ == "__main__":
    main()