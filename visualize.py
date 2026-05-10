# visualize.py — Class distribution, confusion matrix, training curves, results table

import os
import csv
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from config import CLASS_NAMES, RESULTS_DIR


def _ensure_results():
    os.makedirs( RESULTS_DIR, exist_ok=True )


def plot_class_distribution( train_dataset, test_dataset ):
    _ensure_results()
    from collections import Counter

    def count_labels( ds ):
        labels = [int( ds.df.iloc[i]["NewLabel"] ) for i in range( len( ds ) )]
        return [Counter( labels ).get( c, 0 ) for c in range( len( CLASS_NAMES ) )]

    train_counts = count_labels( train_dataset )
    test_counts  = count_labels( test_dataset )
    x = np.arange( len( CLASS_NAMES ) )
    w = 0.35

    fig, ax = plt.subplots( figsize=( 8, 5 ) )
    b1 = ax.bar( x - w/2, train_counts, w, label="Train", color="#2176AE", alpha=0.85 )
    b2 = ax.bar( x + w/2, test_counts,  w, label="Test",  color="#F7882F", alpha=0.85 )
    ax.set_title( "Class Distribution: Train vs Test", fontsize=13, fontweight="bold" )
    ax.set_xlabel( "Shape Category" )
    ax.set_ylabel( "Number of Images" )
    ax.set_xticks( x ); ax.set_xticklabels( CLASS_NAMES )
    ax.legend()
    ax.yaxis.set_major_formatter( ticker.FuncFormatter( lambda v, _: f"{int( v ):,}" ) )
    ax.bar_label( b1, padding=3, fmt="%d" ); ax.bar_label( b2, padding=3, fmt="%d" )
    ax.grid( axis="y", linestyle="--", alpha=0.4 )
    plt.tight_layout()
    path = os.path.join( RESULTS_DIR, "class_distribution.png" )
    plt.savefig( path, dpi=150 ); plt.close()
    print( f"  [Plot] Saved: {path}" )


def plot_confusion_matrix( cm, model_name, exp_id, accuracy ):
    _ensure_results()
    fig, ax = plt.subplots( figsize=( 6, 5 ) )
    sns.heatmap( cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="gray", ax=ax )
    ax.set_title( f"Confusion Matrix — {model_name.upper()} {exp_id}\n"
                 f"Test Accuracy: {accuracy*100:.2f}%", fontsize=11, fontweight="bold" )
    ax.set_xlabel( "Predicted Label" ); ax.set_ylabel( "True Label" )
    plt.tight_layout()
    path = os.path.join( RESULTS_DIR, f"confusion_{model_name}_{exp_id}.png" )
    plt.savefig( path, dpi=150 ); plt.close()
    print( f"  [Plot] Saved: {path}" )


def plot_training_curves( history, model_name, exp_id ):
    _ensure_results()
    epochs   = range( 1, len( history["train_loss"] ) + 1 )
    fig, ( ax1, ax2 ) = plt.subplots( 1, 2, figsize=( 11, 4 ) )

    ax1.plot( epochs, history["train_loss"], marker="o", color="#2176AE", linewidth=2 )
    ax1.set_title( f"Training Loss — {model_name.upper()} {exp_id}" )
    ax1.set_xlabel( "Epoch" ); ax1.set_ylabel( "Loss" )
    ax1.grid( linestyle="--", alpha=0.4 )

    ax2.plot( epochs, [a*100 for a in history["train_acc"]], marker="o", color="#F7882F", linewidth=2 )
    ax2.set_title( f"Training Accuracy — {model_name.upper()} {exp_id}" )
    ax2.set_xlabel( "Epoch" ); ax2.set_ylabel( "Accuracy ( % )" ); ax2.set_ylim( 0, 100 )
    ax2.grid( linestyle="--", alpha=0.4 )

    plt.suptitle( f"{model_name.upper()} | {exp_id}", fontsize=12, fontweight="bold" )
    plt.tight_layout()
    path = os.path.join( RESULTS_DIR, f"training_curves_{model_name}_{exp_id}.png" )
    plt.savefig( path, dpi=150 ); plt.close()
    print( f"  [Plot] Saved: {path}" )


def save_results_table( records: list ):
    _ensure_results()
    path = os.path.join( RESULTS_DIR, "results_table.csv" )
    fieldnames = ["model", "exp_id", "lr", "batch_size", "max_epochs",
                  "augment", "test_accuracy_%", "total_time_min"]

    with open( path, "w", newline="" ) as f:
        writer = csv.DictWriter( f, fieldnames=fieldnames )
        writer.writeheader()
        for r in records:
            writer.writerow( {
                "model":           r["model"],
                "exp_id":          r["exp_id"],
                "lr":              r["lr"],
                "batch_size":      r["batch_size"],
                "max_epochs":      r["max_epochs"],
                "augment":         r["augment"],
                "test_accuracy_%": f"{r['test_accuracy']*100:.2f}",
                "total_time_min":  f"{r['total_time_sec']/60:.2f}",
            } )

    print( f"\n  [Results] Table saved: {path}" )
    print( f"\n{'─'*75}" )
    print( f"{'Model':<18} {'Exp':>5} {'LR':>8} {'BS':>4} {'Ep':>4} "
          f"{'Aug':>5} {'Acc %':>7} {'Time( min )':>10}" )
    print( f"{'─'*75}" )
    for r in records:
        print( f"{r['model']:<18} {r['exp_id']:>5} {r['lr']:>8} "
              f"{r['batch_size']:>4} {r['max_epochs']:>4} "
              f"{'Yes' if r['augment'] else 'No':>5} "
              f"{r['test_accuracy']*100:>6.2f}% "
              f"{r['total_time_sec']/60:>9.2f}m" )
    print( f"{'─'*75}\n" )