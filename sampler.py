"""
sampler.py — Stratified sampling utility for GTSRB dataset

USAGE ( from your project folder ):
  python sampler.py --sample 0.05   → 5% of data, proportions preserved
  python sampler.py --sample 200    → exactly 200 images total ( train ), proportions preserved
  python sampler.py --reset         → delete sampled CSVs, revert to full dataset

After running, config.py TRAIN_CSV / TEST_CSV automatically point to
sampled files if they exist — no changes needed in any other file.
To go back to full dataset: python sampler.py --reset
"""

import os
import argparse
import pandas as pd
from config import DATASET_ROOT, TRAIN_CSV, TEST_CSV, LABEL_MAP, CLASS_NAMES

# Sampled CSV paths ( written next to the originals )
SAMPLED_TRAIN_CSV = os.path.join( DATASET_ROOT, "Train_sampled.csv" )
SAMPLED_TEST_CSV  = os.path.join( DATASET_ROOT, "Test_sampled.csv" )


# ─────────────────────────────────────────────────────────────
# CORE STRATIFIED SAMPLER
# ─────────────────────────────────────────────────────────────

def stratified_sample( df: pd.DataFrame, sample_spec ) -> pd.DataFrame:
    """
    Performs stratified sampling on a DataFrame that already has a 'NewLabel' column.

    sample_spec:
      float in ( 0, 1 )  → fraction of total rows  ( e.g. 0.1 = 10% )
      int   > 1        → absolute number of total rows to keep

    Class proportions are preserved exactly ( floored per class, remainder
    distributed to largest classes to hit the target count ).
    """
    class_groups = df.groupby( "NewLabel" )
    total_rows   = len( df )

    # Resolve target total count
    if isinstance( sample_spec, float ):
        if not ( 0.0 < sample_spec < 1.0 ):
            raise ValueError( "Float sample_spec must be between 0 and 1 ( exclusive )" )
        target_total = max( len( df["NewLabel"].unique() ), int( total_rows * sample_spec ) )
    elif isinstance( sample_spec, int ):
        if sample_spec < len( df["NewLabel"].unique() ):
            raise ValueError( "Integer sample_spec must be >= number of classes" )
        target_total = sample_spec
    else:
        raise TypeError( "sample_spec must be float ( fraction ) or int ( count )" )

    # Compute per-class proportions and initial floored counts
    class_counts   = class_groups.size()
    proportions    = class_counts / total_rows
    raw_counts     = proportions * target_total
    floored_counts = raw_counts.astype( int )

    # Distribute remainder to classes with largest fractional parts
    remainder    = target_total - floored_counts.sum()
    frac_parts   = raw_counts - floored_counts
    top_classes  = frac_parts.nlargest( remainder ).index
    final_counts = floored_counts.copy()
    final_counts[top_classes] += 1

    # Clamp to available rows per class
    final_counts = final_counts.clip( upper=class_counts )

    # Sample each class
    sampled_parts = []
    for label, count in final_counts.items():
        group   = class_groups.get_group( label )
        sampled = group.sample( n=int( count ), random_state=42 )
        sampled_parts.append( sampled )

    return pd.concat( sampled_parts ).reset_index( drop=True )


# ─────────────────────────────────────────────────────────────
# LOAD + REMAP LABELS
# ─────────────────────────────────────────────────────────────

def load_and_remap( csv_path: str ) -> pd.DataFrame:
    df = pd.read_csv( csv_path )
    df = df[df["ClassId"].isin( LABEL_MAP )].reset_index( drop=True )
    df["NewLabel"] = df["ClassId"].map( LABEL_MAP )
    return df


# ─────────────────────────────────────────────────────────────
# PRINT DISTRIBUTION TABLE
# ─────────────────────────────────────────────────────────────

def print_distribution( df_full: pd.DataFrame, df_sampled: pd.DataFrame, split_name: str ):
    print( f"\n  {split_name} distribution:" )
    print( f"  {'Class':<14} {'Full':>8} {'Full%':>7} {'Sampled':>9} {'Sampled%':>10}" )
    print( f"  {'─'*52}" )
    total_full    = len( df_full )
    total_sampled = len( df_sampled )
    for label, name in enumerate( CLASS_NAMES ):
        n_full    = ( df_full["NewLabel"]    == label ).sum()
        n_sampled = ( df_sampled["NewLabel"] == label ).sum()
        print( f"  {name:<14} {n_full:>8,} {n_full/total_full*100:>6.1f}%"
              f" {n_sampled:>9,} {n_sampled/total_sampled*100:>9.1f}%" )
    print( f"  {'─'*52}" )
    print( f"  {'TOTAL':<14} {total_full:>8,} {'100.0%':>7} {total_sampled:>9,} {'100.0%':>10}" )


# ─────────────────────────────────────────────────────────────
# WRITE SAMPLED CSVs + PATCH config.py AT RUNTIME
# ─────────────────────────────────────────────────────────────

def write_sampled_csvs( df_train_sampled, df_test_sampled ):
    df_train_sampled.to_csv( SAMPLED_TRAIN_CSV, index=False )
    df_test_sampled.to_csv( SAMPLED_TEST_CSV,   index=False )
    print( f"\n  Sampled Train CSV → {SAMPLED_TRAIN_CSV}" )
    print( f"  Sampled Test  CSV → {SAMPLED_TEST_CSV}" )


# ─────────────────────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────────────────────

def reset():
    deleted = []
    for path in [SAMPLED_TRAIN_CSV, SAMPLED_TEST_CSV]:
        if os.path.exists( path ):
            os.remove( path )
            deleted.append( path )
    if deleted:
        print( f"  Deleted: {', '.join( deleted )}" )
    else:
        print( "  No sampled CSVs found — nothing to delete" )
    print( "  Reset complete. config.py now uses full dataset." )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_sample_spec( value: str ):
    try:
        f = float( value )
        if f < 1.0:
            return f
        return int( f )
    except ValueError:
        raise argparse.ArgumentTypeError( 
            f"Invalid sample spec '{value}'. Use a fraction like 0.1 or integer like 200." )


def main():
    parser = argparse.ArgumentParser( 
        description="Stratified sampler for GTSRB dataset.\n"
                    "Preserves class proportions in both train and test splits.",
        formatter_class=argparse.RawTextHelpFormatter,
     )
    group = parser.add_mutually_exclusive_group( required=True )
    group.add_argument( 
        "--sample", type=parse_sample_spec, metavar="SPEC",
        help=( 
            "Sampling spec:\n"
            "  Float ( 0–1 ): fraction of dataset  e.g. --sample 0.05\n"
            "  Integer    : absolute count        e.g. --sample 500"
         ),
     )
    group.add_argument( 
        "--reset", action="store_true",
        help="Delete sampled CSVs and revert config.py to full dataset",
     )
    args = parser.parse_args()

    if args.reset:
        reset()
        return

    print( f"\n  Loading full CSVs from: {DATASET_ROOT}" )
    df_train_full = load_and_remap( TRAIN_CSV )
    df_test_full  = load_and_remap( TEST_CSV )
    print( f"  Full train size : {len( df_train_full ):,}" )
    print( f"  Full test  size : {len( df_test_full ):,}" )

    spec = args.sample

    if isinstance( spec, int ):
        train_fraction = spec / len( df_train_full )
        test_spec      = max( 3, int( len( df_test_full ) * train_fraction ) )
        print( f"\n  Sample spec     : {spec} images ( train ) → {train_fraction*100:.2f}% fraction" )
        print( f"  Test sample     : {test_spec} images ( same fraction )" )
    else:
        train_fraction = spec
        test_spec      = spec
        print( f"\n  Sample spec     : {spec*100:.1f}% of each split" )

    df_train_sampled = stratified_sample( df_train_full, spec )
    df_test_sampled  = stratified_sample( df_test_full,  test_spec )

    print_distribution( df_train_full, df_train_sampled, "TRAIN" )
    print_distribution( df_test_full,  df_test_sampled,  "TEST" )

    write_sampled_csvs( df_train_sampled, df_test_sampled )

    print( f"\n  Done. Run  python main.py  to train on the sampled dataset." )
    print( f"  When ready for full run: python sampler.py --reset\n" )


if __name__ == "__main__":
    main()