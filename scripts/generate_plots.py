#!/usr/bin/env python3
"""
Entry point script for generating plots.
Supports both All_Beauty and Handmade_Products datasets.
"""
import os
import sys
import argparse
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for datasets")
    parser.add_argument(
        "--dataset",
        choices=["all_beauty", "handmade"],
        default="handmade",
        help="Dataset to use for plotting"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "all_beauty":
        # Import and run the plots_all_beauty module
        spec = importlib.util.spec_from_file_location(
            "plots_all_beauty",
            os.path.join(PROJECT_ROOT, "src", "plotting", "plots_all_beauty.py")
        )
        plots_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plots_module)
    else:
        # Import and run the plots_handmade module
        spec = importlib.util.spec_from_file_location(
            "plots_handmade",
            os.path.join(PROJECT_ROOT, "src", "plotting", "plots_handmade.py")
        )
        plots_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plots_module)

