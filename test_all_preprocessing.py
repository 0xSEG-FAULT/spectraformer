#!/usr/bin/env python3
# test_all_preprocessing_simple.py

import os
import json
from datetime import datetime

preprocessing_methods = ["S", "SM", "SA", "SA0", "SAM", "SA0M", "S0M", "0M"]
results = {}

print("="*70)
print("Testing all preprocessing methods for Barley dataset")
print("="*70)

for i, method in enumerate(preprocessing_methods, 1):
    print(f"\n[{i}/8] Testing preprocessing: {method}")
    print("-"*70)
    
    # Run training and capture output to a file
    log_file = f"training_{method}.log"
    cmd = (
        f"python train_with_best_params.py "
        f"--csv_path Barley.data.csv "
        f"--label_column Predictor "
        f"--wavelength_start_index 1 "
        f"--wavelength_end_index 332 "
        f"--preprocessing {method} "
        f"--epochs 200 "
        f"--save_model > {log_file} 2>&1"
    )
    
    os.system(cmd)
    
    # Read the log file and extract accuracy
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Find the last line with "Best test accuracy:"
        lines = content.split('\n')
        accuracy = None
        for line in reversed(lines):
            if "Best test accuracy:" in line:
                # Extract the number after "Best test accuracy: "
                parts = line.split("Best test accuracy:")
                if len(parts) > 1:
                    acc_str = parts[1].strip()
                    try:
                        accuracy = float(acc_str)
                        break
                    except:
                        pass
        
        if accuracy is not None:
            results[method] = accuracy
            print(f"✓ {method}: {accuracy:.4f}")
        else:
            print(f"✗ {method}: Could not extract accuracy from log")
            # Print last few lines for debugging
            print("Last 5 lines of log:")
            for line in lines[-5:]:
                if line.strip():
                    print(f"  {line}")
    
    except Exception as e:
        print(f"✗ {method}: Error - {e}")

print("\n" + "="*70)
print("SUMMARY - Best to Worst:")
print("="*70)

if results:
    # Sort by accuracy (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for method, acc in sorted_results:
        bar_length = int(acc * 50)  # Visual bar
        bar = "█" * bar_length
        print(f"{method:8s} : {acc:.4f}  {bar}")
    
    print("\n" + "="*70)
    best_method, best_acc = sorted_results[0]
    print(f"🏆 BEST PREPROCESSING: {best_method} with accuracy {best_acc:.4f}")
    print("="*70)
    
    # Save results to JSON
    with open("preprocessing_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": dict(sorted_results)
        }, f, indent=2)
    print("✓ Results saved to preprocessing_results.json")
else:
    print("❌ No results captured. Check if train_with_best_params.py is working correctly.")