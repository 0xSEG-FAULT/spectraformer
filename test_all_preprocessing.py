# test_all_preprocessing.py
import subprocess
import sys

preprocessing_methods = ["S", "SM", "SA", "SA0", "SAM", "SA0M", "S0M", "0M"]
results = {}

print("Testing all preprocessing methods...")
print("="*60)

for method in preprocessing_methods:
    print(f"\nTesting preprocessing: {method}")
    cmd = [
        "python", "train_with_best_params.py",
        "--csv_path", "Barley.data.csv",
        "--label_column", "Predictor",
        "--wavelength_start_index", "1",
        "--wavelength_end_index", "332",
        "--preprocessing", method,
        "--epochs", "200",
        "--save_model"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Extract final accuracy from output
        for line in result.stdout.split('\n'):
            if "Best test accuracy:" in line:
                acc = float(line.split(":")[-1].strip())
                results[method] = acc
                print(f"  Final accuracy: {acc:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
print("SUMMARY - Best to Worst:")
print("="*60)
for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{method:10s}: {acc:.4f}")
