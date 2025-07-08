import pandas as pd
from tbparse import SummaryReader

log_dirs = {
    "HybridEfficientNetAutoAttViT": "/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/runs/binclass/net-HybridEfficientNetAutoAttViT_traindb-celebdf_face-scale_size-224_seed-0/",
    "EfficientNetB4": "/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/runs/binclass/net-EfficientNetB4_traindb-celebdf_face-scale_size-224_seed-0/"
}

all_scalars = {}

for model_name, log_dir in log_dirs.items():
    try:
        reader = SummaryReader(log_dir)
        df = reader.scalars
        all_scalars[model_name] = df
        print(f"--- Data for {model_name} ---")
        print(df[['step', 'tag', 'value']].to_string())
        print("\n")
        
        # Save to CSV
        output_csv_path = f"{model_name}_scalars.csv"
        df.to_csv(output_csv_path, index=False)
        print(f"Saved scalars for {model_name} to {output_csv_path}")
        
    except Exception as e:
        print(f"Error reading logs for {model_name}: {e}")
