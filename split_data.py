
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_path, output_dir):
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Total rows: {len(df)}")
    
    # Split 80% train, 20% temp (val + test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Split 20% temp into 50% val, 50% test (resulting in 10% val, 10% test)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("Data split completed:")
    print(f"- Train: {len(train_df)} rows")
    print(f"- Val: {len(val_df)} rows")
    print(f"- Test: {len(test_df)} rows")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    raw_file = "data/raw/train.csv"
    processed_dir = "data/processed"
    split_data(raw_file, processed_dir)
