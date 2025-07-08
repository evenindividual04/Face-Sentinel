
import torch
import sys

def read_loss_from_checkpoint(path):
    """Loads a checkpoint and prints the saved loss values."""
    try:
        # Load the checkpoint on the CPU
        state = torch.load(path, map_location='cpu')
        
        # Check for keys that store loss values
        train_loss = state.get('train_loss', 'Not found')
        val_loss = state.get('val_loss', 'Not found')
        iteration = state.get('iteration', 'Not found')
        
        print(f"Checkpoint: {path}")
        print(f"  - Iteration: {iteration}")
        print(f"  - Saved Train Loss: {train_loss}")
        print(f"  - Saved Validation Loss: {val_loss}")
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_checkpoint.py <path_to_checkpoint.pth>")
    else:
        checkpoint_path = sys.argv[1]
        read_loss_from_checkpoint(checkpoint_path)
