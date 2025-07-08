import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_curves(model_name, csv_path):
    df = pd.read_csv(csv_path)

    # Plot Loss Curves
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df[df['tag'] == 'train/loss'], x='step', y='value', label='Train Loss')
    sns.lineplot(data=df[df['tag'] == 'val/loss'], x='step', y='value', label='Validation Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_loss_curves.png')
    plt.show()

    # Plot ROC AUC Curves
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df[df['tag'] == 'train/roc_auc'], x='step', y='value', label='Train ROC AUC')
    sns.lineplot(data=df[df['tag'] == 'val/roc_auc'], x='step', y='value', label='Validation ROC AUC')
    plt.title(f'{model_name} - ROC AUC Curves')
    plt.xlabel('Iteration')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc_auc_curves.png')
    plt.show()

if __name__ == '__main__':
    plot_curves('HybridEfficientNetAutoAttViT', 'HybridEfficientNetAutoAttViT_scalars.csv')
    plot_curves('EfficientNetB4', 'EfficientNetB4_scalars.csv')