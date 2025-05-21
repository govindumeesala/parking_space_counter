#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def set_style():
    """Set the style for all plots"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_training_comparison():
    """Plot training metrics comparison between Model A and Model B"""
    # Model A data
    model_a_epochs = np.arange(1, 6)
    model_a_train_loss = np.array([0.0321, 0.0012, 0.0007, 0.0008, 0.0016])
    model_a_val_acc = np.array([0.9430, 0.9857, 0.9375, 0.9156, 0.9518])

    # Model B data
    model_b_epochs = np.arange(1, 12)
    model_b_train_loss = np.array([0.0800, 0.0405, 0.0252, 0.0198, 0.0173, 
                                  0.0149, 0.0125, 0.0111, 0.0098, 0.0087, 0.0079])
    model_b_val_loss = np.array([0.0350, 0.0208, 0.0153, 0.0175, 0.0168,
                                0.0159, 0.0172, 0.0180, 0.0195, 0.0207, 0.0165])
    model_b_val_acc = np.array([0.9300, 0.9710, 0.9825, 0.9802, 0.9818,
                               0.9850, 0.9842, 0.9830, 0.9820, 0.9810, 0.9852])

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training Loss Comparison
    ax1.plot(model_a_epochs, model_a_train_loss, 'o-', label='Model A (No Augmentation)', linewidth=2)
    ax1.plot(model_b_epochs, model_b_train_loss, 's-', label='Model B (With Augmentation)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy Comparison
    ax2.plot(model_a_epochs, model_a_val_acc * 100, 'o-', label='Model A (No Augmentation)', linewidth=2)
    ax2.plot(model_b_epochs, model_b_val_acc * 100, 's-', label='Model B (With Augmentation)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create separate figure for Model B's detailed metrics
    plt.figure(figsize=(12, 6))
    plt.plot(model_b_epochs, model_b_train_loss, 'o-', label='Training Loss', linewidth=2)
    plt.plot(model_b_epochs, model_b_val_loss, 's-', label='Validation Loss', linewidth=2)
    plt.plot(model_b_epochs, model_b_val_acc, '^-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.title('Model B Training Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add second y-axis for accuracy
    ax3 = plt.gca()
    ax4 = ax3.twinx()
    ax4.set_ylabel('Validation Accuracy')
    plt.tight_layout()
    plt.savefig('model_b_metrics.png', dpi=300, bbox_inches='tight')

def plot_final_comparison():
    """Plot final performance metrics comparison"""
    metrics = ['Accuracy', 'F1-Score']
    model_a_scores = [97.82, 98.00]
    model_b_scores = [99.02, 99.00]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, model_a_scores, width, label='Model A')
    rects2 = ax.bar(x + width/2, model_b_scores, width, label='Model B')

    ax.set_ylabel('Score (%)')
    ax.set_title('Final Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    set_style()
    plot_training_comparison()
    plot_final_comparison()
    print("Plots saved as 'model_comparison.png', 'model_b_metrics.png', and 'final_comparison.png'") 