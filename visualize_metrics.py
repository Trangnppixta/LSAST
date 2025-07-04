import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

def visualize_metrics(metrics_csv, output_dir='outputs'):
    metrics_df = pd.read_csv(metrics_csv)
    metrics_df.dropna(subset=['global_step'], inplace=True)
    metrics_df['global_step'] = metrics_df['global_step'].astype(int)

    plt.plot(metrics_df['global_step'], metrics_df['train/loss_step'], label='train/loss', color='blue')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))
    
    plt.clf()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualize training metrics from CSV file.')
    parser.add_argument('--metrics_csv', type=str, required=True, help='Path to the metrics CSV file.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the output plots.')
    args = parser.parse_args()  
    visualize_metrics(args.metrics_csv, args.output_dir)
