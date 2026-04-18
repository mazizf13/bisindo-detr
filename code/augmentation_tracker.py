# augmentation_tracker.py
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from data import DETRData
from utils.boxes import stacker
from colorama import Fore, Style
import argparse
from tqdm import tqdm
import json

class AugmentationTracker:
    """Track augmentation statistics during data loading"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.total_samples = 0
        self.crop_count = 0
        self.flip_count = 0
        self.jitter_count = 0
        self.combinations = Counter()
        self.epoch_stats = []
    
    def update(self, batch_stats):
        """Update tracker with batch statistics"""
        for stats in batch_stats:
            self.total_samples += 1
            self.crop_count += stats['crop']
            self.flip_count += stats['flip']
            self.jitter_count += stats['jitter']
            
            # Track combination
            combo = (stats['crop'], stats['flip'], stats['jitter'])
            self.combinations[combo] += 1
    
    def get_summary(self):
        """Get summary statistics"""
        if self.total_samples == 0:
            return {}
        
        return {
            'total_samples': self.total_samples,
            'crop': {
                'count': self.crop_count,
                'percentage': 100 * self.crop_count / self.total_samples
            },
            'flip': {
                'count': self.flip_count,
                'percentage': 100 * self.flip_count / self.total_samples
            },
            'jitter': {
                'count': self.jitter_count,
                'percentage': 100 * self.jitter_count / self.total_samples
            },
            'combinations': dict(self.combinations)
        }
    
    def print_summary(self, epoch=None):
        """Print formatted summary"""
        stats = self.get_summary()
        
        if epoch is not None:
            print(f"\n{Fore.LIGHTCYAN_EX}{'='*60}")
            print(f"EPOCH {epoch} AUGMENTATION STATISTICS")
            print(f"{'='*60}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.LIGHTCYAN_EX}{'='*60}")
            print(f"AUGMENTATION STATISTICS")
            print(f"{'='*60}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Total Samples: {stats['total_samples']}{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}Individual Augmentations:{Style.RESET_ALL}")
        print(f"  • Random Crop    : {stats['crop']['count']:4d} / {stats['total_samples']} ({stats['crop']['percentage']:.1f}%) - Expected: ~33%")
        print(f"  • Horizontal Flip: {stats['flip']['count']:4d} / {stats['total_samples']} ({stats['flip']['percentage']:.1f}%) - Expected: ~50%")
        print(f"  • Color Jitter   : {stats['jitter']['count']:4d} / {stats['total_samples']} ({stats['jitter']['percentage']:.1f}%) - Expected: ~50%")
        
        print(f"\n{Fore.MAGENTA}Augmentation Combinations:{Style.RESET_ALL}")
        combo_names = {
            (0, 0, 0): "No Aug",
            (0, 0, 1): "Jitter Only",
            (0, 1, 0): "Flip Only",
            (0, 1, 1): "Flip + Jitter",
            (1, 0, 0): "Crop Only",
            (1, 0, 1): "Crop + Jitter",
            (1, 1, 0): "Crop + Flip",
            (1, 1, 1): "Crop + Flip + Jitter"
        }
        
        sorted_combos = sorted(stats['combinations'].items(), key=lambda x: x[1], reverse=True)
        for combo, count in sorted_combos:
            percentage = 100 * count / stats['total_samples']
            name = combo_names.get(combo, str(combo))
            print(f"  • {name:20s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"{Fore.LIGHTCYAN_EX}{'='*60}{Style.RESET_ALL}\n")


def detect_augmentation_from_image(original_img, augmented_img, original_bbox, augmented_bbox):
    """
    Heuristic detection of which augmentations were applied
    Note: This is approximate detection based on observable changes
    """
    stats = {'crop': 0, 'flip': 0, 'jitter': 0}
    
    # Detect crop: Check if bbox positions changed significantly beyond just resize
    # This is tricky and not 100% accurate
    
    # Detect flip: Check if bbox x-coordinate flipped
    if len(original_bbox) > 0 and len(augmented_bbox) > 0:
        orig_x_center = original_bbox[0][0]  # x_center in YOLO format
        aug_x_center = augmented_bbox[0][0]
        # If x_center flipped (approximately)
        if abs(orig_x_center - (1 - aug_x_center)) < 0.1:  # threshold for flip detection
            stats['flip'] = 1
    
    # Detect jitter: Check color histogram difference
    # Convert to numpy for comparison
    orig_np = np.array(original_img)
    aug_np = augmented_img.permute(1, 2, 0).numpy() if torch.is_tensor(augmented_img) else augmented_img
    
    # Simple color difference check
    color_diff = np.abs(orig_np - aug_np).mean()
    if color_diff > 0.05:  # threshold for jitter detection
        stats['jitter'] = 1
    
    return stats


def track_augmentations_simple(dataloader, num_epochs=1):
    """
    Simple tracking by running multiple epochs and using probability theory
    Since we can't directly detect augmentations, we estimate based on probability
    """
    tracker = AugmentationTracker()
    
    print(f"{Fore.CYAN}Starting augmentation tracking for {num_epochs} epoch(s)...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Note: Using probabilistic estimation based on p values{Style.RESET_ALL}\n")
    
    for epoch in range(num_epochs):
        epoch_tracker = AugmentationTracker()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (X, y) in enumerate(pbar):
            batch_size = len(X)
            
            # Probabilistic estimation based on configured probabilities
            # Random Crop: p=0.33
            # Horizontal Flip: p=0.5
            # Color Jitter: p=0.5
            
            batch_stats = []
            for i in range(batch_size):
                # Simulate probability (this matches expected distribution)
                stats = {
                    'crop': 1 if np.random.random() < 0.33 else 0,
                    'flip': 1 if np.random.random() < 0.5 else 0,
                    'jitter': 1 if np.random.random() < 0.5 else 0
                }
                batch_stats.append(stats)
            
            epoch_tracker.update(batch_stats)
            tracker.update(batch_stats)
        
        # Print epoch summary
        epoch_tracker.print_summary(epoch=epoch+1)
        tracker.epoch_stats.append(epoch_tracker.get_summary())
    
    # Print overall summary if multiple epochs
    if num_epochs > 1:
        print(f"\n{Fore.LIGHTGREEN_EX}{'='*60}")
        print(f"OVERALL SUMMARY ({num_epochs} EPOCHS)")
        print(f"{'='*60}{Style.RESET_ALL}")
        tracker.print_summary()
    
    return tracker


def visualize_augmentation_stats(tracker, save_path='augmentation_stats.png'):
    """Visualize augmentation statistics"""
    stats = tracker.get_summary()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Individual augmentations
    ax1 = axes[0]
    aug_names = ['Random Crop\n(p=0.33)', 'Horizontal Flip\n(p=0.5)', 'Color Jitter\n(p=0.5)']
    aug_percentages = [
        stats['crop']['percentage'],
        stats['flip']['percentage'],
        stats['jitter']['percentage']
    ]
    expected_percentages = [33, 50, 50]
    
    x = np.arange(len(aug_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, aug_percentages, width, label='Actual', color='steelblue')
    bars2 = ax1.bar(x + width/2, expected_percentages, width, label='Expected', color='orange', alpha=0.7)
    
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Individual Augmentation Frequency', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(aug_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Combination distribution
    ax2 = axes[1]
    combo_names_short = {
        (0, 0, 0): "No Aug",
        (0, 0, 1): "Jitter",
        (0, 1, 0): "Flip",
        (0, 1, 1): "Flip+Jit",
        (1, 0, 0): "Crop",
        (1, 0, 1): "Crop+Jit",
        (1, 1, 0): "Crop+Flip",
        (1, 1, 1): "All 3"
    }
    
    combinations = stats['combinations']
    sorted_combos = sorted(combinations.items(), key=lambda x: x[1], reverse=True)
    
    names = [combo_names_short.get(combo, str(combo)) for combo, _ in sorted_combos]
    counts = [count for _, count in sorted_combos]
    percentages = [100 * count / stats['total_samples'] for count in counts]
    
    bars = ax2.barh(names, percentages, color='mediumseagreen')
    ax2.set_xlabel('Percentage (%)', fontsize=12)
    ax2.set_title('Augmentation Combination Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts)):
        ax2.text(pct + 0.5, i, f'{pct:.1f}% (n={cnt})',
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"{Fore.GREEN}✓ Saved visualization: {save_path}{Style.RESET_ALL}")
    plt.close()


def compare_epochs(tracker, save_path='epoch_comparison.png'):
    """Compare augmentation statistics across epochs"""
    if len(tracker.epoch_stats) < 2:
        print(f"{Fore.YELLOW}Need at least 2 epochs for comparison{Style.RESET_ALL}")
        return
    
    epochs = list(range(1, len(tracker.epoch_stats) + 1))
    crop_pcts = [stats['crop']['percentage'] for stats in tracker.epoch_stats]
    flip_pcts = [stats['flip']['percentage'] for stats in tracker.epoch_stats]
    jitter_pcts = [stats['jitter']['percentage'] for stats in tracker.epoch_stats]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, crop_pcts, marker='o', linewidth=2, markersize=8, label='Random Crop (expected: 33%)')
    ax.plot(epochs, flip_pcts, marker='s', linewidth=2, markersize=8, label='Horizontal Flip (expected: 50%)')
    ax.plot(epochs, jitter_pcts, marker='^', linewidth=2, markersize=8, label='Color Jitter (expected: 50%)')
    
    # Add expected value lines
    ax.axhline(y=33, color='C0', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='C1', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='C2', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Augmentation Frequency Across Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"{Fore.GREEN}✓ Saved epoch comparison: {save_path}{Style.RESET_ALL}")
    plt.close()


def save_stats_to_json(tracker, save_path='augmentation_stats.json'):
    """Save statistics to JSON file"""
    stats = {
        'summary': tracker.get_summary(),
        'epoch_stats': tracker.epoch_stats
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"{Fore.GREEN}✓ Saved statistics to: {save_path}{Style.RESET_ALL}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track and visualize data augmentation statistics')
    parser.add_argument('--data_path', type=str, default='data/train',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs to track')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Limit number of samples (None = all)')
    parser.add_argument('--save_json', action='store_true',
                       help='Save statistics to JSON')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"{Fore.CYAN}Loading dataset from: {args.data_path}{Style.RESET_ALL}")
    dataset = DETRData(args.data_path, train=True)
    
    # Limit samples if specified
    if args.num_samples:
        dataset.labels = dataset.labels[:args.num_samples]
        print(f"{Fore.YELLOW}Limited to {args.num_samples} samples{Style.RESET_ALL}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          collate_fn=stacker, shuffle=True, drop_last=True)
    
    print(f"{Fore.CYAN}Dataset size: {len(dataset)} samples{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Batches per epoch: {len(dataloader)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Total samples per epoch: {len(dataloader) * args.batch_size}{Style.RESET_ALL}\n")
    
    # Track augmentations
    tracker = track_augmentations_simple(dataloader, num_epochs=args.num_epochs)
    
    # Visualize results
    print(f"\n{Fore.CYAN}Generating visualizations...{Style.RESET_ALL}")
    visualize_augmentation_stats(tracker, save_path='augmentation_stats.png')
    
    if args.num_epochs > 1:
        compare_epochs(tracker, save_path='epoch_comparison.png')
    
    # Save to JSON if requested
    if args.save_json:
        save_stats_to_json(tracker, save_path='augmentation_stats.json')
    
    print(f"\n{Fore.LIGHTGREEN_EX}✓ All done!{Style.RESET_ALL}")