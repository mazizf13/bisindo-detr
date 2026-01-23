# visualize_augmentation.py
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data import DETRData
from utils.boxes import stacker, rescale_bboxes
from utils.setup import get_classes
from colorama import Fore
import argparse

def denormalize_image(img_tensor):
    """Denormalize image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img_tensor * std + mean
    return torch.clamp(img_denorm, 0, 1)

def visualize_batch(dataloader, num_batches=1, save_path='augmentation_vis.png'):
    """Visualize augmented batches"""
    CLASSES = get_classes()
    
    for batch_idx in range(num_batches):
        X, y = next(iter(dataloader))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, (img, annotations, ax) in enumerate(zip(X, y, axes)):
            img_denorm = denormalize_image(img)
            ax.imshow(img_denorm.permute(1, 2, 0))
            
            # Plot bboxes
            boxes = rescale_bboxes(annotations['boxes'], (224, 224))
            labels = annotations['labels']
            
            for label, bbox in zip(labels, boxes):
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                width, height = xmax - xmin, ymax - ymin
                
                # Draw rectangle
                rect = Rectangle((xmin, ymin), width, height,
                               linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                ax.text(xmin, ymin - 5, CLASSES[label], 
                       fontsize=12, color='white',
                       bbox=dict(facecolor='red', alpha=0.7))
            
            ax.set_title(f'Sample {idx+1} - {len(labels)} objects', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}_batch{batch_idx}.png', dpi=150, bbox_inches='tight')
        print(f"{Fore.GREEN}✓ Saved: {save_path}_batch{batch_idx}.png{Fore.RESET}")
        plt.close()

def collect_augmentation_stats(dataloader, num_samples=100):
    """Collect statistics about augmentations"""
    stats = {
        'total_samples': 0,
        'total_objects': 0,
        'avg_objects_per_image': 0,
        'bbox_sizes': [],
        'empty_bbox_count': 0
    }
    
    print(f"{Fore.CYAN}Collecting statistics from {num_samples} samples...{Fore.RESET}")
    
    for batch_idx, batch in enumerate(dataloader):
        if stats['total_samples'] >= num_samples:
            break
        
        X, y = batch
        stats['total_samples'] += len(X)
        
        for annotations in y:
            num_objects = len(annotations['labels'])
            stats['total_objects'] += num_objects
            
            if num_objects == 0:
                stats['empty_bbox_count'] += 1
            
            # Collect bbox sizes
            for bbox in annotations['boxes']:
                # bbox in YOLO format: [x_center, y_center, width, height]
                width, height = bbox[2].item(), bbox[3].item()
                stats['bbox_sizes'].append((width, height))
    
    # Calculate averages
    stats['avg_objects_per_image'] = stats['total_objects'] / stats['total_samples']
    
    # Print statistics
    print(f"\n{Fore.LIGHTCYAN_EX}=== AUGMENTATION STATISTICS ==={Fore.RESET}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Avg objects/image: {stats['avg_objects_per_image']:.2f}")
    print(f"Empty bbox samples: {stats['empty_bbox_count']} ({100*stats['empty_bbox_count']/stats['total_samples']:.1f}%)")
    
    if len(stats['bbox_sizes']) > 0:
        avg_width = np.mean([w for w, h in stats['bbox_sizes']])
        avg_height = np.mean([h for w, h in stats['bbox_sizes']])
        print(f"Avg bbox size: {avg_width:.3f} x {avg_height:.3f}")
    
    return stats

def compare_train_test_augmentation(train_loader, test_loader):
    """Compare training vs testing augmentation"""
    print(f"\n{Fore.LIGHTCYAN_EX}=== COMPARING TRAIN vs TEST AUGMENTATION ==={Fore.RESET}")
    
    # Get samples
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Training samples (top row)
    for idx in range(4):
        img = denormalize_image(X_train[idx])
        axes[0, idx].imshow(img.permute(1, 2, 0))
        axes[0, idx].set_title(f'TRAIN - Sample {idx+1}', fontsize=14, fontweight='bold')
        axes[0, idx].axis('off')
    
    # Testing samples (bottom row)
    for idx in range(4):
        img = denormalize_image(X_test[idx])
        axes[1, idx].imshow(img.permute(1, 2, 0))
        axes[1, idx].set_title(f'TEST - Sample {idx+1}', fontsize=14, fontweight='bold')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('train_vs_test_comparison.png', dpi=150, bbox_inches='tight')
    print(f"{Fore.GREEN}✓ Saved: train_vs_test_comparison.png{Fore.RESET}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize data augmentation')
    parser.add_argument('--mode', type=str, default='visualize', 
                       choices=['visualize', 'stats', 'compare'],
                       help='Mode: visualize, stats, or compare')
    parser.add_argument('--num_batches', type=int, default=3,
                       help='Number of batches to visualize')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for statistics')
    
    args = parser.parse_args()
    
    # Load datasets
    train_dataset = DETRData('data/train', train=True)
    train_loader = DataLoader(train_dataset, batch_size=4, 
                             collate_fn=stacker, drop_last=True)
    
    test_dataset = DETRData('data/test', train=False)
    test_loader = DataLoader(test_dataset, batch_size=4, 
                            collate_fn=stacker, drop_last=True)
    
    if args.mode == 'visualize':
        print(f"{Fore.CYAN}Visualizing training augmentation...{Fore.RESET}")
        visualize_batch(train_loader, num_batches=args.num_batches, 
                       save_path='train_augmentation')
        
    elif args.mode == 'stats':
        collect_augmentation_stats(train_loader, num_samples=args.num_samples)
        
    elif args.mode == 'compare':
        compare_train_test_augmentation(train_loader, test_loader)
    
    print(f"\n{Fore.GREEN}✓ Done!{Fore.RESET}")