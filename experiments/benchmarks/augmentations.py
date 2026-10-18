import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import glob
import argparse

def create_rotation_matrix(angle, axis='z'):
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    if axis == 'z':
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
    elif axis == 'x':
        rot_matrix = torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], dtype=torch.float32)
    elif axis == 'y':
        rot_matrix = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=torch.float32)
    
    return rot_matrix

def apply_time_warping(body_xyz, time_warp_config):
    """
    Simplified time warping with minimal computational overhead.
    Uses simple interpolation methods for better compatibility.
    """
    N, T, J, D = body_xyz.shape
    augmented = body_xyz.clone()
    
    # Global speed change (simplified)
    if time_warp_config.get('global_speed', True) and random.choice([0, 1]):
        speed_factor = time_warp_config.get('global_speed_range', 0.3)
        speed_multiplier = 1.0 + (torch.rand(1) * 2 - 1) * speed_factor
        
        # Simple approach: just scale the temporal indices
        new_length = max(10, min(int(T * speed_multiplier), T * 2))
        
        # Create new time indices
        if new_length != T:
            # Interpolate each dimension separately for simplicity
            for n in range(N):
                for j in range(J):
                    for d in range(D):
                        # Get the sequence for this joint/dimension
                        sequence = augmented[n, :, j, d]
                        
                        # Simple linear interpolation
                        if new_length != T:
                            # Reshape for interpolation: (T,) -> (1, 1, T)
                            sequence_reshaped = sequence.unsqueeze(0).unsqueeze(0)
                            
                            # Use torch.nn.functional.interpolate
                            interpolated = F.interpolate(sequence_reshaped, size=new_length, mode='linear', align_corners=True)
                            interpolated = interpolated.squeeze(0).squeeze(0)  # Back to (new_length,)
                            
                            if new_length > T:
                                # Upsample: interpolate to longer sequence, then truncate
                                augmented[n, :, j, d] = interpolated[:T]
                            else:
                                # Downsample: interpolate to shorter sequence, then pad
                                augmented[n, :new_length, j, d] = interpolated
                                # Pad with last value
                                if new_length < T:
                                    augmented[n, new_length:, j, d] = interpolated[-1]
    
    # Simplified local speed variations (only apply to root joint)
    if time_warp_config.get('local_speed', True) and random.choice([0, 1]):
        speed_range = time_warp_config.get('local_speed_range', 0.4)
        num_segments = min(time_warp_config.get('num_segments', 3), 2)  # Reduce segments
        
        # Only apply to root joint (joint 0) for simplicity
        if J > 0:
            segment_length = T // num_segments
            
            for seg in range(num_segments):
                start_idx = seg * segment_length
                end_idx = min((seg + 1) * segment_length, T)
                
                if end_idx - start_idx < 5:
                    continue
                
                # Apply speed change to root joint only
                speed_multiplier = 1.0 + (torch.rand(1) * 2 - 1) * speed_range
                segment_length_actual = end_idx - start_idx
                new_segment_length = max(5, min(int(segment_length_actual * speed_multiplier), segment_length_actual * 2))
                
                for n in range(N):
                    for d in range(D):
                        segment_data = augmented[n, start_idx:end_idx, 0, d]
                        
                        if new_segment_length != segment_length_actual:
                            # Reshape for interpolation: (segment_length_actual,) -> (1, 1, segment_length_actual)
                            segment_reshaped = segment_data.unsqueeze(0).unsqueeze(0)
                            
                            # Use torch.nn.functional.interpolate
                            interpolated = F.interpolate(segment_reshaped, size=new_segment_length, mode='linear', align_corners=True)
                            interpolated = interpolated.squeeze(0).squeeze(0)  # Back to (new_segment_length,)
                            
                            if new_segment_length > segment_length_actual:
                                augmented[n, start_idx:end_idx, 0, d] = interpolated[:segment_length_actual]
                            else:
                                augmented[n, start_idx:start_idx + new_segment_length, 0, d] = interpolated
                                if new_segment_length < segment_length_actual:
                                    augmented[n, start_idx + new_segment_length:end_idx, 0, d] = interpolated[-1]
    
    # Simplified frame dropping (only apply to a few random frames)
    if time_warp_config.get('frame_drop', True) and random.choice([0, 1]):
        drop_prob = time_warp_config.get('drop_prob', 0.1)
        
        # Only drop a few frames to keep it simple
        num_drops = max(1, int(T * drop_prob))
        drop_indices = torch.randperm(T)[:num_drops]
        
        # Keep first and last frames
        drop_indices = drop_indices[drop_indices != 0]
        drop_indices = drop_indices[drop_indices != T - 1]
        
        if len(drop_indices) > 0:
            for n in range(N):
                for j in range(J):
                    for d in range(D):
                        sequence = augmented[n, :, j, d]
                        
                        # Simple interpolation for dropped frames
                        for drop_idx in drop_indices:
                            if drop_idx > 0 and drop_idx < T - 1:
                                # Use average of neighboring frames
                                left_val = sequence[drop_idx - 1]
                                right_val = sequence[drop_idx + 1]
                                sequence[drop_idx] = (left_val + right_val) / 2
                        
                        augmented[n, :, j, d] = sequence
    
    return augmented

def apply_geometric_transformations(body_xyz, augmentation_config):
    N, T, J, D = body_xyz.shape
    augmented = body_xyz.clone()

    if augmentation_config.get('rotation', True) and random.choice([0, 1]):
        num_people = augmentation_config.get('num_people', 3)  # Assuming 3 people per sequence
        batch_size = N // num_people
        
        angles = torch.deg2rad(torch.rand(batch_size) * 360)
        
        rot_matrices = torch.stack([
            create_rotation_matrix(angle, axis='z') for angle in angles
        ]) 
        
        reshaped = augmented.reshape(batch_size, num_people, -1, 3) 
        rotated = torch.bmm(reshaped.reshape(batch_size, -1, 3), rot_matrices.transpose(1, 2))  
        augmented = rotated.reshape(N, T, J, 3)
    
    # Random person permutation within each sequence
    if augmentation_config.get('permute_p', True) and random.choice([0, 1]):
        num_people = augmentation_config.get('num_people', 3)  # Assuming 3 people per sequence
        batch_size = N // num_people
        
        # Generate random permutations for each batch
        perm = torch.argsort(torch.rand(batch_size, num_people), dim=-1)  # Shape: [batch_size, num_people]
        idx = torch.arange(batch_size).unsqueeze(-1)  # Shape: [batch_size, 1]
        
        # Reshape to group people by batch
        reshaped = augmented.reshape(batch_size, num_people, T, J, 3)  # [batch_size, num_people, T, J, 3]
        
        # Apply permutation to each batch
        permuted = reshaped[idx, perm]  # [batch_size, num_people, T, J, 3]
        
        # Reshape back to original format
        augmented = permuted.reshape(N, T, J, 3)
    
    # Random horizontal flip (X-axis)
    if augmentation_config.get('horizontal_flip', True) and random.choice([0, 1]):
        augmented = augmented * torch.tensor([-1, 1, 1], dtype=torch.float32)
    
    # Random vertical flip (Y-axis)
    if augmentation_config.get('vertical_flip', True) and random.choice([0, 1]):
        augmented = augmented * torch.tensor([1, -1, 1], dtype=torch.float32)
    
    # Random scaling
    if augmentation_config.get('scaling', True) and random.choice([0, 1]):
        scale_range = augmentation_config.get('scale_range', 0.1)  # ±10%
        scale_factor = 1.0 + (torch.rand(1) * 2 - 1) * scale_range
        augmented = augmented * scale_factor
    
    # Random translation 
    if augmentation_config.get('translation', True) and random.choice([0, 1]):
        trans_range = augmentation_config.get('translation_range', 0.1)
        translation = (torch.rand(3) * 2 - 1) * trans_range
        augmented = augmented + translation
    
    # Add small Gaussian noise to coordinates (invariant augmentation)
    if augmentation_config.get('coordinate_noise', False):
        noise_std = augmentation_config.get('noise_std', 0.01)
        augmented = augmented + torch.randn_like(augmented) * noise_std
    
    return augmented

def apply_combined_augmentations(body_xyz, augmentation_config):
    """
    Apply both geometric and temporal augmentations to the motion data.
    """
    # Apply geometric transformations first
    augmented = apply_geometric_transformations(body_xyz, augmentation_config)
    
    # Apply time warping if enabled
    if augmentation_config.get('time_warping', False):
        time_warp_config = {
            'global_speed': augmentation_config.get('global_speed', True),
            'global_speed_range': augmentation_config.get('global_speed_range', 0.3),
            'local_speed': augmentation_config.get('local_speed', True),
            'local_speed_range': augmentation_config.get('local_speed_range', 0.4),
            'num_segments': augmentation_config.get('num_segments', 3),
            'frame_drop': augmentation_config.get('frame_drop', False),
            'drop_prob': augmentation_config.get('drop_prob', 0.1)
        }
        augmented = apply_time_warping(augmented, time_warp_config)
    
    return augmented

def create_augmentation_configs(num_augmentations=3):
    base_config = {
        'rotation': True,
        'rotation_range': 30, 
        'horizontal_flip': True,
        'vertical_flip': True,
        'scaling': False,
        'scale_range': 0.1,
        'translation': False,
        'translation_range': 0.1,
        # Time warping parameters
        'time_warping': True,
        'global_speed': True,
        'global_speed_range': 0.2,  # Reduced: ±20% speed variation
        'local_speed': True,
        'local_speed_range': 0.25,  # Reduced: ±25% speed variation per segment
        'num_segments': 2,          # Reduced: 2 segments instead of 3
        'frame_drop': True,
        'drop_prob': 0.05           # Reduced: 5% chance to drop each frame
    }
    
    configs = []
    
    configs.append({
        'rotation': False,
        'horizontal_flip': False,
        'vertical_flip': False,
        'scaling': False,
        'translation': False,
        'time_warping': False
    })
    
    for i in range(num_augmentations - 1):
        config = base_config.copy()        
        config['rotation_range'] = 20 + (i * 10)  # 20, 30, 40 degrees
        config['scale_range'] = 0.05 + (i * 0.02)  # 0.05, 0.07, 0.09
        config['translation_range'] = 0.05 + (i * 0.02)  # 0.05, 0.07, 0.09
        
        # Randomly disable some augmentations for variety (reduced randomness)
        if random.choice([0, 0, 1]):  # 33% chance instead of 50%
            config['horizontal_flip'] = False
        if random.choice([0, 0, 1]):
            config['vertical_flip'] = False
        if random.choice([0, 0, 1]):
            config['scaling'] = False
        if random.choice([0, 0, 1]):
            config['translation'] = False
        if random.choice([0, 0, 1]):
            config['time_warping'] = False
        if random.choice([0, 0, 1]):
            config['global_speed'] = False
        if random.choice([0, 0, 1]):
            config['local_speed'] = False
        if random.choice([0, 0, 1]):
            config['frame_drop'] = False
            
        configs.append(config)
    
    return configs

def preprocess_with_augmentations(input_dir, output_dir, num_augmentations=3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    data_files = glob.glob(os.path.join(input_dir, "*.pt"))
    data_files = sorted(data_files)
    
    print(f"Found {len(data_files)} files in {input_dir}")
    print(f"Creating {num_augmentations} augmented versions per sample")
    
    augmentation_configs = create_augmentation_configs(num_augmentations)
    
    for file_path in tqdm(data_files, desc="Processing files"):
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        for aug_idx, config in enumerate(augmentation_configs):
            if aug_idx == 0:
                augmented_body_xyz = data.body_xyz.clone()
            else:
                augmented_body_xyz = apply_combined_augmentations(data.body_xyz, config)
            
            # Create new data object by copying the original and updating body_xyz
            augmented_data = data.clone()
            augmented_data.body_xyz = augmented_body_xyz
            
            output_filename = f"{name_without_ext}_aug{aug_idx:02d}.pt"
            output_path = os.path.join(output_dir, output_filename)
            torch.save(augmented_data, output_path)
    
    print(f"Augmented data saved to {output_dir}")
    print(f"Total files created: {len(data_files) * num_augmentations}")

def run_single_preprocessing(input_dir, output_dir, num_augmentations, seed):
    print(f"Processing directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of augmentations: {num_augmentations}")
    print(f"Seed: {seed}")
    
    preprocess_with_augmentations(
        input_dir=input_dir,
        output_dir=output_dir,
        num_augmentations=num_augmentations,
        seed=seed
    )

def run_complete_preprocessing(base_input_dir, base_output_dir, train_augmentations=3, 
                             val_augmentations=3, seed=42, skip_train=False, skip_val=False):        
    train_input_dir = os.path.join(base_input_dir, "train")
    val_input_dir = os.path.join(base_input_dir, "val")
    
    if not os.path.exists(train_input_dir):
        raise ValueError(f"Training directory {train_input_dir} does not exist")
    if not os.path.exists(val_input_dir):
        raise ValueError(f"Validation directory {val_input_dir} does not exist")
    
    # Create output directories
    train_output_dir = os.path.join(base_output_dir, "train")
    val_output_dir = os.path.join(base_output_dir, "val")
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MOTION CAPTURE DATA AUGMENTATION PREPROCESSING")
    print("=" * 60)
    print(f"Base input directory: {base_input_dir}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Training augmentations: {train_augmentations}")
    print(f"Validation augmentations: {val_augmentations}")
    print(f"Random seed: {seed}")
    print("=" * 60)
    
    success = True
    
    # Process training data
    if not skip_train:
        print("\nProcessing TRAINING data...")
        try:
            run_single_preprocessing(
                input_dir=train_input_dir,
                output_dir=train_output_dir,
                num_augmentations=train_augmentations,
                seed=seed
            )
        except Exception as e:
            print(f"Error processing training data: {e}")
            success = False
    else:
        print("\nSkipping training data preprocessing")
    
    # Process validation data
    if not skip_val:
        print("\nProcessing VALIDATION data...")
        try:
            run_single_preprocessing(
                input_dir=val_input_dir,
                output_dir=val_output_dir,
                num_augmentations=val_augmentations,
                seed=seed + 1  # Different seed for validation
            )
        except Exception as e:
            print(f"Error processing validation data: {e}")
            success = False
    else:
        print("\nSkipping validation data preprocessing")
    
    if success:
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Augmented training data: {train_output_dir}")
        print(f"Augmented validation data: {val_output_dir}")
        
        # Count files
        train_files = len([f for f in os.listdir(train_output_dir) if f.endswith('.pt')])
        val_files = len([f for f in os.listdir(val_output_dir) if f.endswith('.pt')])
        
        print(f"Training files: {train_files}")
        print(f"Validation files: {val_files}")
        
        # Calculate original vs augmented
        original_train = len([f for f in os.listdir(train_input_dir) if f.endswith('.pt')])
        original_val = len([f for f in os.listdir(val_input_dir) if f.endswith('.pt')])
        
        print(f"Original training samples: {original_train}")
        print(f"Augmented training samples: {train_files}")
        print(f"Training data multiplication factor: {train_files / original_train:.1f}x")
        
        print(f"Original validation samples: {original_val}")
        print(f"Augmented validation samples: {val_files}")
        print(f"Validation data multiplication factor: {val_files / original_val:.1f}x")
        
    else:
        print("\n" + "=" * 60)
        print("PREPROCESSING FAILED!")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Preprocess motion capture data with geometric augmentations')
    
    parser.add_argument('--mode', type=str, choices=['single', 'complete'], default='complete',
                       help='Processing mode: single directory or complete train/val preprocessing')
    parser.add_argument('--input_dir', type=str, 
                       help='Input directory containing .pt files (for single mode)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for augmented data (for single mode)')
    parser.add_argument('--num_augmentations', type=int, default=3,
                       help='Number of augmented versions per sample (default: 3)')
    
    parser.add_argument('--base_input_dir', type=str, 
                       default='data/preprocessed/mocap_UMPM_input25_v2',
                       help='Base directory containing train/ and val/ subdirectories (for complete mode)')
    parser.add_argument('--base_output_dir', type=str,
                       default='data/preprocessed/mocap_UMPM_input25_v2_augmented',
                       help='Base directory for output augmented data (for complete mode)')
    parser.add_argument('--train_augmentations', type=int, default=3,
                       help='Number of augmented versions for training data (default: 3)')
    parser.add_argument('--val_augmentations', type=int, default=3,
                       help='Number of augmented versions for validation data (default: 3)')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training data preprocessing (for complete mode)')
    parser.add_argument('--skip_val', action='store_true',
                       help='Skip validation data preprocessing (for complete mode)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.input_dir or not args.output_dir:
            raise ValueError("--input_dir and --output_dir are required for single mode")
        
        if not os.path.exists(args.input_dir):
            raise ValueError(f"Input directory {args.input_dir} does not exist")
        
        run_single_preprocessing(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_augmentations=args.num_augmentations,
            seed=args.seed
        )
        
    elif args.mode == 'complete':
        run_complete_preprocessing(
            base_input_dir=args.base_input_dir,
            base_output_dir=args.base_output_dir,
            train_augmentations=args.train_augmentations,
            val_augmentations=args.val_augmentations,
            seed=args.seed,
            skip_train=args.skip_train,
            skip_val=args.skip_val
        )

if __name__ == "__main__":
    main()
