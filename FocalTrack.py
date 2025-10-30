import cv2
import numpy as np
import matplotlib.pyplot as plt

class FocalTrack:
    """
    Focal Track implementation (Equation 15 from paper)
    """
    
    def __init__(self, a, b, d):
        """
        Args:
            a: Calibrated parameter
            b: Calibrated parameter
            d: Magnification constant (set to 0 for stability)
        """
        self.a = a
        self.b = b
        self.d = d
        
        print(f"\n{'='*60}")
        print("FOCAL TRACK - Depth from Differential Defocus (FIXED)")
        print(f"{'='*60}")
        print("Using Equation 15 with stability improvements")
        print(f"  - Parameters: a={a:.6f}, b={b:.6f}, d={d:.6f}")
        print(f"{'='*60}\n")
    
    def remove_background(self, img, K=21):
        """Remove background illumination using box filter."""
        blurred = cv2.boxFilter(img, -1, (K, K))
        return img - blurred
    
    def estimate_depth(self, img1, img2,
                      hpSize=21, smooth=11,
                      depth_min=0.4, depth_max=1.0,
                      conf_percentile=10):
        """
        Estimate depth using Focal Track algorithm with fixes.
        
        Args:
            img1: First image
            img2: Second image
            hpSize: High-pass filter size for background removal
            smooth: Gaussian smoothing size
            depth_min: Minimum valid depth in meters
            depth_max: Maximum valid depth in meters
            conf_percentile: Percentile for confidence thresholding (default 10)
            
        Returns:
            depth_map: Estimated depth in meters
            confidence_map: Confidence map (0-1)
        """
        print("Processing images with Focal Track (Fixed)...")
        
        # Convert to float64 for numerical stability
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            gray1 = img1.astype(np.float64) / 255.0
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            gray2 = img2.astype(np.float64) / 255.0
        
        # Remove background
        print(f"  - Removing background (K={hpSize})")
        gray1_clean = self.remove_background(gray1, K=hpSize)
        gray2_clean = self.remove_background(gray2, K=hpSize)
        
        # Denoise
        print(f"  - Denoising (sigma={smooth})")
        gray1_clean = cv2.GaussianBlur(gray1_clean, (smooth, smooth), 0)
        gray2_clean = cv2.GaussianBlur(gray2_clean, (smooth, smooth), 0)
        
        # Average for spatial derivatives
        I_avg = (gray1_clean + gray2_clean) / 2.0
        
        # Focus derivative (Iₛ)
        I_s = gray1_clean - gray2_clean
        
        # Spatial derivatives and Laplacian
        print("  - Computing spatial derivatives")
        laplacian = cv2.Laplacian(I_avg, cv2.CV_64F, ksize=3)
        
        # FIX 1: Filter derivatives BEFORE division for stability
        print("  - Applying spatial filtering to derivatives")
        I_s_filtered = cv2.boxFilter(I_s, -1, (21, 21))
        laplacian_filtered = cv2.boxFilter(laplacian, -1, (21, 21))
        
        # FIX 2: Use larger epsilon to prevent division issues
        epsilon = 1e-6  # Increased from 1e-12
        
        # Compute ratio with better numerical stability
        print("  - Computing depth ratio")
        numerator = I_s_filtered
        denominator = laplacian_filtered
        
        # FIX 3: Mask out regions where denominator is too small
        valid_laplacian = np.abs(denominator) > epsilon
        
        # Compute ratio only where valid
        ratio = np.zeros_like(numerator)
        ratio[valid_laplacian] = numerator[valid_laplacian] / (denominator[valid_laplacian] + epsilon)
        
        # FIX 4: Clip ratio to reasonable range to prevent extreme values
        ratio_valid = ratio[valid_laplacian]
        if len(ratio_valid) > 0:
            ratio_5th = np.percentile(ratio_valid, 5)
            ratio_95th = np.percentile(ratio_valid, 95)
            ratio = np.clip(ratio, ratio_5th * 2, ratio_95th * 2)
        
        # Compute depth: Z = a / (b + ratio)
        print("  - Computing final depth")
        denominator_depth = self.b + ratio
        
        # FIX 5: Avoid division by very small denominators
        valid_denominator = np.abs(denominator_depth) > epsilon
        
        Z = np.full_like(ratio, np.nan)
        Z[valid_denominator] = self.a / (denominator_depth[valid_denominator] + epsilon)
        
        # Confidence from I_s² (higher variation = more confident)
        confidence = I_s_filtered ** 2
        
        # FIX 6: More aggressive confidence thresholding
        if conf_percentile > 0:
            conf_threshold = np.percentile(confidence, conf_percentile)
            print(f"  - Applying confidence threshold at {conf_percentile}th percentile: {conf_threshold:.2e}")
            confidence_mask = confidence > conf_threshold
        else:
            confidence_mask = confidence > 1e-6
        
        # Apply all masks
        valid_mask = valid_laplacian & valid_denominator & confidence_mask
        valid_mask &= (Z >= depth_min) & (Z <= depth_max) & (~np.isnan(Z))
        
        Z[~valid_mask] = np.nan
        
        # Normalize confidence for visualization
        confidence_norm = np.zeros_like(confidence)
        if np.max(confidence) > 0:
            confidence_norm = confidence / np.max(confidence)
        confidence_norm[~valid_mask] = 0
        
        # Statistics
        valid_depth = Z[~np.isnan(Z)]
        if len(valid_depth) > 0:
            print(f"\nResults:")
            print(f"  I_s range: [{np.min(I_s_filtered):.6f}, {np.max(I_s_filtered):.6f}]")
            print(f"  Laplacian range: [{np.min(laplacian_filtered):.6f}, {np.max(laplacian_filtered):.6f}]")
            print(f"  Ratio range: [{np.nanmin(ratio):.6f}, {np.nanmax(ratio):.6f}]")
            print(f"  Depth range: [{np.nanmin(Z):.3f}, {np.nanmax(Z):.3f}] m")
            print(f"  Valid pixels: {len(valid_depth):,} / {Z.size:,} ({100*len(valid_depth)/Z.size:.1f}%)")
        else:
            print("\n⚠ Warning: No valid depth pixels found!")
        
        return Z, confidence_norm
    
    def visualize_depth(self, depth, confidence,
                       depth_min=0.4, depth_max=1.0):
        """Visualize depth map and confidence."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Depth map
        im1 = axes[0].imshow(depth, cmap='jet', vmin=depth_min, vmax=depth_max)
        axes[0].set_title('Depth Map (Focal Track - Fixed)')
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], label='Depth (m)')
        
        # Confidence map
        im2 = axes[1].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Confidence Map')
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], label='Confidence')
        
        # Valid depth (masked by confidence)
        valid_depth = depth.copy()
        valid_depth[confidence < 0.1] = np.nan
        im3 = axes[2].imshow(valid_depth, cmap='jet', vmin=depth_min, vmax=depth_max)
        axes[2].set_title('High-Confidence Depth')
        axes[2].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[2], label='Depth (m)')
        
        plt.tight_layout()
        plt.savefig('focal_track_depth_fixed.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization to 'focal_track_depth_fixed.png'")
        plt.show()


def calibrate_focal_track(img1, img2, known_depth_range=(0.4, 1.0), 
                         hpSize=21, smooth=11):
    """
    Calibrate Focal Track parameters - IMPROVED VERSION
    """
    print("\n" + "="*60)
    print("CALIBRATING FOCAL TRACK PARAMETERS (IMPROVED)")
    print("="*60)
    print(f"Expected depth range: {known_depth_range[0]:.1f}m - {known_depth_range[1]:.1f}m")
    
    # Convert to float64
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        gray1 = img1.astype(np.float64) / 255.0
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        gray2 = img2.astype(np.float64) / 255.0
    
    # Process images
    def remove_bg(img, K):
        return img - cv2.boxFilter(img, -1, (K, K))
    
    gray1_clean = cv2.GaussianBlur(remove_bg(gray1, hpSize), (smooth, smooth), 0)
    gray2_clean = cv2.GaussianBlur(remove_bg(gray2, hpSize), (smooth, smooth), 0)
    
    I_avg = (gray1_clean + gray2_clean) / 2.0
    I_s = gray1_clean - gray2_clean
    laplacian = cv2.Laplacian(I_avg, cv2.CV_64F, ksize=3)
    
    # IMPROVED: Filter before computing ratio
    I_s_filtered = cv2.boxFilter(I_s, -1, (21, 21))
    laplacian_filtered = cv2.boxFilter(laplacian, -1, (21, 21))
    
    # Compute ratio with better masking
    epsilon = 1e-6
    valid_mask = np.abs(laplacian_filtered) > epsilon
    
    ratio = np.zeros_like(I_s_filtered)
    ratio[valid_mask] = I_s_filtered[valid_mask] / (laplacian_filtered[valid_mask] + epsilon)
    
    # Use only valid ratios for calibration
    valid_ratio = ratio[valid_mask & np.isfinite(ratio)]
    
    if len(valid_ratio) == 0:
        print("⚠ Warning: No valid ratios found! Using default parameters.")
        return 1.0, 0.0, 0.0
    
    # Use 10th and 90th percentiles for robustness
    ratio_10th = np.percentile(valid_ratio, 10)
    ratio_90th = np.percentile(valid_ratio, 90)
    
    # Calibrate: Z = a / (b + ratio)
    depth_min, depth_max = known_depth_range
    
    # Solve system of equations
    b_calibrated = (depth_max * ratio_10th - depth_min * ratio_90th) / (depth_min - depth_max)
    a_calibrated = depth_max * (b_calibrated + ratio_10th)
    
    print(f"\nCalibration complete:")
    print(f"  Ratio range (10th-90th percentile): [{ratio_10th:.6f}, {ratio_90th:.6f}]")
    print(f"  Calibrated a: {a_calibrated:.6f}")
    print(f"  Calibrated b: {b_calibrated:.6f}")
    print(f"  Calibrated d: 0.000000 (magnification term disabled)")
    print("="*60 + "\n")
    
    return a_calibrated, b_calibrated, 0.0


def main():
    """Main function for Focal Track depth estimation."""
    # Load images
    img1 = cv2.imread('image1.png')
    img2 = cv2.imread('image2.png')
    
    if img1 is None or img2 is None:
        print("Error: Could not load images 'image1.png' and 'image2.png'")
        return None, None
    
    print("Loaded images successfully")
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")
    
    # Calibrate
    a_cal, b_cal, d_cal = calibrate_focal_track(
        img1, img2,
        known_depth_range=(0.4, 1.0),
        hpSize=21,
        smooth=11
    )
    
    # Initialize with calibrated parameters
    focal_track = FocalTrack(a=a_cal, b=b_cal, d=d_cal)
    
    # Estimate depth with better parameters
    depth, confidence = focal_track.estimate_depth(
        img1, img2,
        hpSize=21,
        smooth=11,
        depth_min=0.4,
        depth_max=1.0,
        conf_percentile=10  # Filter bottom 10% confidence
    )
    
    # Statistics
    valid_depth = depth[~np.isnan(depth)]
    if len(valid_depth) > 0:
        print(f"\nDepth Statistics:")
        print(f"  Min:    {np.min(valid_depth):.3f} m")
        print(f"  Max:    {np.max(valid_depth):.3f} m")
        print(f"  Median: {np.median(valid_depth):.3f} m")
        print(f"  Mean:   {np.mean(valid_depth):.3f} m")
        print(f"  Std:    {np.std(valid_depth):.3f} m")
    
    # Visualize
    focal_track.visualize_depth(depth, confidence, depth_min=0.4, depth_max=1.0)
    
    return depth, confidence