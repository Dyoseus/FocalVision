import cv2
import numpy as np
import matplotlib.pyplot as plt

class FocalSplit:
    """
    Focal Split implementation (Equation 11 from paper).
    
    Key differences from Focal Track:
    - Uses image ALIGNMENT (magnification correction) instead of magnification term
    - Implements: Z = a / (b + I_s_tilde / ∇²I_tilde)
    - Where tilde (~) means "aligned/rescaled images"
    - More stable than Focal Track's Equation 15
    """
    
    def __init__(self, a, b, s1, s2):
        """        
        Args:
            a: Calibrated parameter = -A²
            b: Calibrated parameter = -A²(1/f - 1/s)
            s1: Sensor distance 1 (meters)
            s2: Sensor distance 2 (meters)
        """
        self.a = a
        self.b = b
        self.s1 = s1
        self.s2 = s2
        self.consensus_s = (s1 + s2) / 2.0  # Consensus sensor location
        
        print(f"\n{'='*60}")
        print("FOCAL SPLIT - Snapshot Depth from Differential Defocus")
        print(f"{'='*60}")
        print("Using Equation 11 (with image alignment)")
        print(f"  - Parameters: a={a:.6f}, b={b:.6f}")
        print(f"  - Sensor distances: s1={s1:.6f}m, s2={s2:.6f}m")
        print(f"  - Δs = {abs(s2-s1):.6f}m")
        print(f"  - Consensus location: c={self.consensus_s:.6f}m")
        print(f"{'='*60}\n")
    
    def remove_background(self, img, K=21):
        """Remove background illumination using box filter."""
        blurred = cv2.boxFilter(img, -1, (K, K))
        return img - blurred
    
    def align_images(self, img1, img2):
        """
        Align img1 to img2 using homography to correct magnification.
        
        This is the KEY difference from Focal Track:
        We rescale images to a consensus sensor location to remove
        magnification effects, implementing Equation 6 from paper:
        
        I_tilde(x; s) = I(s/c * x)
        
        Args:
            img1: First image (at sensor distance s1)
            img2: Second image (at sensor distance s2)
            
        Returns:
            img1_aligned: img1 rescaled to consensus
            img2_aligned: img2 rescaled to consensus
            R: Rotation matrix
            t: Translation vector
        """
        print("  - Aligning images to consensus sensor location")
        
        # Detect keypoints using SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test (Lowe's ratio)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        print(f"    Found {len(good_matches)} good matches")
        
        if len(good_matches) < 10:
            print("    ⚠ Warning: Few matches found, using identity transform")
            return img1, img2, np.eye(2), np.zeros(2)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Compute homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        if H is None:
            print("    ⚠ Warning: Homography failed, using identity transform")
            return img1, img2, np.eye(2), np.zeros(2)
        
        # Warp img1 to align with img2
        h, w = img2.shape[:2]
        img1_aligned = cv2.warpPerspective(img1, H, (w, h))
        
        # Extract rotation and translation (approximation for small changes)
        R = H[:2, :2]
        t = H[:2, 2]
        
        # Compute scaling factor
        scale = np.sqrt(np.linalg.det(R))
        print(f"    Scale factor: {scale:.6f}")
        print(f"    Translation: [{t[0]:.2f}, {t[1]:.2f}] pixels")
        
        return img1_aligned, img2, R, t
    
    def estimate_depth(self, img1, img2,
                      hpSize=21, smooth=11,
                      depth_min=0.4, depth_max=1.0,
                      conf_percentile=10,
                      use_alignment=True):
        """
        Estimate depth using Focal Split algorithm (Equation 11).
        
        Key equation:
        Z = a / (b + I_s_tilde / ∇²I_tilde)
        
        where:
        - I_s_tilde = I1_aligned - I2_aligned (after magnification correction)
        - ∇²I_tilde = Laplacian of average aligned image
        
        Args:
            img1: First image (sensor distance s1)
            img2: Second image (sensor distance s2)
            hpSize: High-pass filter size for background removal
            smooth: Gaussian smoothing size
            depth_min: Minimum valid depth in meters
            depth_max: Maximum valid depth in meters
            conf_percentile: Percentile for confidence thresholding
            use_alignment: If True, align images (Focal Split). If False, skip (Focal Track)
            
        Returns:
            depth_map: Estimated depth in meters
            confidence_map: Confidence map (0-1)
        """
        print("Processing images with Focal Split...")
        
        # Convert to float64 for numerical stability
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            gray1 = img1.astype(np.float64) / 255.0
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            gray2 = img2.astype(np.float64) / 255.0
        
        # FOCAL SPLIT KEY STEP: Align images to remove magnification
        if use_alignment:
            gray1_aligned, gray2_aligned, R, t = self.align_images(
                (gray1 * 255).astype(np.uint8),
                (gray2 * 255).astype(np.uint8)
            )
            gray1_aligned = gray1_aligned.astype(np.float64) / 255.0
            gray2_aligned = gray2_aligned.astype(np.float64) / 255.0
        else:
            print("  - Skipping alignment (Focal Track mode)")
            gray1_aligned = gray1
            gray2_aligned = gray2
        
        # Remove background
        print(f"  - Removing background (K={hpSize})")
        gray1_clean = self.remove_background(gray1_aligned, K=hpSize)
        gray2_clean = self.remove_background(gray2_aligned, K=hpSize)
        
        # Denoise
        print(f"  - Denoising (sigma={smooth})")
        gray1_clean = cv2.GaussianBlur(gray1_clean, (smooth, smooth), 0)
        gray2_clean = cv2.GaussianBlur(gray2_clean, (smooth, smooth), 0)
        
        # Compute derivatives on ALIGNED images
        # This is I_tilde from Equation 11
        I_avg_tilde = (gray1_clean + gray2_clean) / 2.0
        I_s_tilde = gray1_clean - gray2_clean  # Focus derivative on aligned images
        
        print("  - Computing spatial derivatives on aligned images")
        laplacian_tilde = cv2.Laplacian(I_avg_tilde, cv2.CV_64F, ksize=3)
        
        # Filter derivatives for stability
        print("  - Applying spatial filtering to derivatives")
        I_s_filtered = cv2.boxFilter(I_s_tilde, -1, (21, 21))
        laplacian_filtered = cv2.boxFilter(laplacian_tilde, -1, (21, 21))
        
        # Numerical stability
        epsilon = 1e-6
        
        # Compute ratio: I_s_tilde / ∇²I_tilde
        print("  - Computing depth ratio from aligned derivatives")
        valid_laplacian = np.abs(laplacian_filtered) > epsilon
        
        ratio = np.zeros_like(I_s_filtered)
        ratio[valid_laplacian] = I_s_filtered[valid_laplacian] / (
            laplacian_filtered[valid_laplacian] + epsilon
        )
        
        # Robust outlier clipping
        ratio_valid = ratio[valid_laplacian]
        if len(ratio_valid) > 0:
            ratio_5th = np.percentile(ratio_valid, 5)
            ratio_95th = np.percentile(ratio_valid, 95)
            ratio = np.clip(ratio, ratio_5th * 2, ratio_95th * 2)
        
        # Compute depth: Z = a / (b + ratio)
        # eqn 11 from paper
        print("  - Computing final depth using Equation 11")
        denominator_depth = self.b + ratio
        valid_denominator = np.abs(denominator_depth) > epsilon
        
        Z = np.full_like(ratio, np.nan)
        Z[valid_denominator] = self.a / (denominator_depth[valid_denominator] + epsilon)
        
        # Confidence from I_s_tilde²
        confidence = I_s_filtered ** 2
        
        # Confidence thresholding
        if conf_percentile > 0:
            conf_threshold = np.percentile(confidence, conf_percentile)
            print(f"  - Applying confidence threshold at {conf_percentile}th percentile: {conf_threshold:.2e}")
            confidence_mask = confidence > conf_threshold
        else:
            confidence_mask = confidence > 1e-6
        
        # Apply all validity masks
        valid_mask = valid_laplacian & valid_denominator & confidence_mask
        valid_mask &= (Z >= depth_min) & (Z <= depth_max) & (~np.isnan(Z))
        
        Z[~valid_mask] = np.nan
        
        # Normalize confidence
        confidence_norm = np.zeros_like(confidence)
        if np.max(confidence) > 0:
            confidence_norm = confidence / np.max(confidence)
        confidence_norm[~valid_mask] = 0
        
        # Statistics
        valid_depth = Z[~np.isnan(Z)]
        if len(valid_depth) > 0:
            print(f"\nResults:")
            print(f"  I_s_tilde range: [{np.min(I_s_filtered):.6f}, {np.max(I_s_filtered):.6f}]")
            print(f"  ∇²I_tilde range: [{np.min(laplacian_filtered):.6f}, {np.max(laplacian_filtered):.6f}]")
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
        axes[0].set_title('Depth Map (Focal Split)')
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], label='Depth (m)')
        
        # Confidence map
        im2 = axes[1].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Confidence Map')
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], label='Confidence')
        
        # High-confidence depth
        valid_depth = depth.copy()
        valid_depth[confidence < 0.1] = np.nan
        im3 = axes[2].imshow(valid_depth, cmap='jet', vmin=depth_min, vmax=depth_max)
        axes[2].set_title('High-Confidence Depth')
        axes[2].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[2], label='Depth (m)')
        
        plt.tight_layout()
        plt.savefig('focal_split_depth.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization to 'focal_split_depth.png'")
        plt.show()


def calibrate_focal_split(img1, img2, known_depth_range=(0.4, 1.0), 
                         s1=0.030, s2=0.0304,
                         hpSize=21, smooth=11):
    """
    Calibrate Focal Split parameters.
    
    Uses aligned images to compute ratio statistics and map them
    to the known depth range.
    
    Args:
        img1: First image (at sensor distance s1)
        img2: Second image (at sensor distance s2)
        known_depth_range: Expected (min, max) depth in meters
        s1: Sensor distance 1 in meters (e.g., 30mm = 0.030m)
        s2: Sensor distance 2 in meters (e.g., 30.4mm = 0.0304m)
        hpSize: Background removal filter size
        smooth: Gaussian smoothing size
        
    Returns:
        a_calibrated: Calibrated a parameter
        b_calibrated: Calibrated b parameter
    """
    print("\n" + "="*60)
    print("CALIBRATING FOCAL SPLIT PARAMETERS")
    print("="*60)
    print(f"Expected depth range: {known_depth_range[0]:.1f}m - {known_depth_range[1]:.1f}m")
    print(f"Sensor distances: s1={s1*1000:.2f}mm, s2={s2*1000:.2f}mm")
    print(f"Δs = {abs(s2-s1)*1000:.2f}mm")
    
    # Convert to float64
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        gray1 = img1.astype(np.float64) / 255.0
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        gray2 = img2.astype(np.float64) / 255.0
    
    # Create temporary instance for alignment
    temp_fs = FocalSplit(a=1.0, b=0.0, s1=s1, s2=s2)
    
    # Align images
    gray1_aligned, gray2_aligned, R, t = temp_fs.align_images(
        (gray1 * 255).astype(np.uint8),
        (gray2 * 255).astype(np.uint8)
    )
    gray1_aligned = gray1_aligned.astype(np.float64) / 255.0
    gray2_aligned = gray2_aligned.astype(np.float64) / 255.0
    
    # Process aligned images
    def remove_bg(img, K):
        return img - cv2.boxFilter(img, -1, (K, K))
    
    gray1_clean = cv2.GaussianBlur(remove_bg(gray1_aligned, hpSize), (smooth, smooth), 0)
    gray2_clean = cv2.GaussianBlur(remove_bg(gray2_aligned, hpSize), (smooth, smooth), 0)
    
    # Compute derivatives on aligned images
    I_avg_tilde = (gray1_clean + gray2_clean) / 2.0
    I_s_tilde = gray1_clean - gray2_clean
    laplacian_tilde = cv2.Laplacian(I_avg_tilde, cv2.CV_64F, ksize=3)
    
    # Filter before computing ratio
    I_s_filtered = cv2.boxFilter(I_s_tilde, -1, (21, 21))
    laplacian_filtered = cv2.boxFilter(laplacian_tilde, -1, (21, 21))
    
    # Compute ratio
    epsilon = 1e-6
    valid_mask = np.abs(laplacian_filtered) > epsilon
    
    ratio = np.zeros_like(I_s_filtered)
    ratio[valid_mask] = I_s_filtered[valid_mask] / (laplacian_filtered[valid_mask] + epsilon)
    
    # Use only valid ratios for calibration
    valid_ratio = ratio[valid_mask & np.isfinite(ratio)]
    
    if len(valid_ratio) == 0:
        print("⚠ Warning: No valid ratios found! Using default parameters.")
        return 1.0, 0.0
    
    # Use 10th and 90th percentiles for robustness
    ratio_10th = np.percentile(valid_ratio, 10)
    ratio_90th = np.percentile(valid_ratio, 90)
    
    # Calibrate: Z = a / (b + ratio)
    depth_min, depth_max = known_depth_range
    
    # Solve system of equations:
    # depth_max = a / (b + ratio_10th)  <- near objects, small ratio
    # depth_min = a / (b + ratio_90th)  <- far objects, large ratio
    b_calibrated = (depth_max * ratio_10th - depth_min * ratio_90th) / (depth_min - depth_max)
    a_calibrated = depth_max * (b_calibrated + ratio_10th)
    
    print(f"\nCalibration complete:")
    print(f"  Ratio range (10th-90th percentile): [{ratio_10th:.6f}, {ratio_90th:.6f}]")
    print(f"  Calibrated a: {a_calibrated:.6f}")
    print(f"  Calibrated b: {b_calibrated:.6f}")
    print("="*60 + "\n")
    
    return a_calibrated, b_calibrated


def main():
    """Main function for Focal Split depth estimation."""
    # Load images captured at different sensor distances
    img1 = cv2.imread('image1.png')  # Captured at sensor distance s1
    img2 = cv2.imread('image2.png')  # Captured at sensor distance s2
    
    if img1 is None or img2 is None:
        print("Error: Could not load images 'image1.png' and 'image2.png'")
        print("These should be captured simultaneously at different sensor distances")
        print("(e.g., using a beamsplitter setup)")
        return None, None
    
    print("Loaded images successfully")
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")
    
    # Hardware parameters (adjust based on your setup)
    s1 = 0.030   # Sensor distance 1: 30mm
    s2 = 0.0304  # Sensor distance 2: 30.4mm (0.4mm difference)
    
    # Calibrate
    a_cal, b_cal = calibrate_focal_split(
        img1, img2,
        known_depth_range=(0.4, 1.0),
        s1=s1,
        s2=s2,
        hpSize=21,
        smooth=11
    )
    
    # Initialize Focal Split with calibrated parameters
    focal_split = FocalSplit(a=a_cal, b=b_cal, s1=s1, s2=s2)
    
    # Estimate depth
    depth, confidence = focal_split.estimate_depth(
        img1, img2,
        hpSize=21,
        smooth=11,
        depth_min=0.4,
        depth_max=1.0,
        conf_percentile=10,
        use_alignment=True  # Set False to compare with Focal Track
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
    focal_split.visualize_depth(depth, confidence, depth_min=0.4, depth_max=1.0)
    
    return depth, confidence


if __name__ == "__main__":
    depth, confidence = main()