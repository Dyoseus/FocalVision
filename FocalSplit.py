import cv2
import numpy as np
import matplotlib.pyplot as plt


class FocalSplit:
    """
    Focal Split implementation (Equation 11 from paper).

    Assumptions:
    - The two input images are already aligned (no magnification correction needed here).
    - I directly treat the inputs as the aligned/rescaled images (tilde variables).
    """

    def __init__(self, a, b, s1, s2):
        self.a = a
        self.b = b
        self.s1 = s1
        self.s2 = s2
        self.consensus_s = (s1 + s2) / 2.0

        print(f"\n{'='*60}")
        print("FOCAL SPLIT - Snapshot Depth from Differential Defocus")
        print(f"{'='*60}")
        print("Using Equation 11 (inputs assumed pre-aligned)")
        print(f"  - Parameters: a={a:.6f}, b={b:.6f}")
        print(f"  - Sensor distances: s1={s1:.6f}m, s2={s2:.6f}m")
        print(f"  - Δs = {abs(s2-s1):.6f}m")
        print(f"  - Consensus location: c={self.consensus_s:.6f}m")
        print(f"{'='*60}\n")

    def remove_background(self, img, K=21):
        """Remove background illumination using a box filter (high-pass)."""
        blurred = cv2.boxFilter(img, -1, (K, K))
        return img - blurred

    def estimate_depth(
        self,
        img1,
        img2,
        hpSize=21,
        smooth=11,
        depth_min=0.4,
        depth_max=1.0,
        conf_percentile=10,  
    ):
        print("Processing images with Focal Split")

        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            gray1 = img1.astype(np.float64) / 255.0

        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            gray2 = img2.astype(np.float64) / 255.0

        # Inputs are assumed already aligned
        gray1_aligned = gray1
        gray2_aligned = gray2

        print(f"  - Removing background (K={hpSize})")
        gray1_clean = self.remove_background(gray1_aligned, K=hpSize)
        gray2_clean = self.remove_background(gray2_aligned, K=hpSize)

        print(f"  - Denoising (kernel={smooth}x{smooth})")
        gray1_clean = cv2.GaussianBlur(gray1_clean, (smooth, smooth), 0)
        gray2_clean = cv2.GaussianBlur(gray2_clean, (smooth, smooth), 0)

        # Tilde variables are just the processed aligned inputs
        I_avg_tilde = (gray1_clean + gray2_clean) / 2.0
        I_s_tilde = gray1_clean - gray2_clean

        print("  - Computing spatial derivatives")
        laplacian_tilde = cv2.Laplacian(I_avg_tilde, cv2.CV_64F, ksize=3)

        print("  - Applying spatial filtering to derivatives")
        I_s_filtered = cv2.boxFilter(I_s_tilde, -1, (21, 21))
        laplacian_filtered = cv2.boxFilter(laplacian_tilde, -1, (21, 21))

        epsilon = 1e-6

        print("  - Computing ratio I_s_tilde / Laplacian(I_avg_tilde)")
        valid_laplacian = np.abs(laplacian_filtered) > epsilon

        ratio = np.zeros_like(I_s_filtered)
        ratio[valid_laplacian] = I_s_filtered[valid_laplacian] / (laplacian_filtered[valid_laplacian] + epsilon)

        # Robust clipping
        ratio_valid = ratio[valid_laplacian]
        if ratio_valid.size > 0:
            r5 = np.percentile(ratio_valid, 5)
            r95 = np.percentile(ratio_valid, 95)
            ratio = np.clip(ratio, r5 * 2, r95 * 2)

        print("  - Computing depth Z = a / (b + ratio)")
        denom = self.b + ratio
        valid_denom = np.abs(denom) > epsilon

        Z = np.full_like(ratio, np.nan)
        Z[valid_denom] = self.a / (denom[valid_denom] + epsilon)

        
        confidence_raw = (I_s_tilde ** 2)

        # For visualization, use log scaling to expand small differences
        confidence_vis = np.log1p(confidence_raw)

        # Robust normalize using percentiles (avoids one outlier flattening everything)
    
        valid_mask = valid_laplacian & valid_denom
        valid_mask &= np.isfinite(Z)
        valid_mask &= (Z >= depth_min) & (Z <= depth_max)

        Z[~valid_mask] = np.nan

        # Normalize confidence on valid pixels only
        confidence_norm = np.zeros_like(confidence_vis, dtype=np.float64)
        conf_vals = confidence_vis[valid_mask]

        if conf_vals.size > 0:
            lo = np.percentile(conf_vals, 1)
            hi = np.percentile(conf_vals, 99)
            if hi > lo:
                confidence_norm[valid_mask] = np.clip((confidence_vis[valid_mask] - lo) / (hi - lo), 0, 1)
            else:
                # Fallback if the distribution is basically constant
                confidence_norm[valid_mask] = 1.0

        #lenient confidence filtering
        if conf_percentile and conf_percentile > 0:
            conf_vals_norm = confidence_norm[valid_mask]
            if conf_vals_norm.size > 0:
                thresh = np.percentile(conf_vals_norm, conf_percentile)
                print(f"  - Lenient confidence filter: dropping bottom {conf_percentile}% (threshold={thresh:.4f})")
                keep = confidence_norm >= thresh
                Z[~keep] = np.nan
                confidence_norm[~np.isfinite(Z)] = 0.0

        valid_depth = Z[np.isfinite(Z)]
        if valid_depth.size > 0:
            print("\nResults:")
            print(f"  I_s range: [{np.min(I_s_filtered):.6f}, {np.max(I_s_filtered):.6f}]")
            print(f"  Laplacian range: [{np.min(laplacian_filtered):.6f}, {np.max(laplacian_filtered):.6f}]")
            print(f"  Ratio range: [{np.nanmin(ratio):.6f}, {np.nanmax(ratio):.6f}]")
            print(f"  Depth range: [{np.nanmin(Z):.3f}, {np.nanmax(Z):.3f}] m")
            print(f"  Valid pixels: {valid_depth.size:,} / {Z.size:,} ({100*valid_depth.size/Z.size:.1f}%)")
        else:
            print("\n⚠ Warning: No valid depth pixels found!")

        return Z, confidence_norm

    def visualize_depth(self, depth, confidence, depth_min=0.4, depth_max=1.0):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(depth, cmap="jet", vmin=depth_min, vmax=depth_max)
        axes[0].set_title("Depth Map (Focal Split)")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0], label="Depth (m)")

        im2 = axes[1].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("Confidence Map")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], label="Confidence")

        plt.tight_layout()
        plt.savefig("focal_split_depth.png", dpi=150, bbox_inches="tight")
        print("\n✓ Saved visualization to 'focal_split_depth.png'")
        plt.show()


def calibrate_focal_split(
    img1,
    img2,
    known_depth_range=(0.4, 1.0),
    s1=0.030,
    s2=0.0304,
    hpSize=21,
    smooth=11
):
    """
    Heuristic calibration using ratio percentiles, assuming inputs are already aligned.
    """
    print("\n" + "=" * 60)
    print("CALIBRATING FOCAL SPLIT PARAMETERS (pre-aligned inputs)")
    print("=" * 60)
    print(f"Expected depth range: {known_depth_range[0]:.1f}m - {known_depth_range[1]:.1f}m")
    print(f"Sensor distances: s1={s1*1000:.2f}mm, s2={s2*1000:.2f}mm")
    print(f"Δs = {abs(s2-s1)*1000:.2f}mm")

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        gray1 = img1.astype(np.float64) / 255.0

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        gray2 = img2.astype(np.float64) / 255.0

    def remove_bg(img, K):
        return img - cv2.boxFilter(img, -1, (K, K))

    gray1_clean = cv2.GaussianBlur(remove_bg(gray1, hpSize), (smooth, smooth), 0)
    gray2_clean = cv2.GaussianBlur(remove_bg(gray2, hpSize), (smooth, smooth), 0)

    I_avg = (gray1_clean + gray2_clean) / 2.0
    I_s = gray1_clean - gray2_clean
    lap = cv2.Laplacian(I_avg, cv2.CV_64F, ksize=3)

    I_s_f = cv2.boxFilter(I_s, -1, (21, 21))
    lap_f = cv2.boxFilter(lap, -1, (21, 21))

    eps = 1e-6
    good = np.abs(lap_f) > eps

    ratio = np.zeros_like(I_s_f)
    ratio[good] = I_s_f[good] / (lap_f[good] + eps)

    ratio_good = ratio[good & np.isfinite(ratio)]
    if ratio_good.size == 0:
        print("Warning: No valid ratios found, Using default parameters.")
        return 1.0, 0.0

    r10 = np.percentile(ratio_good, 10)
    r90 = np.percentile(ratio_good, 90)

    zmin, zmax = known_depth_range

    b_cal = (zmax * r10 - zmin * r90) / (zmin - zmax)
    a_cal = zmax * (b_cal + r10)

    print("\nCalibration complete:")
    print(f"  Ratio range (10th-90th percentile): [{r10:.6f}, {r90:.6f}]")
    print(f"  Calibrated a: {a_cal:.6f}")
    print(f"  Calibrated b: {b_cal:.6f}")
    print("=" * 60 + "\n")

    return a_cal, b_cal


def main():
    img1 = cv2.imread("image1.png")
    img2 = cv2.imread("image2.png")

    if img1 is None or img2 is None:
        print("Error: Could not load images 'image1.png' and 'image2.png'")
        return None, None

    print("Loaded images successfully")
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")

    s1 = 0.030
    s2 = 0.0304

    a_cal, b_cal = calibrate_focal_split(
        img1, img2,
        known_depth_range=(0.4, 1.0),
        s1=s1, s2=s2,
        hpSize=21,
        smooth=11
    )

    focal_split = FocalSplit(a=a_cal, b=b_cal, s1=s1, s2=s2)

    depth, confidence = focal_split.estimate_depth(
        img1, img2,
        hpSize=21,
        smooth=11,
        depth_min=0.4,
        depth_max=1.0,
        conf_percentile=0
    )

    valid_depth = depth[np.isfinite(depth)]
    if valid_depth.size > 0:
        print("\nDepth Statistics:")
        print(f"  Min:    {np.min(valid_depth):.3f} m")
        print(f"  Max:    {np.max(valid_depth):.3f} m")
        print(f"  Median: {np.median(valid_depth):.3f} m")
        print(f"  Mean:   {np.mean(valid_depth):.3f} m")
        print(f"  Std:    {np.std(valid_depth):.3f} m")

    focal_split.visualize_depth(depth, confidence, depth_min=0.4, depth_max=1.0)
    return depth, confidence


if __name__ == "__main__":
    depth, confidence = main()
