import os
import pickle
import cv2
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

WORKING_RANGE = np.linspace(0.4 + 4e-7 * 0, 0.4 + 4e-7 * 2400000, 193)
HEATMAP_RANGE = [
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
]
highpass_filter_size = 21
smooth_kernel = (11, 11)
lap_kernel = 11


def getImageWindow(img, position):
    """Crop window from img."""
    return img[
        position[0, 0]: position[0, 1],
        position[1, 0]: position[1, 1],
    ]


def removeLowFreqInfo(img, ksize):
    """High-pass filter."""
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias


def plotSingleResult(Zkf, Ztrue, pathname, title=None):
    """Plot 2D histogram (heatmap) of true vs predicted depth."""
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    heatmap, xedges, yedges = np.histogram2d(
        Ztrue.flatten(),
        Zkf.flatten(),
        bins=len(WORKING_RANGE),
        range=HEATMAP_RANGE,
    )
    
    ax.plot(WORKING_RANGE, WORKING_RANGE, "w", alpha=0.5, linewidth=2, label='Ideal')
    
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(heatmap, extent=extent, origin="lower", cmap='hot')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Predicted Depth (m)")
    if title is not None:
        ax.set_title(title)
    
    fig.tight_layout()
    plt.savefig(pathname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved heatmap to '{pathname}'")
    
    return heatmap


def filterResultByConfidence(ZArray, ZConfidence, confidence_level=0.95):
    """Filter depth array by confidence threshold."""
    if confidence_level == 0:
        return np.copy(ZArray)
    
    ZConfidence_ = ZConfidence.flatten()
    ZConfidence_ = ZConfidence_[ZConfidence_ < np.inf]
    
    ZConfidence_f = np.where(ZConfidence < np.inf, ZConfidence, np.nan)
    
    sortZkfConfidence = np.sort(ZConfidence_)
    confidenceLevel = sortZkfConfidence[
        int((len(sortZkfConfidence) - 1) * confidence_level)
    ]
    
    print(f"  Confidence threshold ({confidence_level*100:.0f}%): {confidenceLevel:.2e}")
    
    ZArray_ = np.where(
        (ZArray < WORKING_RANGE.max()) & (ZArray > WORKING_RANGE.min()),
        ZArray,
        np.nan,
    )
    
    return np.where(ZConfidence_f > confidenceLevel, ZArray_, np.nan)


def plotWorkingArea(errors, output_path):
    """Plot predicted depth vs true depth with error bands."""
    font = {"size": 14}
    matplotlib.rc("font", **font)
    
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    plot_range = WORKING_RANGE
    
    ax.set_xlim(WORKING_RANGE.min(), WORKING_RANGE.max())
    ax.set_ylim(WORKING_RANGE.min(), WORKING_RANGE.max())
    
    ax.plot(plot_range, plot_range, linewidth=2, color="black", label="Ideal")
    ax.plot(
        plot_range, plot_range * 1.1, 
        linewidth=1, color="black", linestyle="dashed", alpha=0.5, label="±10%"
    )
    ax.plot(
        plot_range, plot_range * 0.9, 
        linewidth=1, color="black", linestyle="dashed", alpha=0.5
    )
    
    colors = ["red", "green", "blue", "brown", "pink", "purple", 
              "orange", "cyan", "magenta", "yellow"]
    
    for i, (confidenceLevel, current_error, current_meanDepth, current_meanDifference) in enumerate(errors):
        sort_index = np.argsort(WORKING_RANGE)
        sorted_working_range = WORKING_RANGE[sort_index]
        sorted_current_meanDepth = current_meanDepth[sort_index]
        sorted_meanDifference = current_meanDifference[sort_index]
        
        points_within_10pct = len(WORKING_RANGE[current_error < WORKING_RANGE * 0.1])
        print(f"  Confidence {confidenceLevel*100:.0f}%: {points_within_10pct}/{len(WORKING_RANGE)} points within 10% error")
        
        label = f"Conf={confidenceLevel*100:.0f}%" if confidenceLevel > 0 else "No filter"
        ax.plot(
            sorted_working_range,
            sorted_current_meanDepth,
            linewidth=2,
            color=colors[i % len(colors)],
            label=label
        )
        
        ax.fill_between(
            sorted_working_range,
            (sorted_current_meanDepth + sorted_meanDifference),
            (sorted_current_meanDepth - sorted_meanDifference),
            color=colors[i % len(colors)],
            alpha=0.2,
        )
    
    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Predicted Depth (m)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved working area plot to '{output_path}'")


def analyze_pkl_dataset(pkl_path, A, B, kernelSize=21, output_dir="analysis_results"):
    """
    Analyze entire PKL dataset and generate accuracy plots.
    
    Args:
        pkl_path: Path to PKL file
        A: Calibrated A parameter
        B: Calibrated B parameter
        kernelSize: Patch size
        output_dir: Output directory
        
    Returns:
        errors: List of [confidence_level, MAE, mean_depth, mean_difference]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ANALYZING PKL DATASET")
    print("="*60)
    print(f"Dataset: {pkl_path}")
    print(f"Parameters: A={A:.6f}, B={B:.6f}")
    print(f"Kernel size: {kernelSize}")
    
    texture_position = np.array([[100, 480], [150, 530]], dtype=np.int64)
    
    dataDicts = pickle.load(open(pkl_path, "rb"))
    print(f"Loaded {len(dataDicts)} samples")
    
    List_Z_pred = []
    List_Z_true = []
    List_Confidence = []
    
    print("\nProcessing samples...")
    for i, sample in enumerate(dataDicts):
        loc = sample[0]["Loc"]
        images = np.array([x["Img"] for x in sample]).astype(np.float32)
        
        imgrhoPlus = images[0]
        imgrhoMinus = images[1]
        img = (imgrhoPlus + imgrhoMinus) / 2
        
        imgrhoPlus = cv2.cvtColor(imgrhoPlus, cv2.COLOR_BGR2GRAY)
        imgrhoMinus = cv2.cvtColor(imgrhoMinus, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = getImageWindow(img, texture_position)
        imgrhoPlus = getImageWindow(imgrhoPlus, texture_position)
        imgrhoMinus = getImageWindow(imgrhoMinus, texture_position)
        
        img = removeLowFreqInfo(img, highpass_filter_size)
        imgrhoPlus = removeLowFreqInfo(imgrhoPlus, highpass_filter_size)
        imgrhoMinus = removeLowFreqInfo(imgrhoMinus, highpass_filter_size)

        img = cv2.GaussianBlur(img, smooth_kernel, 0)
        imgrhoPlus = cv2.GaussianBlur(imgrhoPlus, smooth_kernel, 0)
        imgrhoMinus = cv2.GaussianBlur(imgrhoMinus, smooth_kernel, 0)
        
        Laplacian_I = img - cv2.GaussianBlur(img, (lap_kernel, lap_kernel), 0)
        I_s_t = (imgrhoPlus - imgrhoMinus) / 2
        
        V = Laplacian_I
        W = A * Laplacian_I + B * I_s_t
        
        if kernelSize > 1:
            kernel = np.ones((kernelSize, 1))
            VW = signal.convolve2d(V * W, kernel, "same", "symm")
            VW = signal.convolve2d(VW, kernel.T, "same", "symm")
            W2 = signal.convolve2d(W**2, kernel, "same", "symm")
            W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
            Z_pred = np.divide(VW, W2, out=np.zeros_like(V), where=W2 != 0)
        else:
            Z_pred = np.divide(V, W, out=np.zeros_like(V), where=W != 0)
        
        depth = 0.4 + loc * 4e-7
        Z_true = np.full(Z_pred.shape, depth)
        Confidence = I_s_t**2

        List_Z_pred.append(Z_pred[21:-21, 21:-21])
        List_Z_true.append(Z_true[21:-21, 21:-21])
        List_Confidence.append(Confidence[21:-21, 21:-21])
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(dataDicts)} samples processed")
    
    List_Z_pred = np.array(List_Z_pred)
    List_Z_true = np.array(List_Z_true)
    List_Confidence = np.array(List_Confidence)
    

    print("\nGenerating heatmap...")
    plotSingleResult(
        List_Z_pred,
        List_Z_true,
        os.path.join(output_dir, "heatmap.png"),
        "Depth Estimation Heatmap"
    )
    

    print("\nComputing error statistics...")
    confidenceLevels = [0, 0.5, 0.7, 0.9, 0.95]
    errors = []
    
    for confidenceLevel in confidenceLevels:
        filteredZMaps = filterResultByConfidence(
            List_Z_pred, List_Confidence, confidenceLevel
        )
        
        current_error = np.nanmean(
            np.abs(filteredZMaps - List_Z_true),
            axis=(-1, -2),
        )
        current_meanDepth = np.nanmean(
            filteredZMaps.reshape(filteredZMaps.shape[0], -1),
            axis=-1,
        )
        current_meanDifference = np.nanmean(
            np.abs(
                filteredZMaps.reshape(filteredZMaps.shape[0], -1)
                - np.repeat(
                    current_meanDepth,
                    filteredZMaps.shape[1] * filteredZMaps.shape[2],
                ).reshape(filteredZMaps.shape[0], -1)
            ),
            axis=-1,
        )
        errors.append([
            confidenceLevel,
            current_error,
            current_meanDepth,
            current_meanDifference,
        ])
        
        mae = np.nanmean(np.abs(filteredZMaps - List_Z_true))
        print(f"  Confidence {confidenceLevel*100:.0f}%: MAE = {mae:.6f} m")
    

    print("\nGenerating working area plot...")
    plotWorkingArea(errors, os.path.join(output_dir, "working_area.png"))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to '{output_dir}/'")
    
    return errors


def main():
    """Analyze PKL dataset with calibrated parameters."""
    
    pkl_path = "saved_list_20250321_36d5_far.pkl"
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found!")
        return
    

    if os.path.exists('calibrated_AB_parameters.pkl'):
        print("Loading calibrated parameters...")
        with open('calibrated_AB_parameters.pkl', 'rb') as f:
            params = pickle.load(f)
        A = params['A']
        B = params['B']
        print(f"  A = {A:.6f}")
        print(f"  B = {B:.6f}")
    else:
        print("Warning: No calibrated parameters found, using defaults")
        print("Run calibrate_AB_parameters.py first!")
        A = 1.23
        B = 0.19
    

    errors = analyze_pkl_dataset(
        pkl_path,
        A, B,
        kernelSize=21,
        output_dir="analysis_results"
    )
    
    
    print("\nSummary:")
    print("-" * 60)
    for conf_level, _, _, _ in errors:
        filtered = filterResultByConfidence(
            np.random.randn(10, 10),  # Dummy for threshold calc
            np.random.randn(10, 10),
            conf_level
        )
        print(f"  {conf_level*100:.0f}% confidence filtering applied")
    print("-" * 60)


if __name__ == "__main__":
    main()