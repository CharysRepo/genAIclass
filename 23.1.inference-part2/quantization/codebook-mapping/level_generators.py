import numpy as np
import scipy.stats as stats

def generate_nf4_levels() -> np.ndarray:
    """Compute true NF4 quantization levels."""
    # Divide the left half [0, 0.5] into 7 equal regions (8 boundary points)
    neg_probs = np.linspace(0, 0.5, 8)
    neg_quantiles = stats.norm.ppf(neg_probs)
    neg_quantiles[0] = -3.5  # Approximate -inf boundary
    neg_levels = (neg_quantiles[:-1] + neg_quantiles[1:]) / 2

    # Divide the right half [0.5, 1.0] into 8 equal regions (9 boundary points)
    pos_probs = np.linspace(0.5, 1.0, 9)
    pos_quantiles = stats.norm.ppf(pos_probs)
    pos_quantiles[-1] = 3.5  # Approximate +inf boundary
    pos_levels = (pos_quantiles[:-1] + pos_quantiles[1:]) / 2

    # Stitch them together: 7 negative midpoints, an exact 0.0, and 8 positive midpoints
    raw_levels = np.concatenate([neg_levels, [0.0], pos_levels])
    normalized_levels = raw_levels / np.max(np.abs(raw_levels))    
    return normalized_levels.astype(np.float32)
