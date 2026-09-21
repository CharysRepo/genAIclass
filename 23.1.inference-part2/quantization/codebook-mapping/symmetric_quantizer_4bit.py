import numpy as np
from level_generators import generate_nf4_levels

class FourBitQuantizer:
    def __init__(self):
        self.abs_max = None

    def fit(self, weights: np.ndarray) -> float:
        """Calculates and stores the absolute maximum scaling factor from the weights."""
        self.abs_max = np.max(np.abs(weights))
        
        # Handle zero-weight safety edge case
        if self.abs_max == 0:
            self.abs_max = 1.0
            
        return self.abs_max

    def quantize(self, weights: np.ndarray, levels:np.ndarray) -> np.ndarray:
        """Quantizes weights into 4-bit indices (0-15) using the pre-calculated scale."""
        if self.abs_max is None:
            raise ValueError("Quantizer has no scale calibration. Run fit() before calling quantize().")
            
        # Scale weights to the maximum range
        scaled_weights = weights / self.abs_max
        
        # Map each scaled weight to the closest available value index
        diffs = np.abs(levels[:, np.newaxis] - scaled_weights.flatten())
        quantized_indices = np.argmin(diffs, axis=0).astype(np.uint8)
            
        return quantized_indices.reshape(weights.shape)

    def dequantize(self, quantized_indices: np.ndarray, levels:np.ndarray) -> np.ndarray:
        """Reconstructs approximate weights from 4-bit indices using the stored scale."""
        if self.abs_max is None:
            raise ValueError("Quantizer has no scale calibration. Run fit() before calling dequantize().")
            
        raw_floats = levels[quantized_indices]
        
        reconstructed_weights = raw_floats * self.abs_max
        return reconstructed_weights


if __name__ == "__main__":
    np.random.seed(42)
    #original_weights = np.random.randn(10000).astype(np.float32)
    original_weights = np.random.normal(loc=0.0, scale=0.2, size=10).astype(np.float32)
    #original_weights[75] = 12.5
    #original_weights[102] = -14.0

    quantizer = FourBitQuantizer()
    quantizer.fit(original_weights)

    all_items = [
        ("NF4", generate_nf4_levels())
    ]
    mse_results = {}
    for label, levels in all_items:
        print(f"\n{'='*40}\nData Type Format: {label}\n{'='*40}")
        print("Generated Quantization Levels Grid:")
        print(levels)

        print("\nOriginal Weights:")
        print(original_weights)

        four_bit_indices = quantizer.quantize(original_weights, levels)
        print(f"\n4-bit Indices: {four_bit_indices}")

        reconstructed_weights = quantizer.dequantize(four_bit_indices, levels)
        print("\nReconstructed Weights:")
        print(reconstructed_weights)

        error = np.mean((original_weights - reconstructed_weights) ** 2)
        print(f"\nAverage Quantization Error(MSE):  {error:.4f}")
        mse_results[label] = (levels, error)
