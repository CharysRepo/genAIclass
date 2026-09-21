import numpy as np
from level_generators import generate_nf4_levels

class BlockFourBitQuantizer:
    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.block_scales = None
        self.original_shape = None
        self.padded_length = None

    def block_fit(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculates and stores an absolute maximum scaling factor for every block.
        Handles flat or multi-dimensional tensors seamlessly.
        """
        self.original_shape = weights.shape
        flat_weights = weights.flatten()
        n_elements = len(flat_weights)
        
        # Calculate trailing padding requirements if not divisible by block_size
        remainder = n_elements % self.block_size
        padding_needed = (self.block_size - remainder) % self.block_size
        self.padded_length = n_elements + padding_needed
        
        if padding_needed > 0:
            flat_weights = np.concatenate([flat_weights, np.zeros(padding_needed, dtype=np.float32)])
            
        # Segment array into standalone computational rows/blocks
        blocks = flat_weights.reshape(-1, self.block_size)
        
        # Extract scale coefficients across each individual block row axis
        self.block_scales = np.max(np.abs(blocks), axis=1, keepdims=True)
        
        # Handle zero-weight safety edge case per block
        self.block_scales[self.block_scales == 0] = 1.0
            
        return self.block_scales

    def block_quantize(self, weights: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """
        Quantizes weights into 4-bit indices (0-15) based on localized block scales.
        """
        if self.block_scales is None:
            raise ValueError("Quantizer has no scale calibration. Run block_fit() before calling block_quantize().")
            
        flat_weights = weights.flatten()
        padding_needed = self.padded_length - len(flat_weights)
        
        if padding_needed > 0:
            flat_weights = np.concatenate([flat_weights, np.zeros(padding_needed, dtype=np.float32)])
            
        blocks = flat_weights.reshape(-1, self.block_size)
        
        # Scale each block by its unique localized scale factor
        scaled_blocks = blocks / self.block_scales
        
        # High-speed vectorized distance calculation via NumPy broadcasting
        # Dimensions: (16 levels, number of blocks, block_size)
        diffs = np.abs(levels[:, np.newaxis, np.newaxis] - scaled_blocks)
        quantized_indices_blocks = np.argmin(diffs, axis=0).astype(np.uint8)
        
        # Flatten and strip padding out before returning index grid
        quantized_indices_flat = quantized_indices_blocks.flatten()
        if padding_needed > 0:
            quantized_indices_flat = quantized_indices_flat[:-padding_needed]
            
        return quantized_indices_flat.reshape(self.original_shape)

    def block_dequantize(self, quantized_indices: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """
        Reconstructs approximate weights from 4-bit indices using block scales.
        """
        if self.block_scales is None:
            raise ValueError("Quantizer has no scale calibration. Run block_fit() before calling block_dequantize().")
            
        flat_indices = quantized_indices.flatten()
        padding_needed = self.padded_length - len(flat_indices)
        
        if padding_needed > 0:
            # Pad indices with zero (maps safely to a valid level index)
            flat_indices = np.concatenate([flat_indices, np.zeros(padding_needed, dtype=np.uint8)])
            
        indices_blocks = flat_indices.reshape(-1, self.block_size)
        
        # Map indices back to float levels inside the block matrix
        raw_floats_blocks = levels[indices_blocks]
        
        # Rescale blocks using their respective scalar factors
        reconstructed_blocks = raw_floats_blocks * self.block_scales
        
        # Flatten and remove padding elements to match the true original layout
        reconstructed_flat = reconstructed_blocks.flatten()
        if padding_needed > 0:
            reconstructed_flat = reconstructed_flat[:-padding_needed]
            
        return reconstructed_flat.reshape(self.original_shape)

if __name__ == "__main__":
    np.random.seed(42)
    original_weights = np.random.normal(loc=0.0, scale=0.1, size=192).astype(np.float32)
    original_weights[75] = 12.5
    original_weights[102] = -14.0

    quantizer = BlockFourBitQuantizer(block_size=64)
    quantizer.block_fit(original_weights)

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

        four_bit_indices = quantizer.block_quantize(original_weights, levels)
        print(f"\n4-bit Indices: {four_bit_indices}")

        reconstructed_weights = quantizer.block_dequantize(four_bit_indices, levels)
        print("\nReconstructed Weights:")
        print(reconstructed_weights)

        error = np.mean((original_weights - reconstructed_weights) ** 2)
        print(f"\nAverage Quantization Error(MSE):  {error:.4f}")
        mse_results[label] = (levels, error)
