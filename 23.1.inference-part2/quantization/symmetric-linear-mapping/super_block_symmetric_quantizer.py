import numpy as np

class SuperblockSymmetricQuantizer:
    """
    Implements a two-level hierarchical (Superblock) quantization scheme.
    Similar to Q4_K:
    - Superblock size: Large chunk (e.g., 64 or 256 elements)
    - Sub-block size: Small local chunk (e.g., 16 elements)
    """
    def __init__(self, bits: int = 4, super_block_size: int = 64, sub_block_size: int = 16):
        if super_block_size % sub_block_size != 0:
            raise ValueError("Superblock size must be a clean multiple of sub-block size.")
            
        self.bits = bits
        self.sb_size = super_block_size
        self.sub_size = sub_block_size
        self.sub_per_sb = super_block_size // sub_block_size
        
        # Target lower-bit limits for the core data (e.g., 4-bit)
        self._qmax = (2 ** (bits - 1)) - 1
        self._qmin = -(2 ** (bits - 1))
        
        # Quantization parameters to store
        self.super_scales = None  # Master scale per superblock
        self.sub_scales = None    # Local weight modifier per sub-block
        self.padding_size = 0

    def _pad_data(self, data: np.ndarray) -> np.ndarray:
        """Pads flat data to align perfectly with the large Superblock boundaries."""
        flat_data = data.flatten()
        remainder = flat_data.size % self.sb_size
        if remainder != 0:
            self.padding_size = self.sb_size - remainder
            flat_data = np.pad(flat_data, (0, self.padding_size), mode='constant', constant_values=0)
        else:
            self.padding_size = 0
        return flat_data

    def fit(self, data: np.ndarray):
        """Calculates both the coarse master scales and fine-grained sub-scales."""
        flat_data = self._pad_data(data)
        num_superblocks = flat_data.size // self.sb_size
        
        # 1. Reshape into [Num_Superblocks, Num_Subblocks_Per_Superblock, Subblock_Size]
        hierarchical_data = flat_data.reshape(num_superblocks, self.sub_per_sb, self.sub_size)
        
        # 2. Extract Superblock Master Scales (Max absolute value within the entire 256/64 elements)
        # Shape: (num_superblocks,)
        sb_max = np.max(np.abs(hierarchical_data), axis=(1, 2))
        sb_max[sb_max == 0] = 1.0
        self.super_scales = sb_max / 127.0  # Quantize master scale implicitly to 8-bit scale factor range
        
        # 3. Normalize data by master scale to isolate sub-block variations
        # Shape: (num_superblocks, sub_per_sb, sub_size)
        normalized_data = hierarchical_data / self.super_scales[:, None, None]
        
        # 4. Extract Sub-block local scale modifiers
        # Shape: (num_superblocks, sub_per_sb)
        sub_max = np.max(np.abs(normalized_data), axis=2)
        sub_max[sub_max == 0] = 1.0
        
        # Quantize sub_scales to a small integer range (e.g., 6-bit grid up to 63) to match Q4_K savings
        self.sub_scales = np.round((sub_max / self._qmax) * 16) / 16
        self.sub_scales[self.sub_scales == 0] = 1.0 / self._qmax

    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Quantizes elements to target bits using the compound hierarchical scales."""
        if self.super_scales is None or self.sub_scales is None:
            raise ValueError("Quantizer must be fitted before running quantization.")
            
        original_shape = data.shape
        flat_data = self._pad_data(data)
        num_superblocks = flat_data.size // self.sb_size
        
        hierarchical_data = flat_data.reshape(num_superblocks, self.sub_per_sb, self.sub_size)
        
        # Compute combined effective scale matrix: Shape (num_superblocks, sub_per_sb, 1)
        effective_scale = (self.super_scales[:, None] * self.sub_scales)[:, :, None]
        
        # Quantize core elements
        q = np.round(hierarchical_data / effective_scale)
        q = np.clip(q, self._qmin, self._qmax)
        
        q_flat = q.flatten()
        if self.padding_size > 0:
            q_flat = q_flat[:-self.padding_size]
            
        return q_flat.reshape(original_shape).astype(np.int8)

    def dequantize(self, quantized_data: np.ndarray) -> np.ndarray:
        """Reconstructs original floats by cascading sub-block and superblock metrics."""
        if self.super_scales is None or self.sub_scales is None:
            raise ValueError("Missing valid scale matrices for dequantization pipeline.")
            
        original_shape = quantized_data.shape
        flat_q = quantized_data.flatten()
        
        if self.padding_size > 0:
            flat_q = np.pad(flat_q, (0, self.padding_size), mode='constant', constant_values=0)
            
        num_superblocks = flat_q.size // self.sb_size
        hierarchical_q = flat_q.reshape(num_superblocks, self.sub_per_sb, self.sub_size)
        
        # Reconstruct element grid multiplying back both scale levels
        effective_scale = (self.super_scales[:, None] * self.sub_scales)[:, :, None]
        dq = hierarchical_q.astype(np.float32) * effective_scale
        
        dq_flat = dq.flatten()
        if self.padding_size > 0:
            dq_flat = dq_flat[:-self.padding_size]
            
        return dq_flat.reshape(original_shape)


def evaluate(original_data, bits, super_block_size, sub_block_size):
    print(original_data)
    quantizer = SuperblockSymmetricQuantizer(bits=bits, super_block_size=super_block_size, sub_block_size=sub_block_size)
    quantizer.fit(original_data)

    quantized_data = quantizer.quantize(original_data)
    print(f"Quantized Data:    {quantized_data}")

    reconstructed_data = quantizer.dequantize(quantized_data)
    print(f"Dequantized Data:         {reconstructed_data}")

    error = np.mean((original_data - reconstructed_data) ** 2)
    print(f"\nAverage Quantization Error(MSE):  {error:.4f}")    
    

if __name__ == "__main__":
    np.random.seed(42)
    print("--- TEST CASE 1: Highly Skewed Data ---")
    skewed_data = np.random.uniform(0.0, 10.0, size=100000).astype(np.float32)
    evaluate(skewed_data, bits=4, super_block_size=64, sub_block_size=16)
    
    print("\n--- TEST CASE 2: Balanced Data ---")
    balanced_data = np.random.uniform(-5.0, 5.0, size=100000).astype(np.float32)
    evaluate(balanced_data, bits=4, super_block_size=64, sub_block_size=16)

    print("\n--- TEST CASE 3: Outlier Data ---")
    outlier_data = np.random.uniform(-10.0, 10.0, 128).astype(np.float32)
    outlier_indices = [5, 12, 18, 24, 45, 50, 58, 60, 90, 95, 102, 115]
    outlier_values = [120.0, -135.0, 140.0, -110.0, 150.0, -125.0, 130.0, -145.0, 115.0, -122.0, 138.0, -141.0]
    outlier_data[outlier_indices] = outlier_values
    evaluate(outlier_data, bits=4, super_block_size=64, sub_block_size=16)

