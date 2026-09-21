import numpy as np

class BlockLevelAsymmetricQuantizer:
    def __init__(self, bits: int, block_size: int = 64):
        self.bits = bits
        self.block_size = block_size
        self._qmax = (2 ** (bits - 1)) - 1
        self._qmin = -(2 ** (bits - 1))
        
        # Metadata storage per block
        self.scales = None
        self.offsets = None
        self.original_shape = None 

    def _pad_and_reshape(self, data: np.ndarray) -> np.ndarray:
        """Flattens input data, applies zero-padding to align with block_size, 
        and reshapes into a 2D array of blocks.
        """
        flat_data = data.flatten()
        pad_size = (self.block_size - (flat_data.size % self.block_size)) % self.block_size
        if pad_size > 0:
            flat_data = np.concatenate([flat_data, np.zeros(pad_size, dtype=data.dtype)])
        return flat_data.reshape(-1, self.block_size)

    def fit(self, data: np.ndarray):
        # Cache the true input shape for dequantization slicing
        self.original_shape = data.shape
        
        # Use the separate padding helper method
        blocks = self._pad_and_reshape(data)
        
        # Calculate local tracking limits per block
        min_vals = blocks.min(axis=1, keepdims=True)
        max_vals = blocks.max(axis=1, keepdims=True)
        
        # Handle zero variance blocks safely
        self.scales = (max_vals - min_vals) / (self._qmax - self._qmin)
        self.scales = np.where(self.scales == 0, 1.0, self.scales)
        
        raw_offsets = self._qmin - (min_vals / self.scales)
        self.offsets = np.clip(np.round(raw_offsets), self._qmin, self._qmax).astype(np.int32)
        
    def quantize(self, data: np.ndarray) -> np.ndarray:
        if self.scales is None or self.offsets is None:
            raise ValueError("Quantizer must be fitted before calling quantize().")
            
        # Use the separate padding helper method
        blocks = self._pad_and_reshape(data)
        
        # Quantize blocks concurrently via vectorized broadcast matrices
        q_blocks = np.round(blocks / self.scales) + self.offsets
        q_blocks = np.clip(q_blocks, self._qmin, self._qmax)
        return q_blocks.astype(np.int8 if self.bits <= 8 else np.int16)

    def dequantize(self, quantized_data: np.ndarray) -> np.ndarray:
        if self.scales is None or self.offsets is None or self.original_shape is None:
            raise ValueError("Quantizer requires parameters and original shape metadata.")
            
        # Reconstruct block values using corresponding scales/offsets
        deq_blocks = (quantized_data.astype(np.float32) - self.offsets) * self.scales
        flat_reconstructed = deq_blocks.flatten()
        
        # Crop away tracking padding elements to perfectly match original data shape
        total_elements = np.prod(self.original_shape)
        return flat_reconstructed[:total_elements].reshape(self.original_shape)

def evaluate(original_data, bits, block_size):
    print(original_data)
    quantizer = BlockLevelAsymmetricQuantizer(bits=bits, block_size=block_size)
    quantizer.fit(original_data)

    quantized_data = quantizer.quantize(original_data)
    print(f"Quantized Data:    {quantized_data}")

    reconstructed_data = quantizer.dequantize(quantized_data)
    print(f"Dequantized Data:         {reconstructed_data}")

    error = np.mean((original_data - reconstructed_data) ** 2)
    print(f"\nAverage Quantization Error(MSE):  {error:.4f}") 

if __name__ == "__main__":
    np.random.seed(42)
    bits = 4
    block_size = 32
    print("--- TEST CASE 1: Highly Skewed Data ---")
    skewed_data = np.random.uniform(0.0, 10.0, size=100000).astype(np.float32)
    evaluate(skewed_data, bits, block_size)
    
    print("--- TEST CASE 2: Balanced Data ---")
    balanced_data = np.random.uniform(-5.0, 5.0, size=100000).astype(np.float32)
    evaluate(balanced_data, bits, block_size)

    print("--- TEST CASE 3: Outlier Data ---")
    outlier_data = np.random.uniform(-10.0, 10.0, 128).astype(np.float32)
    outlier_indices = [5, 12, 18, 24, 45, 50, 58, 60, 90, 95, 102, 115]
    outlier_values = [120.0, -135.0, 140.0, -110.0, 150.0, -125.0, 130.0, -145.0, 115.0, -122.0, 138.0, -141.0]
    outlier_data[outlier_indices] = outlier_values
    evaluate(outlier_data, bits, block_size)
