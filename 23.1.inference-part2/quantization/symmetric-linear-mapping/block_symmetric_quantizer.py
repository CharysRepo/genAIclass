import numpy as np

class BlockLevelSymmetricQuantizer:
    """Handles symmetric quantization and dequantization of numerical data in fixed-size blocks."""
    def __init__(self, bits: int, block_size: int):
        self.bits = bits
        self.block_size = block_size
        self._qmax = (2 ** (bits - 1)) - 1
        self._qmin = -(2 ** (bits - 1))
        self.scales = None
        self.padding_size = 0  # Tracks added zeros if data doesn't align with block_size

    def _pad_and_reshape(self, data: np.ndarray) -> np.ndarray:
        """Pads flat data with trailing zeros and reshapes it into block segments."""
        flat_data = data.flatten()
        remainder = flat_data.size % self.block_size
        
        if remainder != 0:
            self.padding_size = self.block_size - remainder
            flat_data = np.pad(flat_data, (0, self.padding_size), mode='constant', constant_values=0)
        else:
            self.padding_size = 0
            
        return flat_data.reshape(-1, self.block_size)

    def fit(self, data: np.ndarray) -> np.ndarray:
        """Calculates and stores symmetric scale factors for every individual data block."""
        reshaped_data = self._pad_and_reshape(data)
        
        # Calculate absolute peak value per row (block)
        abs_max = np.max(np.abs(reshaped_data), axis=1)
        
        # Prevent division by zero for completely empty blocks
        abs_max[abs_max == 0] = 1.0 * self._qmax
        
        # Scale factor vector corresponding to each block
        self.scales = abs_max / self._qmax

    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Converts float data to signed integers segmenting by its respective block scale."""
        if self.scales is None:
            raise ValueError("Quantizer must be fitted on data before calling quantize().")
            
        original_shape = data.shape
        reshaped_data = self._pad_and_reshape(data)
        
        # Line up scales along vertical axis using broadcasting axis manipulation [:, None]
        q = np.round(reshaped_data / self.scales[:, None])
        q = np.clip(q, self._qmin, self._qmax)
        
        # Flatten vector structure and un-pad if excess items were appended during fit
        q_flat = q.flatten()
        if self.padding_size > 0:
            q_flat = q_flat[:-self.padding_size]
            
        return q_flat.reshape(original_shape).astype(np.int8 if self.bits <= 8 else np.int16)

    def dequantize(self, quantized_data: np.ndarray) -> np.ndarray:
        """Reconstructs original float formats from block-quantized integer arrays."""
        if self.scales is None:
            raise ValueError("Quantizer needs valid scale factors to dequantize data.")
            
        original_shape = quantized_data.shape
        flat_q = quantized_data.flatten()
        
        # Reproduce original padding layout to maintain correct matrix positions
        if self.padding_size > 0:
            flat_q = np.pad(flat_q, (0, self.padding_size), mode='constant', constant_values=0)
            
        reshaped_q = flat_q.reshape(-1, self.block_size)
        dq = reshaped_q.astype(np.float32) * self.scales[:, None]
        
        dq_flat = dq.flatten()
        if self.padding_size > 0:
            dq_flat = dq_flat[:-self.padding_size]
            
        return dq_flat.reshape(original_shape)

def evaluate(original_data, bits, block_size):
    print(original_data)
    quantizer = BlockLevelSymmetricQuantizer(bits=bits, block_size=block_size)
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
    block_size = 64
    print("--- TEST CASE 1: Highly Skewed Data ---")
    skewed_data = np.random.uniform(0.0, 10.0, size=100000).astype(np.float32)
    evaluate(skewed_data, bits, block_size)
    
    print("\n--- TEST CASE 2: Balanced Data ---")
    balanced_data = np.random.uniform(-5.0, 5.0, size=100000).astype(np.float32)
    evaluate(balanced_data, bits, block_size)

    print("\n--- TEST CASE 3: Outlier Data ---")
    outlier_data = np.random.uniform(-10.0, 10.0, 128).astype(np.float32)
    outlier_indices = [5, 12, 18, 24, 45, 50, 58, 60, 90, 95, 102, 115]
    outlier_values = [120.0, -135.0, 140.0, -110.0, 150.0, -125.0, 130.0, -145.0, 115.0, -122.0, 138.0, -141.0]
    outlier_data[outlier_indices] = outlier_values
    evaluate(outlier_data, bits, block_size)
