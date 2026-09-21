import numpy as np

class AsymmetricQuantizer:
    def __init__(self, bits: int):
        self.bits = bits
        self._qmax = (2 ** (bits - 1)) - 1
        self._qmin = -(2 ** (bits - 1))
        self.scale = None
        self.offset = None

    def fit(self, data: np.ndarray) -> tuple[float, int]:
        """Calculates and stores the optimal scale and signed zero-point offset for the data."""
        min_val, max_val = data.min(), data.max()
        
        # Prevent division by zero if all data elements are identical
        if min_val == max_val:
            self.scale = 1.0
            self.offset = 0
            return self.scale, self.offset

        # Calculate step size (scale)
        self.scale = (max_val - min_val) / (self._qmax - self._qmin)
        
        # Calculate signed zero-point offset, round, and clamp strictly to signed integer bounds
        raw_offset = self._qmin - (min_val / self.scale)
        self.offset = int(np.clip(np.round(raw_offset), self._qmin, self._qmax))
        
        return self.scale, self.offset

    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Converts floating-point values into quantized signed integers."""
        if self.scale is None or self.offset is None:
            raise ValueError("Quantizer must be fitted on data before calling quantize().")
            
        q = np.round(data / self.scale) + self.offset
        q = np.clip(q, self._qmin, self._qmax)
        return q.astype(np.int8 if self.bits <= 8 else np.int16)

    def dequantize(self, quantized_data: np.ndarray) -> np.ndarray:
        """Reconstructs approximate float values from quantized signed integers."""
        if self.scale is None or self.offset is None:
            raise ValueError("Quantizer needs valid scale and offset parameters to dequantize data.")
            
        return (quantized_data.astype(np.float32) - self.offset) * self.scale


def evaluate(original_data, bits):
    print(original_data)
    quantizer = AsymmetricQuantizer(bits=bits)
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
    print("--- TEST CASE 1: Highly Skewed Data ---")
    skewed_data = np.random.uniform(0.0, 10.0, size=100000).astype(np.float32)
    evaluate(skewed_data, bits=bits)
    
    print("\n--- TEST CASE 2: Balanced Data ---")
    balanced_data = np.random.uniform(-5.0, 5.0, size=100000).astype(np.float32)
    evaluate(balanced_data, bits=bits)

    print("\n--- TEST CASE 3: Outlier Data ---")
    outlier_data = np.random.uniform(-10.0, 10.0, 128).astype(np.float32)
    outlier_indices = [5, 12, 18, 24, 45, 50, 58, 60, 90, 95, 102, 115]
    outlier_values = [120.0, -135.0, 140.0, -110.0, 150.0, -125.0, 130.0, -145.0, 115.0, -122.0, 138.0, -141.0]
    outlier_data[outlier_indices] = outlier_values
    evaluate(outlier_data, bits=bits)