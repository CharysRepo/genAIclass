import numpy as np

class SymmetricQuantizer:
    def __init__(self, bits: int):
        self.bits = bits
        self._qmax = (2 ** (bits - 1)) - 1
        self._qmin = -(2 ** (bits - 1))
        self.scale = None

    def fit(self, data: np.ndarray) -> float:
        """Calculates and stores the symmetric scale factor based on input data."""
        abs_max = np.max(np.abs(data))

        if abs_max == 0:
            self.scale = 1.0
            return self.scale
        
        # Calculate step size (scale)
        self.scale = abs_max / self._qmax 

    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Converts float data to signed integers using the calculated scale."""
        if self.scale is None:
            raise ValueError("Quantizer must be fitted on data before calling quantize().")
            
        q = np.round(data / self.scale)
        q = np.clip(q, self._qmin, self._qmax)
        return q.astype(np.int8 if self.bits <= 8 else np.int16)

    def dequantize(self, quantized_data: np.ndarray) -> np.ndarray:
        """Reconstructs float values from quantized integer data."""
        if self.scale is None:
            raise ValueError("Quantizer needs a valid scale factor to dequantize data.")
            
        return quantized_data.astype(np.float32) * self.scale

def evaluate(original_data, bits):
    print(original_data)
    quantizer = SymmetricQuantizer(bits=bits)
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

    