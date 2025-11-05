# streamlit_demo/export_quantize.py
"""
Usage:
    python export_quantize.py path/to/model.onnx path/to/model.quant.onnx
"""
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_quantize.py input.onnx output.quant.onnx")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    quantize_dynamic(inp, outp, weight_type=QuantType.QInt8)
    print("Quantized model saved to", outp)
