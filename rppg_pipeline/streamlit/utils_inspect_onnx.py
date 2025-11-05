import sys
import onnxruntime as ort

def inspect(path):
    sess = ort.InferenceSession(path)
    print("Inputs:")
    for i in sess.get_inputs():
        print("  name:", i.name, "shape:", i.shape, "type:", i.type)
    print("Outputs:")
    for o in sess.get_outputs():
        print("  name:", o.name, "shape:", o.shape, "type:", o.type)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils_inspect_onnx.py path/to/model.onnx")
    else:
        inspect(sys.argv[1])
