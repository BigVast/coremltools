import onnxmltools
import coremltools
from onnx_coreml import convert

# Load a Core ML model
coreml_model = coremltools.utils.load_spec('D:/1.mlmodel')

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model, target_opset=7)
# tar_opset max = 9

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'D:/1_opset7.onnx')
