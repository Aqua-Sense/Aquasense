import pickle
import onnx
from onnx import OperatorSetIdProto
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
initial_type = [("float_input", FloatTensorType([None, 5]))]
onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)
with open("water_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ water_model.onnx created")

model = onnx.load("water_model.onnx")
has_ml = any(o.domain == "ai.onnx.ml" for o in model.opset_import)
if not has_ml:
    ml_opset = OperatorSetIdProto(domain="ai.onnx.ml", version=2)
    model.opset_import.append(ml_opset)
    print("→ Added ai.onnx.ml::2")

has_onnx = any(o.domain == "ai.onnx" for o in model.opset_import)
if not has_onnx:
    onnx_opset = OperatorSetIdProto(domain="ai.onnx", version=13)
    model.opset_import.insert(0, onnx_opset)
    print("→ Added ai.onnx::13")

onnx.save(model, "water_model.onnx")
print("✅ opset_import fixed in water_model.onnx")

quantize_dynamic(
    model_input="water_model.onnx",
    model_output="water_model_quant.onnx",
    weight_type=QuantType.QInt8
)
print("✅ water_model_quant.onnx created")
