import os
import sys
import io
import torch
import numpy as np
import onnxruntime as ort
from trackformer import MobileViT_XT

# Fix encoding issues in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def validate():
    os.chdir(r"C:\Users\Drizzy\Desktop\KINECT-NIR")
    print("[NIGHTWATCH-CORE] Iniciando validacion PyTorch vs ONNX (Sanity Check)...")

    onnx_path = "NIGHTWATCH_MOBILEVIT_XT.onnx"
    weights_path = "nightwatch_mobilevit.pth"

    if not os.path.exists(onnx_path) or not os.path.exists(weights_path):
        print(f"Error: Faltan archivos. Asegurate de que {onnx_path} y {weights_path} existen.")
        return

    # 1. Cargar PyTorch Model
    print("-> Cargando modelo PyTorch...")
    pt_model = MobileViT_XT()
    pt_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    pt_model.eval()

    # 2. Cargar ONNX Runtime Session
    print("-> Cargando ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    num_tests = 100
    max_diff_det = 0.0
    max_diff_reg = 0.0

    print(f"-> Lanzando {num_tests} pruebas aleatorias...")
    
    with torch.no_grad():
        for i in range(num_tests):
            # Batch=1, Channels=3, Time=16, H=64, W=64
            dummy_input = torch.randn(1, 3, 16, 64, 64)

            # PyTorch Inference
            pt_p_det, pt_coords = pt_model(dummy_input)
            pt_p_det_np = pt_p_det.numpy()
            pt_coords_np = pt_coords.numpy()

            # ONNX Inference
            ort_inputs = {input_name: dummy_input.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            ort_p_det_np, ort_coords_np = ort_outs[0], ort_outs[1]

            # Compare
            diff_det = np.max(np.abs(pt_p_det_np - ort_p_det_np))
            diff_reg = np.max(np.abs(pt_coords_np - ort_coords_np))

            if diff_det > max_diff_det: max_diff_det = diff_det
            if diff_reg > max_diff_reg: max_diff_reg = diff_reg

    print("\n--- RESULTADOS DEL SANITY CHECK ---")
    print(f"Error Maximo Deteccion (p_det): {max_diff_det:.8f}")
    print(f"Error Maximo Regresion (coords): {max_diff_reg:.8f}")

    if max_diff_det < 1e-5 and max_diff_reg < 1e-5:
        print("\n[OK] PARIDAD MATEMATICA CONFIRMADA. El modelo ONNX es un clon perfecto.")
    else:
        print("\n[ERROR] Divergencia matematica detectada. ONNX no es equivalente.")

if __name__ == "__main__":
    validate()
