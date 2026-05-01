import os
import sys
import io
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from trackformer import MobileViT_XT

os.chdir(r"C:\Users\Drizzy\Desktop\KINECT-NIR")

def export_to_onnx():
    print("[NIGHTWATCH-TRACKFORMER] Iniciando puente de soberania PyTorch -> ONNX...")
    
    # 1. Instanciar la red
    model = MobileViT_XT()
    
    # 2. Cargar los pesos entrenados
    weights_path = "nightwatch_mobilevit.pth"
    if not os.path.exists(weights_path):
        print(f"Error: No se encontro el archivo de pesos '{weights_path}'. Entrena el modelo primero.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print("[OK] Pesos cargados correctamente.")
    
    # 3. Crear el tensor de entrada "Dummy" con el shape exacto que espera C++
    # Batch=1, Channels=3, Time=16, H=64, W=64
    dummy_input = torch.randn(1, 3, 16, 64, 64)
    
    # 4. Exportar
    onnx_path = "NIGHTWATCH_MOBILEVIT_XT.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14, # Opset 14 es estable para atención y transformers en TensorRT
        do_constant_folding=True,
        input_names=['input_cube'],
        output_names=['p_det', 'coords'],
        dynamic_axes={'input_cube': {0: 'batch_size'}, 'p_det': {0: 'batch_size'}, 'coords': {0: 'batch_size'}}
    )
    
    print(f"OK: Exito! Modelo exportado a: {onnx_path}")
    print("\n[INSTRUCCIONES PARA TENSORRT]")
    print("Para compilar el .engine final, ejecuta el siguiente comando en tu consola si tienes TensorRT instalado:")
    print("trtexec --onnx=NIGHTWATCH_MOBILEVIT_XT.onnx --saveEngine=NIGHTWATCH_MOBILEVIT_XT.engine --int8")

if __name__ == "__main__":
    export_to_onnx()
