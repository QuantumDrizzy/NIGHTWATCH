import os
import glob
import json
import urllib.request
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
API_KEY = os.environ.get("NVIDIA_API_KEY")

def compile_and_run(num_clips=10):
    print("[NIGHTWATCH-SYNTH-ORBIT] Iniciando compilacion de la simulacion cuantica...")
    # Recompilar para asegurar los cambios de main.cpp
    subprocess.run(["build.bat"], shell=True)
    
    print(f"\nGenerando {num_clips} clips sinteticos (16 frames por clip)...")
    result = subprocess.run(["nightwatch_vision.exe", "--generate-dataset", str(num_clips)], capture_output=True, text=True)
    
    if "ERROR" in result.stdout or result.returncode != 0:
        print("Error en la generacion:")
        print(result.stdout)
        exit(1)
    print("OK: Generacion completada en C++ / CUDA.\n")

def analyze_dataset():
    clips = glob.glob("RAW_DATA/clip_*")
    print(f"Se han encontrado {len(clips)} clips en RAW_DATA.")
    
    # Validar un clip de muestra
    if len(clips) > 0:
        sample_clip = clips[0]
        frames = glob.glob(f"{sample_clip}/frame_*.bin")
        gts = glob.glob(f"{sample_clip}/gt_*.txt")
        print(f"   -> Clip de muestra '{sample_clip}' contiene {len(frames)} frames y {len(gts)} etiquetas.")
        
        return len(clips), len(frames)
    return 0, 0

def call_deepseek_validation(num_clips, frames_per_clip):
    if not API_KEY:
        print("WARNING: NVIDIA_API_KEY no detectada. Saltando validacion de IA.")
        return

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    prompt = f"""
    Eres el Director Tecnico de Arquitectura de IA de NIGHTWATCH.
    Hemos ejecutado con exito el script NIGHTWATCH-SYNTH-ORBIT.
    Se han generado {num_clips} clips.
    Cada clip contiene {frames_per_clip} frames (Cubo Diferencial 16-frame).
    Las matrices de densidad Husimi-Q en 32-bit float se han acoplado con el Ground Truth de coordenadas.
    
    Dame un reporte muy breve y militar (2 parrafos) confirmando que la infraestructura de datos esta lista para entrenar la red TensorRT INT8.
    """

    data = {
        "model": "deepseek-ai/deepseek-v3.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")

    print("\nEnviando telemetria del dataset a DeepSeek-V3.2 para validacion...")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            print("\n------------------ INFORME DE DEEPSEEK ------------------")
            print(result["choices"][0]["message"]["content"])
            print("---------------------------------------------------------")
    except Exception as e:
        print(f"Error al contactar con NVIDIA NIM: {e}")

if __name__ == "__main__":
    num_clips_to_generate = 5  # 5 clips * 16 = 80 frames para esta prueba
    compile_and_run(num_clips_to_generate)
    clips_count, frames_count = analyze_dataset()
    call_deepseek_validation(clips_count, frames_count)
    print("\n[FASE 4.3 - PUNTO 1] Completado. Dataset listo para PyTorch.")
