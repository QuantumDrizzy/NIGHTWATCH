import json
import os
import nightwatch_db

telemetry_file = "C:/Users/Drizzy/Desktop/KINECT-NIR/telemetry.jsonl"

if os.path.exists(telemetry_file):
    print(f"Poblando BBDD desde {telemetry_file}...")
    count = 0
    with open(telemetry_file, "r") as f:
        # Leemos los ultimos 500 registros para tener datos frescos
        lines = f.readlines()[-500:]
        for line in lines:
            try:
                data = json.loads(line)
                # Simulamos la clasificacion para tener variedad
                az = data.get("az", 0)
                alt = data.get("alt", 0)
                d2 = data.get("d2", 0)
                p_det = data.get("p_det", 0)
                
                cls = "B"
                label = "Test Object"
                if d2 < 5: cls = "A"; label = "Synthetic Sat"
                elif d2 > 15: cls = "X"; label = "Anomalous Track"
                
                omega = 0.5 + (count % 10) * 0.1 # dummy omega
                
                nightwatch_db.log_contact(cls, az, alt, d2, p_det, label, omega)
                count += 1
            except:
                continue
    print(f"¡Listo! {count} registros insertados.")
else:
    print("Archivo de telemetría no encontrado.")
