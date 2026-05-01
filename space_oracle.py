import os
import time
import requests
from skyfield.api import EarthSatellite, load, wgs84

class SpaceOracle:
    def __init__(self, cache_file="tle_cache.txt"):
        self.cache_file = cache_file
        self.satellites = []
        self.ts = load.timescale()
        
        # Coordenadas aproximadas de observacion en el monte (ej: Madrid)
        # Esto se ajustara con GPS de la Jetson despues, pero para la Fase 5 sirve hardcodeado
        self.observer = wgs84.latlon(40.4168, -3.7038)
        self._load_or_download_tles()
        
    def update_observer(self, lat, lon):
        self.observer = wgs84.latlon(lat, lon)
        print(f"[SPACE ORACLE] Observador actualizado: Lat {lat:.4f}, Lon {lon:.4f}")
        
    def _load_or_download_tles(self):
        # Chequear si cache existe y si tiene menos de 7 dias
        download_needed = True
        if os.path.exists(self.cache_file):
            file_age_days = (time.time() - os.path.getmtime(self.cache_file)) / (3600 * 24)
            if file_age_days < 7.0:
                print(f"[SPACE ORACLE] TLE cache encontrado. Edad: {file_age_days:.1f} dias. Cargando offline...")
                download_needed = False
            else:
                print(f"[SPACE ORACLE] TLE cache expirado ({file_age_days:.1f} dias). Renovando...")
        
        if download_needed:
            self._download_tles()
            
        self._parse_tles()

    def _download_tles(self):
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        print(f"[SPACE ORACLE] Conectando con Celestrak: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            with open(self.cache_file, "w") as f:
                f.write(response.text)
            print("[SPACE ORACLE] Base de datos TLE descargada y cacheada con exito.")
        except Exception as e:
            print(f"[SPACE ORACLE] ERROR critico al descargar TLEs: {e}")
            if os.path.exists(self.cache_file):
                print("[SPACE ORACLE] Usando cache antiguo por fallo de red.")
            else:
                raise Exception("Sin cache local y sin internet. Imposible operar el Oraculo.")

    def _parse_tles(self):
        # Skyfield loader para TLEs
        self.satellites = load.tle_file(self.cache_file)
        print(f"[SPACE ORACLE] Catálogo montado: {len(self.satellites)} satelites activos listos para correlacion.")

    def find_match(self, az_deg, alt_deg, tolerance_deg=2.0):
        # Nota: En el hardware real, el Kinect medira la posicion (x,y) en pixeles
        # y la Matriz de Calibracion lo pasara a Azimut/Elevacion.
        # Por ahora simulamos la entrada en azimut y elevacion.
        
        t = self.ts.now()
        
        best_match = None
        best_dist = float('inf')
        
        # Filtro burdo: En un entorno real se usaria un arbol KD o filtrado por plano orbital,
        # pero para demostrar el concepto, buscaremos la distancia angular menor.
        for sat in self.satellites:
            try:
                difference = sat - self.observer
                topocentric = difference.at(t)
                alt, az, distance = topocentric.altaz()
                
                # Descartar los que esten bajo el horizonte rapido
                if alt.degrees < 10:
                    continue
                    
                dist_az = abs(az.degrees - az_deg)
                dist_alt = abs(alt.degrees - alt_deg)
                
                # Distancia euclidea simple (aproximacion)
                dist = (dist_az**2 + dist_alt**2)**0.5
                
                if dist < best_dist and dist < tolerance_deg:
                    best_dist = dist
                    best_match = sat
            except:
                pass
                
        if best_match:
            return best_match.name, best_dist
        return None, None

if __name__ == "__main__":
    oracle = SpaceOracle()
    # Prueba de concepto con una posicion arbitraria
    print("Test buscando satelite en Az=180, Alt=45...")
    name, dist = oracle.find_match(180.0, 45.0, tolerance_deg=5.0)
    if name:
        print(f"Match TLE encontrado: {name} (Desviacion: {dist:.2f} grados)")
    else:
        print("No hay catalogados cerca (Posible UAP o Basura espacial).")
