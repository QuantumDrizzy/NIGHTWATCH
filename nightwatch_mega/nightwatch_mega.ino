#include <Wire.h>
#include <Servo.h>
#include <TinyGPS++.h>

// --- Configuración de Pines ---
#define PIN_SERVO_AZ 9
#define PIN_SERVO_ALT 10

// --- Objetos ---
TinyGPSPlus gps;
Servo servoAz;
Servo servoAlt;

// --- Variables de Estado (IMU) ---
const int MPU_ADDR = 0x68;
float roll = 0, pitch = 0, yaw = 0;
float accX, accY, accZ;

// --- Variables de Estado (Navegación) ---
double lat = 0.0, lon = 0.0, alt = 0.0;

// --- Variables de Estado (Slew-to-Cue) ---
float targetAz = 90.0;
float targetAlt = 90.0;
float currentAz = 90.0;
float currentAlt = 90.0;

// --- Control de Tiempos ---
unsigned long lastAttTime = 0;
unsigned long lastSlewTime = 0;

void setup() {
  // Inicializar puertos Serial
  Serial.begin(115200);   // Comunicación con Python (Serial0)
  Serial1.begin(9600);    // Comunicación con GPS NEO-6M (Serial1)
  
  // Inicializar Servos
  servoAz.attach(PIN_SERVO_AZ);
  servoAlt.attach(PIN_SERVO_ALT);
  servoAz.write((int)currentAz);
  servoAlt.write((int)currentAlt);

  // Inicializar MPU6050
  Wire.begin();
  Wire.setClock(400000); // I2C a 400kHz para lecturas rápidas
  
  // Despertar MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Serial.println("$SYS,NIGHTWATCH_MEGA_READY*00");
}

void loop() {
  // 1. Lectura Asíncrona del GPS (byte a byte, sin bloquear)
  while (Serial1.available() > 0) {
    gps.encode(Serial1.read());
  }

  // 2. Lectura Asíncrona de Comandos desde Python ($SLEW,az,alt)
  readSerialCommands();

  // 3. Interpolación Suave de Servos (Slew-to-Cue)
  updateServos();

  // 4. Envío de Telemetría de Actitud ($ATT) a 1 Hz
  unsigned long now = millis();
  if (now - lastAttTime >= 1000) {
    lastAttTime = now;
    readIMU();
    
    if (gps.location.isValid()) {
      lat = gps.location.lat();
      lon = gps.location.lng();
    }
    if (gps.altitude.isValid()) {
      alt = gps.altitude.meters();
    }

    sendAttitudePacket();
  }
}

// --- Subsistema IMU ---
void readIMU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Registro inicial de acelerómetro
  if (Wire.endTransmission(false) != 0) return; // Timeout o error
  
  Wire.requestFrom(MPU_ADDR, 6, true);
  if (Wire.available() == 6) {
    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    
    // Cálculo aproximado de Pitch y Roll desde Acelerómetro
    accX = ax / 16384.0;
    accY = ay / 16384.0;
    accZ = az / 16384.0;
    
    roll  = atan2(accY, accZ) * 180.0 / PI;
    pitch = atan2(-accX, sqrt(accY * accY + accZ * accZ)) * 180.0 / PI;
  }
}

// --- Subsistema de Protocolo ---
void sendAttitudePacket() {
  // Formato: $ATT,roll,pitch,yaw,lat,lon,alt*CS
  char buffer[100];
  snprintf(buffer, sizeof(buffer), "$ATT,%.2f,%.2f,%.2f,%.6f,%.6f,%.1f*", 
           roll, pitch, yaw, lat, lon, alt);
  
  // Cálculo de Checksum NMEA
  byte cs = 0;
  for (int i = 1; buffer[i] != '*'; i++) {
    cs ^= buffer[i];
  }
  
  Serial.print(buffer);
  if (cs < 16) Serial.print("0");
  Serial.println(cs, HEX);
}

void readSerialCommands() {
  static char cmdBuffer[64];
  static byte cmdIndex = 0;
  
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = '\0';
        parseCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else {
      if (cmdIndex < 63) {
        cmdBuffer[cmdIndex++] = c;
      }
    }
  }
}

void parseCommand(char* cmd) {
  // Ejemplo: $SLEW,180.5,45.2
  if (strncmp(cmd, "$SLEW,", 6) == 0) {
    char* comma1 = strchr(cmd + 6, ',');
    if (comma1) {
      *comma1 = '\0';
      targetAz = atof(cmd + 6);
      targetAlt = atof(comma1 + 1);
      
      // Limitar rangos físicos de los servos (ej: 0-180 grados)
      if(targetAz < 0) targetAz = 0; if(targetAz > 180) targetAz = 180;
      if(targetAlt < 0) targetAlt = 0; if(targetAlt > 180) targetAlt = 180;
    }
  }
}

// --- Subsistema Slew-to-Cue (Interpolación) ---
void updateServos() {
  unsigned long now = millis();
  if (now - lastSlewTime >= 15) { // 60Hz update rate para movimiento suave
    lastSlewTime = now;
    
    // Movimiento proporcional (suavizado)
    float diffAz = targetAz - currentAz;
    float diffAlt = targetAlt - currentAlt;
    
    if (abs(diffAz) > 0.1) currentAz += diffAz * 0.1;
    if (abs(diffAlt) > 0.1) currentAlt += diffAlt * 0.1;
    
    servoAz.write((int)currentAz);
    servoAlt.write((int)currentAlt);
  }
}
